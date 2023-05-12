from __future__ import annotations
from typing import Tuple, Dict, Union
import os

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.configurations import OptimizerConfig, KeypointNetConfig
from src.keypointnet.losses import KeypointNetLosses
from src.utils import init_backbone, initialize_config_file
from src.keypointnet.debug import debug_keypoints
from src.don.synthetizer import augment_images_and_compute_correspondences as sythetize
from src.keypointnet.geometry_transformations import kabsch_tranformation


class KeypointNetwork(pl.LightningModule):

    def __init__(self, yaml_config_path: str) -> None:
        super(KeypointNetwork, self).__init__()

        # Init. configuration
        config = initialize_config_file(yaml_config_path)
        self.keypointnet_config = KeypointNetConfig.from_dictionary(config)
        self.optim_config = OptimizerConfig.from_dictionary(config)

        self.debug_path = self.keypointnet_config.keypointnet.debug_path
        self.debug = self.keypointnet_config.keypointnet.debug

        # Modified backbone to extract conv. features
        self.backbone = init_backbone(self.keypointnet_config.keypointnet.backbone)
        self.backbone = torch.nn.Sequential(self.backbone.conv1,
                                            self.backbone.bn1,
                                            self.backbone.relu,
                                            self.backbone.layer1,
                                            self.backbone.layer2,
                                            self.backbone.layer3,
                                            self.backbone.layer4,
                                            torch.nn.Conv2d(self.backbone.inplanes,
                                                            self.keypointnet_config.keypointnet.bottleneck_dimension,
                                                            kernel_size=3))

        self.spatial_layer = torch.nn.Conv2d(self.keypointnet_config.keypointnet.bottleneck_dimension,
                                             self.keypointnet_config.keypointnet.n_keypoints,
                                             kernel_size=1,
                                             padding='same')

        # Init. Loss
        self.loss_function = KeypointNetLosses(self.keypointnet_config.loss)

        # random colors for debugging
        self.colors = [(np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255))
                       for _ in range(self.keypointnet_config.keypointnet.n_keypoints)]

    @ staticmethod
    def _compute_spatial_expectations(mask: Union[torch.Tensor, None],
                                      spatial_probs: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:

        us = torch.arange(0, mask.shape[-2], 1, dtype=torch.float32, device=mask.device)
        vs = torch.arange(0, mask.shape[-1], 1, dtype=torch.float32, device=mask.device)
        grid = torch.meshgrid(us, vs, indexing='ij')

        exp_u = torch.sum(spatial_probs * grid[0], dim=(-2, -1))
        exp_v = torch.sum(spatial_probs * grid[1], dim=(-2, -1))

        if mask is not None:
            exp_m = torch.sum(spatial_probs * mask.unsqueeze(dim=1), dim=(-2, -1))
            return (torch.stack([exp_u, exp_v], dim=-1), exp_m)
        else:
            return (torch.stack([exp_u, exp_v], dim=-1), None)

    @staticmethod
    def _upsample(x: torch.Tensor, ref_x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x,
                             size=ref_x.size()[-2:],
                             mode='bilinear',
                             align_corners=True)

    def _forward(self,
                 image: torch.Tensor,
                 masks: Union[torch.Tensor, None]) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:

        backbone_features = self.backbone.forward(image)
        spatial_weights = self.spatial_layer.forward(backbone_features)
        spatial_weights = self._upsample(spatial_weights, image)

        flat_weights = spatial_weights.reshape(spatial_weights.shape[0],
                                               spatial_weights.shape[1],
                                               spatial_weights.shape[2] * spatial_weights.shape[3])

        flat_probs = F.softmax(flat_weights, dim=-1)
        spatial_probs = flat_probs.reshape(flat_probs.shape[0],
                                           flat_probs.shape[1],
                                           image.shape[2],
                                           image.shape[3])

        exp_uvs, exp_mask = self._compute_spatial_expectations(masks, spatial_probs)

        return spatial_probs, exp_uvs, exp_mask

    def configure_optimizers(self):

        if self.optim_config.name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.optim_config.learning_rate,
                                         weight_decay=self.optim_config.weight_decay)

        if self.optim_config.enable_schedule:
            sch = torch.optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.optim_config.schedular_step_size,
                                                  gamma=self.optim_config.gamma,
                                                  verbose=False)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch}
            }
        else:
            return optimizer

    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image, mask, backgrounds = batch["RGBs-A"], batch["Masks-A"], batch["Random-Backgrounds"]
        images_a, matches_a, masks_a, images_b, matches_b, masks_b = sythetize(image,
                                                                               mask,
                                                                               backgrounds,
                                                                               n_correspondences=25,
                                                                               enable_prespective_augmentation=False)

        optimal_rotation_a_to_b, optimal_translation_a_to_b = kabsch_tranformation(matches_a, matches_b)

        spat_probs_a, exp_uv_a, exp_mask_a = self._forward(images_a, masks_a)
        spat_probs_b, exp_uv_b, exp_mask_b = self._forward(images_b, masks_b)

        computed_data = {"Spatial-Probs-A": spat_probs_a,
                         "Spatial-Probs-B": spat_probs_b,
                         "Spatial-Expectations-A": exp_uv_a,
                         "Spatial-Expectations-B": exp_uv_b,
                         "Spatial-Masks-A": exp_mask_a,
                         "Spatial-Masks-B": exp_mask_b,
                         "Optimal-Rotation-A2B": optimal_rotation_a_to_b,
                         "Optimal-Translation-A2B": optimal_translation_a_to_b}

        if (self.debug or self.trainer.current_epoch == self.trainer.max_epochs - 1 or self.trainer.validating):
            debug_keypoints(images_a,
                            exp_uv_a,
                            images_b,
                            exp_uv_b,
                            self.colors,
                            self.debug_path)

        return self.loss_function(computed_data)

    def detail_log(self, loss: Dict[str, torch.Tensor], placeholder_name: str):
        _loss = loss.copy()
        _loss.pop("Total", None)
        self.logger.experiment.add_scalars(placeholder_name, _loss, self.trainer.global_step)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss["Total"])
        self.detail_log(loss, "training loss")

        if self.optim_config.enable_schedule and self.trainer.current_epoch != 0:
            sch = self.lr_schedulers()
            sch.step()

        return loss["Total"]

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss["Total"])
        self.detail_log(loss, "validation loss")
        return loss["Total"]

    def compute_dense_local_descriptors(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            x = self.backbone.forward(image)

        return F.interpolate(x,
                             size=image.size()[-2:],
                             mode='bilinear',
                             align_corners=True)


def load_trained_don_model(yaml_config_path: str, trained_model_path: Union[str, None] = None) -> KeypointNetwork:

    if trained_model_path is None:
        config = initialize_config_file(yaml_config_path)
        trained_model_name = config["trainer"]["model_checkpoint_name"] + ".ckpt"
        trained_model_path = os.path.join(config["trainer"]["model_path"], trained_model_name)

    model = KeypointNetwork(yaml_config_path)
    trained_model = model.load_from_checkpoint(trained_model_path, yaml_config_path=yaml_config_path)

    return trained_model
