from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.configurations import OptimizerConfig, KeypointNetConfig
from src.keypointnet.losses import KeypointNetLosses
from src.utils import init_backbone, initialize_config_file
from src.don.synthetizer import augment_images_and_compute_correspondences as synthetize


class KeypointNetwork(pl.LightningModule):

    def __init__(self, yaml_config_path: str) -> None:
        super(KeypointNetwork, self).__init__()

        # Init. configuration
        config = initialize_config_file(yaml_config_path)
        self.keypointnet_config = KeypointNetConfig.from_dictionary(config)
        self.optim_config = OptimizerConfig.from_dictionary(config)

        self.backbone = init_backbone(self.keypointnet_config.keypointnet.backbone)
        self.backbone.fc = torch.nn.Conv2d(self.backbone.inplanes,
                                           self.keypointnet_config.keypointnet.bottleneck_dimension,
                                           kernel_stride=3)

        self.spatial_layer = torch.nn.Conv2d(self.keypointnet_config.keypointnet.bottleneck_dimension,
                                             self.keypointnet_config.keypointnet.n_keypoints,
                                             kernel_size=3,
                                             padding='same')

        self.depth_layer = torch.nn.Conv2d(self.keypointnet_config.keypointnet.bottleneck_dimension,
                                           self.keypointnet_config.keypointnet.n_keypoints,
                                           kernel_size=3,
                                           padding='same')

        # Init. Loss
        self.loss_function = KeypointNetLosses(self.keypointnet_config.loss)

        # random colors for debugging
        self.colors = [(np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255))
                       for _ in range(self.keypointnet_config.keypointnet.n_keypoints)]

    @staticmethod
    def _compute_spatial_expectations(depth: torch.Tensor,
                                      mask: torch.Tensor,
                                      spatial_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        us = torch.arange(0, depth.shape[-2], 1, dtype=torch.float32, device=depth.device)
        vs = torch.arange(0, depth.shape[-1], 1, dtype=torch.float32, device=depth.device)
        grid = torch.meshgrid(us, vs, indexing='ij')

        number_of_keypoints = spatial_probs.shape[0]

        exp_u = torch.sum(spatial_probs * grid[0].unsqueeze(dim=0).tile(number_of_keypoints, 1, 1), dim=(-2, -1))
        exp_v = torch.sum(spatial_probs * grid[1].unsqueeze(dim=0).tile(number_of_keypoints, 1, 1), dim=(-2, -1))
        exp_d = torch.sum(spatial_probs * depth, dim=(-2, -1))
        exp_m = torch.sum(spatial_probs * mask, dim=(-2, -1))

        return torch.hstack([exp_u, exp_v, exp_d]), exp_m

    def _forward(self,
                 input: torch.Tensor,
                 masks: torch.Tensor) -> torch.Tensor:

        x = self.backbone.forward(input)
        resized_x = F.interpolate(x,
                                  size=input.size()[-2:],
                                  mode='bilinear',
                                  align_corners=True)

        spatial_weights = self.spatial_layer.forward(resized_x)
        flat_weights = spatial_weights.reshape(spatial_weights.shape[0],
                                               spatial_weights.shape[1],
                                               spatial_weights.shape[2] * spatial_weights.shape[3])

        flat_probs = F.softmax(flat_weights, dim=-1)
        spatial_probs = flat_probs.reshape(flat_probs.shape[0],
                                           flat_probs.shape[1],
                                           input.shape[2],
                                           input.shape[3])

        depth = torch.relu(self.depth_layer.forward(resized_x)) + 1e-12

        exp_uvd, exp_mask = self._compute_spatial_expectations(depth, masks, spatial_probs)

        return spatial_probs, exp_uvd, exp_mask

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
                "lr_scheduler": {
                    "scheduler": sch
                }
            }
        else:
            return optimizer

    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rgb_a, rgb_b = batch["RGBs-A"], batch["RGBs-B"]
        mask_a, mask_b = batch["Masks-A"], batch["Masks-B"]

        spat_probs_a, exp_uvd_a, exp_mask_a = self._forward(rgb_a, mask_a)
        spat_probs_b, exp_uvd_b, exp_mask_b = self._forward(rgb_b, mask_b)

        computated_data = {"Spatial-Probs-A": spat_probs_a,
                           "Spatial-Probs-B": spat_probs_b,
                           "Spatial-Expectations-A": exp_uvd_a,
                           "Spatial-Expectations-B": exp_uvd_b,
                           "Spatial-Masks-A": exp_mask_a,
                           "Spatial-Masks-B": exp_mask_b}

        return self.loss_function(batch.update(computated_data))

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("training_loss", {**loss})

        if self.optim_config.enable_schedule:
            sch = self.lr_schedulers()
            sch.step()

        return loss["Total"]

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("validation_loss", {**loss})

        return loss["Total"]

    def validation_epoch_end(self, outputs):
        if self.keypointnet_config.loss.reduction == 'sum':
            loss = torch.sum(torch.hstack(outputs))
        else:
            loss = torch.mean(torch.hstack(outputs))
        self.log("val_loss", loss)

        return loss

    def get_dense_representation(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.resnet.forward(input)
            return F.interpolate(x,
                                 size=input.size()[-2:],
                                 mode='bilinear',
                                 align_corners=True)
