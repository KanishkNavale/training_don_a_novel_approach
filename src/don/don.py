from typing import Union
import os

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from src.don.synthetizer import augment_images_and_compute_correspondences as synthetize
from src.don.debug import debug_correspondences, debug_descriptors
from src.configurations import DONConfig, OptimizerConfig
from src.don.losses import PixelwiseCorrespondenceLoss, PixelwiseNTXentLoss
from src.utils import init_backbone, initialize_config_file


class DON(pl.LightningModule):

    def __init__(self, yaml_config_path: str) -> None:
        super(DON, self).__init__()

        # Init. configuration
        config = initialize_config_file(yaml_config_path)
        self.don_config = DONConfig.from_dictionary(config)
        self.optim_config = OptimizerConfig.from_dictionary(config)
        self.debug = self.don_config.don.debug
        self.debug_path = self.don_config.don.debug_path

        self.backbone = init_backbone(self.don_config.don.backbone)
        self.backbone.fc = torch.nn.Conv2d(self.backbone.inplanes,
                                           self.don_config.don.descriptor_dimension,
                                           kernel_size=1)

        # Init. loss function
        if self.don_config.loss.name == 'pixelwise_correspondence_loss':
            self.loss_function = PixelwiseCorrespondenceLoss(reduction=self.don_config.loss.reduction)

        elif self.don_config.loss.name == 'pixelwise_ntxent_loss':
            self.loss_function = PixelwiseNTXentLoss(self.don_config.loss.temperature,
                                                     self.don_config.loss.reduction)

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

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(input)
        scaled_map = F.interpolate(feature_map,
                                   size=input.size()[2:],
                                   mode='bilinear',
                                   align_corners=True)
        return scaled_map

    def _step(self, batch) -> torch.Tensor:
        image, mask, backgrounds = batch["RGBs-A"], batch["Masks-A"], batch["Random-Backgrounds"]
        image_a, matches_a, image_b, matches_b = synthetize(image,
                                                            mask,
                                                            backgrounds,
                                                            self.don_config.don.n_correspondence)
        dense_descriptors_a = self._forward(image_a)
        dense_descriptors_b = self._forward(image_b)

        loss = self.loss_function(dense_descriptors_a,
                                  dense_descriptors_b,
                                  matches_a,
                                  matches_b)

        # Override to save the last debug for the last epoch
        if (self.debug or
            self.trainer.current_epoch == self.trainer.max_epochs - 1 or
                self.trainer.current_epoch % 50 == 0):
            debug_correspondences(image_a,
                                  matches_a,
                                  image_b,
                                  matches_b,
                                  self.debug_path)
            debug_descriptors(image_a,
                              dense_descriptors_a,
                              image_b,
                              dense_descriptors_b,
                              self.debug_path)

        return loss

    def training_step(self, batch, batch_idx):

        loss = self._step(batch)
        self.log("train_loss", loss)

        if self.optim_config.enable_schedule:
            sch = self.lr_schedulers()
            sch.step()

        return loss

    def validation_step(self, batch, batch_idx):

        loss = self._step(batch)
        self.log("val_loss", loss)

        return loss

    def compute_dense_local_descriptors(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._forward(image)

    def compute_descriptors_from_numpy_image(self, numpy_image: np.ndarray) -> np.ndarray:
        image_tensor = torch.as_tensor(numpy_image, device=self.device, dtype=self.dtype) / 255.0

        if self.device != torch.device("cpu"):
            image_tensor = image_tensor.pin_memory(True)

        descriptors = self.compute_dense_local_descriptors(image_tensor.permute(2, 0, 1).unsqueeze(dim=0))
        return descriptors.squeeze(dim=0).permute(1, 2, 0).cpu().numpy()


def load_trained_don_model(yaml_config_path: str, trained_model_path: Union[str, None] = None) -> DON:

    if trained_model_path is None:
        config = initialize_config_file(yaml_config_path)
        trained_model_name = config["trainer"]["model_checkpoint_name"] + ".ckpt"
        trained_model_path = os.path.join(config["trainer"]["model_path"], trained_model_name)

    model = DON(yaml_config_path)
    trained_model = model.load_from_checkpoint(trained_model_path, yaml_config_path=yaml_config_path)

    return trained_model
