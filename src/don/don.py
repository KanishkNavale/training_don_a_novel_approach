from typing import List, Tuple

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl

from src.don.synthetizer import compute_correspondence_and_augmented_images as synthetize
from src.configurations import DONConfig, OptimizerConfig
from src.don.losses import PixelwiseCorrespondenceLoss, PixelwiseNTXentLoss
from src.utils import init_backbone


class DON(pl.LightningModule):

    def __init__(self) -> None:
        super(DON, self).__init__()

    def config(self, don_fig: DONConfig, optim_config: OptimizerConfig) -> None:

        # Init. configuration
        self.don_config = don_fig
        self.optim_config = optim_config

        self.backbone = init_backbone(self.don_config.don.backbone)
        self.backbone.fc = torch.nn.Conv2d(self.backbone.inplanes,
                                           self.don_config.don.descriptor_dimension,
                                           kernel_size=3)

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
                                         weight_decay=1e-4)

        else:
            raise NotImplementedError(f"This optimizer: {self.optim_config.name} is not implemented")

        if self.optim_config.enable_schedular:
            sch = torch.optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.optim_config.schedular_step_size,
                                                  gamma=0.9)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": "train_loss",

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
        image, mask = batch["RGBs-A"], batch["Masks-A"]
        image_a, matches_a, image_b, matches_b = synthetize(image,
                                                            mask,
                                                            self.don_config.don.n_correspondence)
        dense_descriptors_a = self._forward(image_a)
        dense_descriptors_b = self._forward(image_b)

        loss = self.loss_function(dense_descriptors_a,
                                  dense_descriptors_b,
                                  matches_a,
                                  matches_b)

        return loss

    def training_step(self, batch, batch_idx):

        loss = self._step(batch)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        loss = self._step(batch)
        self.log("val_loss", loss)

        return loss

    @ torch.no_grad()
    def compute_dense_local_descriptors(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._forward(input)
