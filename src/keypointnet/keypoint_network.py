import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.configurations import OptimizerConfig, KeypointNetConfig
from src.keypointnet.losses import KeypointNetLosses
from src.utils import init_backbone


class KeypointNetwork(pl.LightningModule):

    def __init__(self) -> None:
        super(KeypointNetwork, self).__init__()

    def config(self,
               keypointnet_config: KeypointNetConfig,
               optim_config: OptimizerConfig) -> None:

        # Init. configuration
        self.keypointnet_config = keypointnet_config
        self.optim_config = optim_config

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

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
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

        depth = self.depth_layer.forward(resized_x)

        return spatial_probs, depth

    def configure_optimizers(self):
        if self.optim_config.name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_config.learning_rate, weight_decay=1e-4)

        else:
            raise NotImplementedError(f"This optimizer: {self.optim_config.name} is not implemented")

        if self.optim_config.enable_schedular:
            sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.optim_config.schedular_step_size, gamma=0.99)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": "train_loss",

                }
            }
        else:
            return optimizer

    def training_step(self, batch, batch_idx):

        batch_pcls_a = batch["RGBs-A"]
        batch_pcls_b = batch["RGBs-B"]

        spat_exp_a, depth_a = self._forward(batch_pcls_a)
        spat_exp_b, depth_b = self._forward(batch_pcls_b)

        batch_loss = self.loss_function(depth_a,
                                        depth_b,
                                        batch["Intrinsics-A"],
                                        batch["Intrinsics-B"],
                                        batch["Extrinsics-A"],
                                        batch["Extrinsics-B"],
                                        batch["Masks-A"],
                                        batch["Masks-B"],
                                        spat_exp_a,
                                        spat_exp_b)

        self.log("Training Loss", {**batch_loss})

        return batch_loss["Total"]

    def training_step_end(self, step_output) -> None:
        sch = self.lr_schedulers()
        sch.step()

    def validation_step(self, batch, batch_idx):

        batch_pcls_a = batch["RGBs-A"]
        batch_pcls_b = batch["RGBs-B"]

        spat_exp_a, depth_a = self._forward(batch_pcls_a)
        spat_exp_b, depth_b = self._forward(batch_pcls_b)

        batch_loss = self.loss_function(depth_a,
                                        depth_b,
                                        batch["Intrinsics-A"],
                                        batch["Intrinsics-B"],
                                        batch["Extrinsics-A"],
                                        batch["Extrinsics-B"],
                                        batch["Masks-A"],
                                        batch["Masks-B"],
                                        spat_exp_a,
                                        spat_exp_b)

        self.log("val_pass", {**batch_loss})

        return batch_loss["Total"]

    def validation_epoch_end(self, outputs):
        loss = torch.sum(torch.hstack(outputs)) if self.config.loss.reduction == 'sum' else torch.mean(torch.hstack(outputs))
        self.log("Validation Loss", loss)
        return loss

    def get_dense_representation(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.resnet.forward(input)
            return F.interpolate(x,
                                 size=input.size()[-2:],
                                 mode='bilinear',
                                 align_corners=True)
