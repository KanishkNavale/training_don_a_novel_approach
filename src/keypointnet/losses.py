from typing import Tuple, List, Dict

import torch

from src.keypointnet.geometry_transformations import camera_to_world_coordinates, pixel_to_camera_coordinates
from src.configurations.keypointnet import Loss


class KeypointNetLosses:
    def __init__(self, loss_config: Loss) -> None:
        self.name = 'Keypoint Network Losses'

        # Init. configuration
        self.reduction = loss_config.reduction
        self.config = loss_config

    @staticmethod
    def _compute_multiview_consistency_loss(cam_coords_a: torch.Tensor,
                                            extrinsic_a: torch.Tensor,
                                            cam_coords_b: torch.Tensor,
                                            extrinsic_b: torch.Tensor) -> torch.Tensor:

        world_coords_a = camera_to_world_coordinates(cam_coords_a, extrinsic_a)
        world_coords_b = camera_to_world_coordinates(cam_coords_b, extrinsic_b)

        return torch.mean(torch.linalg.norm(world_coords_a - world_coords_b, dim=-1), dim=-1)

    @staticmethod
    def _compute_separation_loss(cam_coords_a: torch.Tensor,
                                 cam_coords_b: torch.Tensor,
                                 margin: float) -> torch.Tensor:

        mask = torch.triu(torch.ones((3, 3), dtype=torch.bool), diagonal=1).unsqueeze(0).tile(cam_coords_a.shape[0], 1, 1)

        pointwise_distances_a = torch.masked_select(torch.cdist(cam_coords_a, cam_coords_a) * mask, mask)
        pointwise_distances_b = torch.masked_select(torch.cdist(cam_coords_b, cam_coords_b) * mask, mask)

        # Reshape the upper triangle into a 2D tensor
        min_distance_a = pointwise_distances_a.view(pointwise_distances_a.shape[0], -1).min(-1)
        min_distance_b = pointwise_distances_b.view(pointwise_distances_a.shape[0], -1).min(-1)

        kernelized_margin = torch.exp(- torch.square(min_distance_a) / (2 * margin**2)) + torch.exp(- torch.square(min_distance_b) / (2 * margin**2))

        return kernelized_margin / 2

    @staticmethod
    def _compute_silhoutte_loss(exp_mask_a: torch.Tensor, exp_mask_b: torch.Tensor) -> torch.Tensor:
        logged_exp_a = torch.log(exp_mask_a + 1e-16).mean(dim=-1)
        logged_exp_b = torch.log(exp_mask_b + 1e-16).mean(dim=-1)

        return (logged_exp_a + logged_exp_b) / 2

    def _reduce(self, list_of_tensors: List[torch.Tensor]) -> torch.Tensor:
        if self.config.reduction == "sum":
            return torch.sum(torch.vstack(list_of_tensors))
        else:
            return torch.mean(torch.vstack(list_of_tensors))

    def _compute_weighted_losses(self,
                                 consistency_loss: List[torch.Tensor],
                                 pose_loss: List[torch.Tensor],
                                 separation_loss: List[torch.Tensor],
                                 sihoutte_loss: List[torch.Tensor]) -> torch.Tensor:

        loss = torch.hstack([self._reduce(consistency_loss),
                             self._reduce(pose_loss),
                             self._reduce(separation_loss),
                             self._reduce(sihoutte_loss)])

        return self.config.loss_ratios_as_tensor.type(loss.dtype).to(loss.device) * loss

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        intrinsics_a = data["Intrinsics-A"]
        intrinsics_b = data["Intrinsics-B"]
        extrinsics_a = data["Extrinsics-A"]
        extrinsics_b = data["Extrinsics-B"]
        spatial_probs_a = data["Spatial-Probs-A"]
        spatial_probs_b = data["Spatial-Probs-B"]
        exp_uvd_a = data["Spatial-Expectations-A"]
        exp_uvd_b = data["Spatial-Expectations-B"]
        exp_mask_a = data["Spatial-Masks-A"]
        exp_mask_b = data["Spatial-Masks-B"]

        cam_coords_a = pixel_to_camera_coordinates(exp_uvd_a[:, :, :-1],
                                                   exp_uvd_a[:, :, -1].unsqueeze(dim=-1),
                                                   intrinsics_a)
        cam_coords_b = pixel_to_camera_coordinates(exp_uvd_b[:, :, :-1],
                                                   exp_uvd_b[:, :, -1].unsqueeze(dim=-1),
                                                   intrinsics_b)

        mvc_loss = self._compute_multiview_consistency_loss(cam_coords_a,
                                                            extrinsics_a,
                                                            cam_coords_b,
                                                            extrinsics_b)
        sep_loss = self._compute_separation_loss(cam_coords_a, cam_coords_b, self.config.margin)
        sil_loss = self._compute_silhoutte_loss(exp_mask_a, exp_mask_b)

        weighted_batch_loss = self._compute_weighted_losses([mvc_loss,
                                                             torch.zeros_like(mvc_loss),
                                                             sep_loss,
                                                             sil_loss])

        return {"Total": torch.sum(weighted_batch_loss),
                "Consistency": weighted_batch_loss[0],
                "Relative Pose": weighted_batch_loss[1],
                "Separation": weighted_batch_loss[2],
                "Silhoutte": weighted_batch_loss[3],
                }
