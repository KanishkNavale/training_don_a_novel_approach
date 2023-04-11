from typing import Dict

import torch
from pytorch3d.transforms import se3_log_map
from kornia.geometry import relative_transformation

from src.keypointnet.geometry_transformations import camera_to_world_coordinates, pixel_to_camera_coordinates, kabsch_tranformation
from src.configurations.keypointnet import Loss


class KeypointNetLosses:
    def __init__(self, loss_config: Loss) -> None:

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
    def _compute_separation_loss(uv_a: torch.Tensor,
                                 uv_b: torch.Tensor,
                                 margin: float) -> torch.Tensor:

        mask = torch.triu(torch.ones((uv_a.shape[1], uv_a.shape[1]), dtype=torch.bool, device=uv_a.device),
                          diagonal=1).unsqueeze(0).tile(uv_a.shape[0], 1, 1)

        splicing_indices = torch.triu_indices(uv_a.shape[1], uv_a.shape[1], offset=1, device=uv_a.device)
        splicing_indices = splicing_indices.permute(1, 0)

        masked_pointwise_distances_a = torch.cdist(uv_a, uv_a) * mask
        masked_pointwise_distances_b = torch.cdist(uv_b, uv_b) * mask

        spliced_pointwise_distances_a = masked_pointwise_distances_a[:, splicing_indices[:, 0], splicing_indices[:, 1]]
        spliced_pointwise_distances_b = masked_pointwise_distances_b[:, splicing_indices[:, 0], splicing_indices[:, 1]]

        # Reshape the upper triangle into a 2D tensor
        min_distance_a = torch.min(spliced_pointwise_distances_a, dim=-1)[0]
        min_distance_b = torch.min(spliced_pointwise_distances_b, dim=-1)[0]

        kernelized_margin = torch.exp(-0.5 * torch.square(min_distance_a / margin)) + torch.exp(-0.5 * torch.square(min_distance_b / margin))

        return kernelized_margin / 2.0

    @staticmethod
    def _compute_pose_loss(cam_coords_a: torch.Tensor,
                           cam_coords_b: torch.Tensor,
                           extrinsics_a: torch.Tensor,
                           extrinsics_b: torch.Tensor) -> torch.Tensor:

        predicted_trafo_a_to_b = kabsch_tranformation(cam_coords_a, cam_coords_b)
        truth_trafo_a_to_b = relative_transformation(extrinsics_a, extrinsics_b)

        relative_pose = relative_transformation(truth_trafo_a_to_b, predicted_trafo_a_to_b)
        pose_log_map = se3_log_map(relative_pose.permute(0, 2, 1), eps=1e-2)

        geodesic_distance = torch.linalg.norm(pose_log_map, dim=-1)
        return geodesic_distance

    @staticmethod
    def _compute_silhoutte_loss(exp_mask_a: torch.Tensor, exp_mask_b: torch.Tensor) -> torch.Tensor:
        logged_exp_a = -1.0 * torch.log(exp_mask_a + 1e-16)
        logged_exp_b = -1.0 * torch.log(exp_mask_b + 1e-16)

        return (logged_exp_a.mean(dim=-1) + logged_exp_b.mean(dim=-1)) / 2

    def _reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.config.reduction == "sum":
            return torch.sum(tensor)
        else:
            return torch.mean(tensor)

    def _compute_weighted_losses(self,
                                 consistency_loss: torch.Tensor,
                                 pose_loss: torch.Tensor,
                                 separation_loss: torch.Tensor,
                                 sihoutte_loss: torch.Tensor) -> torch.Tensor:

        loss = torch.hstack([self._reduce(consistency_loss),
                             self._reduce(pose_loss),
                             self._reduce(separation_loss),
                             self._reduce(sihoutte_loss)])

        return self.config.loss_ratios_as_tensor.type(loss.dtype).to(loss.device) * loss

    def __call__(self,
                 batch_data: Dict[str, torch.Tensor],
                 computed_data: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:

        intrinsics_a = batch_data["Intrinsics-A"]
        intrinsics_b = batch_data["Intrinsics-B"]
        extrinsics_a = batch_data["Extrinsics-A"]
        extrinsics_b = batch_data["Extrinsics-B"]
        spatial_probs_a = computed_data["Spatial-Probs-A"]
        spatial_probs_b = computed_data["Spatial-Probs-B"]
        exp_uvd_a = computed_data["Spatial-Expectations-A"]
        exp_uvd_b = computed_data["Spatial-Expectations-B"]
        exp_mask_a = computed_data["Spatial-Masks-A"]
        exp_mask_b = computed_data["Spatial-Masks-B"]

        cam_coords_a = pixel_to_camera_coordinates(exp_uvd_a[:, :, :-1],
                                                   exp_uvd_a[:, :, -1],
                                                   intrinsics_a)
        cam_coords_b = pixel_to_camera_coordinates(exp_uvd_b[:, :, :-1],
                                                   exp_uvd_b[:, :, -1],
                                                   intrinsics_b)

        mvc_loss = self._compute_multiview_consistency_loss(cam_coords_a,
                                                            extrinsics_a,
                                                            cam_coords_b,
                                                            extrinsics_b)

        pose_loss = self._compute_pose_loss(cam_coords_a, cam_coords_b, extrinsics_a, extrinsics_b)

        sep_loss = self._compute_separation_loss(cam_coords_a, cam_coords_b, self.config.margin)

        sil_loss = self._compute_silhoutte_loss(exp_mask_a, exp_mask_b)

        weighted_batch_loss = self._compute_weighted_losses(mvc_loss,
                                                            pose_loss,
                                                            sep_loss,
                                                            sil_loss)

        return {"Total": torch.sum(weighted_batch_loss),
                "Consistency": weighted_batch_loss[0],
                "Relative Pose": weighted_batch_loss[1],
                "Separation": weighted_batch_loss[2],
                "Silhoutte": weighted_batch_loss[3],
                }
