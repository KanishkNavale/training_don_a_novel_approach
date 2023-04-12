from typing import Dict

import torch
from pytorch3d.transforms import so3_rotation_angle
from kornia.geometry import relative_transformation

from src.keypointnet.geometry_transformations import camera_to_world_coordinates, pixel_to_camera_coordinates, kabsch_tranformation
from src.configurations.keypointnet import Loss
from src.distances import guassian_distance_kernel


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
    def _compute_pointwise_distances(points: torch.Tensor) -> torch.Tensor:

        mask = torch.triu(torch.ones((points.shape[1], points.shape[1]), dtype=torch.bool, device=points.device),
                          diagonal=1).unsqueeze(0).tile(points.shape[0], 1, 1)

        splicing_indices = torch.triu_indices(points.shape[1], points.shape[1], offset=1, device=points.device)
        splicing_indices = splicing_indices.permute(1, 0)

        masked_pointwise_distances = torch.cdist(points, points) * mask

        return masked_pointwise_distances[:, splicing_indices[:, 0], splicing_indices[:, 1]]

    @staticmethod
    def kernel_distance(x: torch.Tensor, margin: float) -> torch.Tensor:
        cutoff = torch.max(torch.zeros_like(x), margin - x)
        return torch.mean(cutoff, dim=-1)

    def _compute_separation_loss(self,
                                 uv_a: torch.Tensor,
                                 uv_b: torch.Tensor,
                                 margin: float) -> torch.Tensor:

        pointwise_distances_a = self._compute_pointwise_distances(uv_a)
        pointwise_distances_b = self._compute_pointwise_distances(uv_b)

        kernel_distance_a = self.kernel_distance(pointwise_distances_a, margin)
        kernel_distance_b = self.kernel_distance(pointwise_distances_b, margin)

        return 0.5 * (kernel_distance_a + kernel_distance_b)

    @staticmethod
    def _compute_pose_loss(cam_coords_a: torch.Tensor,
                           cam_coords_b: torch.Tensor,
                           extrinsics_a: torch.Tensor,
                           extrinsics_b: torch.Tensor) -> torch.Tensor:

        predicted_trafo_a_to_b = kabsch_tranformation(cam_coords_a, cam_coords_b, noise=1e-6)
        truth_trafo_a_to_b = relative_transformation(extrinsics_a, extrinsics_b)

        relative_pose = relative_transformation(truth_trafo_a_to_b, predicted_trafo_a_to_b)
        distance_angle = so3_rotation_angle(relative_pose[:, :3, :3], eps=1e-2)
        distance_linear = torch.linalg.norm(relative_pose[:, :3, 3], dim=-1)

        return torch.abs(distance_angle) + distance_linear

    @staticmethod
    def _compute_silhoutte_loss(exp_mask_a: torch.Tensor, exp_mask_b: torch.Tensor) -> torch.Tensor:
        logged_exp_a = -1.0 * torch.log(exp_mask_a + 1e-16)
        logged_exp_b = -1.0 * torch.log(exp_mask_b + 1e-16)

        return 0.5 * (logged_exp_a.mean(dim=-1) + logged_exp_b.mean(dim=-1))

    @staticmethod
    def _penalize_broad_probs(spat_probs: torch.Tensor, spat_exp: torch.Tensor) -> torch.Tensor:

        us = torch.arange(0,
                          spat_probs.shape[-2],
                          1,
                          dtype=torch.float32,
                          device=spat_probs.device)
        vs = torch.arange(0,
                          spat_probs.shape[-1],
                          1,
                          dtype=torch.float32,
                          device=spat_probs.device)

        grid = torch.stack(torch.meshgrid(us, vs, indexing='ij'), dim=-1)

        tiled_grid = grid[None, None, :].tile(spat_probs.shape[0], spat_exp.shape[1], 1, 1, 1)
        tiled_exp = spat_exp[:, :, None, None, :]  # .tile(1, 1, spat_probs.shape[-2], spat_probs.shape[-1], 1)

        distances = torch.linalg.norm(tiled_grid - tiled_exp, dim=-1)
        mask: torch.Tensor = distances > 2.0
        masked_distances = distances * mask.detach()

        return torch.mean(torch.sum(spat_probs * masked_distances, dim=(-2, -1)), dim=-1)

    def _compute_variance_loss(self, uv_a: torch.Tensor,
                               spat_probs_a: torch.Tensor,
                               uv_b: torch.Tensor,
                               spat_probs_b: torch.Tensor) -> torch.Tensor:

        var_a = self._penalize_broad_probs(spat_probs_a, uv_a)
        var_b = self._penalize_broad_probs(spat_probs_b, uv_b)

        return 0.5 * (var_a + var_b)

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

        spat_probs_a = computed_data["Spatial-Probs-A"]
        spat_probs_b = computed_data["Spatial-Probs-B"]
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

        var_loss = self._compute_variance_loss(exp_uvd_a[:, :, :-1],
                                               spat_probs_a,
                                               exp_uvd_b[:, :, :-1],
                                               spat_probs_b)

        sep_loss = self._compute_separation_loss(exp_uvd_a[:, :, :-1],
                                                 exp_uvd_b[:, :, :-1],
                                                 self.config.margin)

        sil_loss = self._compute_silhoutte_loss(exp_mask_a,
                                                exp_mask_b)

        weighted_batch_loss = self._compute_weighted_losses(mvc_loss,
                                                            var_loss,
                                                            sep_loss,
                                                            sil_loss)

        return {"Total": torch.sum(weighted_batch_loss),
                "Consistency": weighted_batch_loss[0],
                "Spatial": weighted_batch_loss[1],
                "Separation": weighted_batch_loss[2],
                "Silhoutte": weighted_batch_loss[3],
                }
