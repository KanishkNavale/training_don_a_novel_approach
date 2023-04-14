from typing import Dict

import torch

from src.keypointnet.geometry_transformations import kabsch_tranformation
from src.configurations.keypointnet import Loss


class KeypointNetLosses:
    def __init__(self, loss_config: Loss) -> None:

        # Init. configuration
        self.reduction = loss_config.reduction
        self.config = loss_config

    @staticmethod
    def _compute_multiview_consistency_loss(uvs_a: torch.Tensor,
                                            uvs_b: torch.Tensor,
                                            rotation_a_to_b: torch.Tensor,
                                            translation_a_to_b: torch.Tensor) -> torch.Tensor:

        projected_uvs_a_to_uvs_b = (rotation_a_to_b @ uvs_a.permute(0, 2, 1) + translation_a_to_b).permute(0, 2, 1)

        return torch.mean(torch.linalg.norm(uvs_b - projected_uvs_a_to_uvs_b, dim=-1), dim=-1)

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
    def _compute_pose_loss(uvs_a: torch.Tensor,
                           uvs_b: torch.Tensor,
                           rotation_a_to_b: torch.Tensor,
                           translation_a_to_b: torch.Tensor) -> torch.Tensor:

        predicted_rotation, predicted_translation = kabsch_tranformation(uvs_a, uvs_b)
        rotation_distance = torch.linalg.norm(rotation_a_to_b - predicted_rotation, dim=(-2, -1))
        translation_distance = torch.mean(torch.linalg.norm(translation_a_to_b - predicted_translation, dim=-1), dim=-1)

        return rotation_distance + translation_distance

    @staticmethod
    def _compute_stable_log(x: torch.tensor) -> torch.Tensor:
        clipped_x = torch.clamp(x, min=1e-16, max=1.0)
        return torch.abs(torch.log(clipped_x))

    def _compute_silhoutte_loss(self, exp_mask_a: torch.Tensor, exp_mask_b: torch.Tensor) -> torch.Tensor:
        logged_exp_a = self._compute_stable_log(exp_mask_a)
        logged_exp_b = self._compute_stable_log(exp_mask_b)

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

        tiled_grid = grid[None, None, :].tile(
            spat_probs.shape[0], spat_exp.shape[1], 1, 1, 1)
        tiled_exp = spat_exp[:, :, None, None, :]

        distances = torch.linalg.norm(tiled_grid - tiled_exp, dim=-1)
        mask: torch.Tensor = torch.where(distances > 2.0, 1.0, 0.0)
        masked_distances = distances * mask

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
                                 sihoutte_loss: torch.Tensor,
                                 divergence_loss: torch.Tensor) -> torch.Tensor:

        loss = torch.hstack([self._reduce(consistency_loss),
                             self._reduce(pose_loss),
                             self._reduce(separation_loss),
                             self._reduce(sihoutte_loss),
                             self._reduce(divergence_loss)])

        return self.config.loss_ratios_as_tensor.type(loss.dtype).to(loss.device) * loss

    def __call__(self, computed_data: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:

        spat_probs_a = computed_data["Spatial-Probs-A"]
        spat_probs_b = computed_data["Spatial-Probs-B"]
        exp_uvs_a = computed_data["Spatial-Expectations-A"]
        exp_uvs_b = computed_data["Spatial-Expectations-B"]
        exp_mask_a = computed_data["Spatial-Masks-A"]
        exp_mask_b = computed_data["Spatial-Masks-B"]
        optimal_rotation = computed_data["Optimal-Rotation-A2B"]
        optimal_translation = computed_data["Optimal-Translation-A2B"]

        mvc_loss = self._compute_multiview_consistency_loss(exp_uvs_a,
                                                            exp_uvs_b,
                                                            optimal_rotation,
                                                            optimal_translation)

        var_loss = self._compute_variance_loss(exp_uvs_a,
                                               spat_probs_a,
                                               exp_uvs_b,
                                               spat_probs_b)

        sep_loss = self._compute_separation_loss(exp_uvs_a,
                                                 exp_uvs_b,
                                                 self.config.margin)

        sil_loss = self._compute_silhoutte_loss(exp_mask_a,
                                                exp_mask_b)

        pose_loss = self._compute_pose_loss(exp_uvs_a,
                                            exp_uvs_b,
                                            optimal_rotation,
                                            optimal_translation)

        weighted_batch_loss = self._compute_weighted_losses(mvc_loss,
                                                            pose_loss,
                                                            sep_loss,
                                                            sil_loss,
                                                            var_loss,)

        return {"Total": torch.sum(weighted_batch_loss),
                "Consistency": weighted_batch_loss[0],
                "Pose": weighted_batch_loss[1],
                "Separation": weighted_batch_loss[2],
                "Silhoutte": weighted_batch_loss[3],
                "Divergence": weighted_batch_loss[4]
                }
