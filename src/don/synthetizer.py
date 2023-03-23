from typing import List, Tuple

import torch
import numpy as np
import torchvision.transforms as T


def _stack_image_with_mask_and_grid(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masks = masks.unsqueeze(dim=1)

    us = torch.arange(0, images.shape[-2], 1, dtype=torch.float32, device=images.device)
    vs = torch.arange(0, images.shape[-1], 1, dtype=torch.float32, device=images.device)
    grid = torch.meshgrid(us, vs, indexing='ij')
    spatial_grid = torch.stack(grid)

    tiled_spatial_grid = spatial_grid.unsqueeze(dim=0).tile(images.shape[0], 1, 1, 1)

    return torch.cat([images, masks, tiled_spatial_grid], dim=1)


def _destack_image_mask_spatialgrid(augmented_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = augmented_image[:, :3, :, :]
    mask = augmented_image[:, 3, :, :]
    spatial_grid = torch.round(augmented_image[:, 4:, :])

    return image, mask, spatial_grid


def _get_random_augmentation(image: torch.Tensor) -> torch.Tensor:
    if 1 == np.random.randint(0, 3):
        return T.RandomAffine(degrees=60, translate=(0, 0.2))(image)

    elif 1 == np.random.randint(0, 3):
        return T.RandomPerspective(distortion_scale=0.2, p=1.0)(image)

    elif 1 == np.random.randint(0, 3):
        return T.RandomVerticalFlip(p=1.0)(image)

    else:
        return image


def compute_correspondence_and_augmented_images(
        images: torch.Tensor,
        masks: torch.Tensor,
        n_correspondences: int) -> torch.Tensor:

    augmented_image_a = _get_random_augmentation(_stack_image_with_mask_and_grid(images, masks))
    augmented_image_b = _get_random_augmentation(_stack_image_with_mask_and_grid(images, masks))

    augmented_images_a, masks_a, grids_a = _destack_image_mask_spatialgrid(augmented_image_a)
    augmented_images_b, _, grids_b = _destack_image_mask_spatialgrid(augmented_image_b)

    matches_a: List[torch.Tensor] = []
    matches_b: List[torch.Tensor] = []

    for mask_a, grid_a, grid_b in zip(masks_a, grids_a, grids_b):

        valid_pixels_a = torch.where(mask_a != 0.0)

        us = valid_pixels_a[0]
        vs = valid_pixels_a[1]

        # Reducing computation costs
        trimming_indices = torch.linspace(0,
                                          us.shape[0] - 1,
                                          steps=10 * n_correspondences)
        us = us[trimming_indices.long()].type(torch.float32)
        vs = vs[trimming_indices.long()].type(torch.float32)

        valid_pixels_a = torch.vstack([us, vs]).permute(1, 0)
        valid_grids_a = grid_a[:, us.long(), vs.long()].permute(1, 0)
        tiled_valid_grids_a = valid_grids_a.view(valid_grids_a.shape[0], valid_grids_a.shape[1], 1, 1)

        spatial_grid_distances = torch.linalg.norm(grid_b - tiled_valid_grids_a, dim=1)

        match_indices_a, ubs, vbs = torch.where(spatial_grid_distances == 0.0)

        mutual_match_a = valid_pixels_a[match_indices_a.long()]
        mutual_match_b = torch.vstack([ubs, vbs]).permute(1, 0)

        trimming_indices = torch.linspace(0,
                                          mutual_match_a.shape[0] - 1,
                                          steps=n_correspondences)
        trimming_indices = trimming_indices.type(torch.int64)

        matches_a.append(mutual_match_a[trimming_indices])
        matches_b.append(mutual_match_b[trimming_indices])

    return augmented_images_a, torch.stack(matches_a), augmented_images_b, torch.stack(matches_b)
