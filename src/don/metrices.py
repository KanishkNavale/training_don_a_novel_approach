from typing import List, Tuple, Union, Dict

from tqdm import tqdm
import numpy as np
import torch

from src.don.synthetizer import augment_images_and_compute_correspondences as synthetize
from src.distances import l2, cosine_similarity
from src.don import DON
from src.datamodule import DataModule


def PCK(descriptor_image_a: torch.Tensor,
        descriptor_image_b: torch.Tensor,
        matches_a: torch.Tensor,
        matches_b: torch.Tensor,
        k: float) -> float:

    us = torch.arange(0, descriptor_image_a.shape[-2], 1, dtype=torch.float32, device=descriptor_image_a.device)
    vs = torch.arange(0, descriptor_image_a.shape[-1], 1, dtype=torch.float32, device=descriptor_image_a.device)
    grid = torch.meshgrid(us, vs, indexing='ij')

    queried_descriptors_a = descriptor_image_a[:, matches_a[:, 0], matches_a[:, 1]].permute(1, 0)
    tiled_queried_desciptors_a = queried_descriptors_a.reshape((matches_a.shape[0], 1, 1, descriptor_image_a.shape[0]))
    tiled_image_b = descriptor_image_b.unsqueeze(0).permute(0, 2, 3, 1)

    spatial_distances_of_descriptors_a_in_image_b = l2(tiled_image_b, tiled_queried_desciptors_a, dim=-1)
    kernel_distances = torch.exp(-1.0 * torch.square(spatial_distances_of_descriptors_a_in_image_b)) + 1e-16
    spatial_probabilities = kernel_distances / torch.sum(kernel_distances, dim=(1, 2), keepdim=True)

    if not torch.allclose(torch.sum(spatial_probabilities, dim=(1, 2)),
                          torch.ones(matches_a.shape[0],
                                     dtype=spatial_probabilities.dtype,
                                     device=spatial_probabilities.device)):
        raise ValueError("Spatial probabilities do not add up to 1.0")

    spatial_expectations_u = torch.sum(torch.multiply(grid[0], spatial_probabilities), dim=(1, 2))
    spatial_expectations_v = torch.sum(torch.multiply(grid[1], spatial_probabilities), dim=(1, 2))
    spatial_expectation_uv = torch.hstack([spatial_expectations_u[:, None], spatial_expectations_v[:, None]])

    similarities = cosine_similarity(spatial_expectation_uv, matches_b.type(spatial_expectation_uv.dtype))

    inversion = (similarities >= k).type(torch.float32)

    return inversion.mean().cpu()


def AUC_for_PCK(trained_model: DON,
                datamodule: DataModule,
                iterations: int,
                n_correspondences: int,
                device: Union[torch.device, None] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    # Init. Datamodule
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    dataset = datamodule.val_dataloader(batch_size=1)

    # Init. device
    if device is not torch.device("cpu"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move the model to device
    trained_model = trained_model.to(device)

    PCK_PROFILES: List[torch.Tensor] = []

    for i in tqdm(range(iterations), desc="Benchmarking Metric: AUC of PCK@k", total=iterations):

        PROFILE = torch.zeros(100, dtype=torch.float32)

        for k in tqdm(range(1, 101), desc=f"Iteration {i + 1}/{iterations}", total=100):

            # Sample a random pair of images
            batch: Dict[str, torch.Tensor] = next(iter(dataset))
            image, mask, backgrounds = batch["RGBs-A"], batch["Masks-A"], batch["Random-Backgrounds"]

            image = image.to(device=device, non_blocking=True)
            mask = mask.to(device=device, non_blocking=True)
            backgrounds = backgrounds.to(device=device, non_blocking=True)

            # Synthetize a pair of images
            image_a, matches_a, image_b, matches_b = synthetize(image,
                                                                mask,
                                                                backgrounds,
                                                                n_correspondences)

            # Compute descriptors
            descriptor_a = trained_model.compute_dense_local_descriptors(image_a)
            descriptor_b = trained_model.compute_dense_local_descriptors(image_b)

            # Compute PCK
            pck = PCK(descriptor_a.squeeze(0),
                      descriptor_b.squeeze(0),
                      matches_a.squeeze(0),
                      matches_b.squeeze(0),
                      k / 100.0)

            PROFILE[k - 1] = pck

        PCK_PROFILES.append(PROFILE)

    AUC = [torch.trapz(profile, torch.linspace(0, 1, 100)).numpy() for profile in PCK_PROFILES]
    PCK_PROFILES = [profile.numpy() for profile in PCK_PROFILES]

    return (PCK_PROFILES, AUC)
