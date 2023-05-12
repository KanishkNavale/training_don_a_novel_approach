from typing import List, Tuple, Union, Dict
import gc

from tqdm import tqdm
import numpy as np
import torch

from src.don.synthetizer import augment_images_and_compute_correspondences as synthetize
from src.distances import l2, cosine_similarity
from src.don import DON
from src.datamodule import DataModule


@torch.jit.script
def argmin_2d(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D spatial argmin of a tensor along the last two dimensions.

    Args:
        tensor: A PyTorch tensor with shape (batch_size, height, width).

    Returns:
        A tensor with shape (batch_size, 2), where the last dimension contains
        the x and y coordinates of the argmin for each batch item in the input.
    """
    # Flatten the tensor along the spatial dimensions
    flattened_tensor = tensor.view(tensor.size(0), -1)

    # Compute the argmin along the flattened dimensions
    argmin = torch.argmin(flattened_tensor, dim=-1)

    # Convert the argmin indices to coordinates
    x = argmin % tensor.size(-1)
    y = argmin // tensor.size(-1)

    # Stack the x and y coordinates to form the output tensor
    output = torch.stack((x, y), dim=-1)

    return output


def PCK(descriptor_image_a: torch.Tensor,
        descriptor_image_b: torch.Tensor,
        matches_a: torch.Tensor,
        matches_b: torch.Tensor,
        k: float) -> float:

    queried_descriptors_a = descriptor_image_a[:, matches_a[:, 0], matches_a[:, 1]].permute(1, 0)
    tiled_queried_desciptors_a = queried_descriptors_a.reshape((matches_a.shape[0], 1, 1, descriptor_image_a.shape[0]))
    tiled_image_b = descriptor_image_b.unsqueeze(0).permute(0, 2, 3, 1)

    spatial_distances_of_descriptors_a_in_image_b = l2(tiled_image_b, tiled_queried_desciptors_a, dim=-1)

    indices = argmin_2d(spatial_distances_of_descriptors_a_in_image_b)

    similarities = cosine_similarity(indices.type(descriptor_image_a.dtype),
                                     matches_b.type(descriptor_image_a.dtype))

    iversion = (similarities >= k).type(torch.float32)

    return iversion.mean().cpu()


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

    for i in tqdm(range(iterations),
                  desc="Benchmarking Metric: AUC of PCK@K~[1, 100]",
                  total=iterations):

        PROFILE = torch.zeros(100, dtype=torch.float32)

        for k in tqdm(range(1, 101),
                      desc=f"Iteration {i + 1}/{iterations}",
                      total=100):

            # Sample a random pair of images
            batch: Dict[str, torch.Tensor] = dataset.dataset.__getitem__(np.random.randint(0, len(dataset.dataset) - 1))
            image, mask, backgrounds = batch["RGBs-A"], batch["Masks-A"], batch["Random-Backgrounds"]

            image = image.to(device=device, non_blocking=True).unsqueeze(dim=0)
            mask = mask.to(device=device, non_blocking=True).unsqueeze(dim=0)
            backgrounds = backgrounds.to(device=device, non_blocking=True).unsqueeze(dim=0)

            # Synthetize a pair of images
            image_a, matches_a, _, image_b, matches_b, _ = synthetize(image,
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

            # Stop some memry leaks
            del (batch,
                 image,
                 mask,
                 backgrounds,
                 image_a,
                 matches_a,
                 image_b,
                 matches_b,
                 descriptor_a,
                 descriptor_b)
            gc.collect()

        PCK_PROFILES.append(PROFILE)

    AUC = [torch.trapz(profile, torch.linspace(0, 1, 100)).numpy() for profile in PCK_PROFILES]
    PCK_PROFILES = [profile.numpy() for profile in PCK_PROFILES]

    return (PCK_PROFILES, AUC)
