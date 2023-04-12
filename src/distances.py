import torch


@torch.jit.script
def l2(target: torch.Tensor, source: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.linalg.norm(source - target, ord=2, dim=dim)


@torch.jit.script
def l1(target: torch.Tensor, source: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.linalg.norm(source - target, ord=1, dim=dim)


def guassian_distance_kernel(target: torch.Tensor,
                             source: torch.Tensor,
                             temperature: float,
                             dim: int = -1) -> torch.Tensor:
    distances = l2(target, source, dim)
    weights = torch.exp((-1.0 * distances) / temperature)
    return weights


@torch.jit.script
def cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    product = torch.sum(a * b, dim=dim)
    normalizer = torch.linalg.norm(a, dim=dim) * torch.linalg.norm(b, dim=dim)
    return product / (normalizer + 1e-12)


@torch.jit.script
def exp_guassian_distance_kernel(target: torch.Tensor,
                                 source: torch.Tensor,
                                 temperature: torch.Tensor,
                                 dim: int = -1) -> torch.Tensor:
    distances = l2(target, source, dim)
    weights = torch.exp((-1.0 * distances) / torch.exp(temperature))
    return weights


@torch.jit.script
def compute_keypoint_expectation(image: torch.Tensor,
                                 keypoint: torch.Tensor,
                                 temp: torch.Tensor) -> torch.Tensor:
    weights = exp_guassian_distance_kernel(image, keypoint, temp)
    spatial_probabilities = weights / weights.sum()
    sum_of_spatial_probs = torch.sum(spatial_probabilities)

    if not torch.allclose(sum_of_spatial_probs, torch.ones_like(sum_of_spatial_probs)):
        raise ValueError("Spatial probabilities don't add upto 1.0")
    else:
        return spatial_probabilities


@torch.jit.script
def compute_keypoint_confident_expectation(image: torch.Tensor,
                                           keypoint: torch.Tensor,
                                           temp: torch.Tensor,
                                           confidence: torch.Tensor) -> torch.Tensor:
    spatial_expectation = compute_keypoint_expectation(image, keypoint, temp)
    normalized_expectation = spatial_expectation / spatial_expectation.max()
    zero = torch.zeros(1, device=normalized_expectation.device, dtype=normalized_expectation.dtype)
    return torch.where(normalized_expectation >= confidence, normalized_expectation, zero)
