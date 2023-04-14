import torch
from kornia.geometry import Rt_to_matrix4x4


def pixel_to_camera_coordinates(uv: torch.Tensor,
                                depth: torch.Tensor,
                                intrinsic: torch.Tensor) -> torch.Tensor:
    # Processing batch tensor computation
    depth = depth.unsqueeze(dim=-1)
    uv1 = torch.cat([uv, torch.ones_like(depth)], dim=-1)
    uvz = torch.mul(uv1, depth)

    coordinates: torch.Tensor = torch.linalg.inv(intrinsic.unsqueeze(dim=1)) @ uvz.unsqueeze(dim=-1)
    return coordinates.squeeze(dim=-1)


def camera_to_world_coordinates(cam_coords: torch.Tensor,
                                extrinsic: torch.Tensor) -> torch.Tensor:
    cam_coords = cam_coords.unsqueeze(dim=-1)
    one_padded_cam_coords = torch.cat([cam_coords, torch.ones_like(cam_coords[:, :, :1])], dim=-2)
    coordinates = (extrinsic.unsqueeze(dim=1) @ one_padded_cam_coords)[:, :, :3]
    return coordinates.squeeze(dim=-1)


def kabsch_tranformation(source: torch.Tensor,
                         target: torch.Tensor,
                         noise: float = 1e-4) -> torch.Tensor:
    """Computes the relative transform between two sets of 3D points

    Reference: Kabsch, Wolfgang.
               "A discussion of the solution for the best rotation to relate two sets of vectors." 
               Acta Crystallographica Section A: Crystal Physics, Diffraction, 
               Theoretical and General Crystallography 34.5 (1978): 827-828."

    Args:
        source (torch.Tensor): tensor of 3D points of dims. (*[optional], N, 3)
        target (torch.Tensor): tensor of 3D points of dims. (*[optional], N, 3)

    Returns:
        torch.Tensor: source -> target transformation as SE3.

    """
    source = source.type(torch.float32)
    target = target.type(torch.float32)

    source = source + torch.randn_like(source).uniform_(0.0, noise)
    target = target + torch.randn_like(target).uniform_(0.0, noise)

    if not source.shape[-2:-1] >= torch.Size([3]) or not target.shape[-2:-1] >= torch.Size([3]):
        raise ValueError(
            "The number of points in the tensors must be >= 3.Accepted tensor size -> *B[optional] x [ >= 3] x 3")

    if len(source.shape) == 2 and len(target.shape) == 2:
        batched_source = source.unsqueeze(dim=0)
        batched_target = target.unsqueeze(dim=0)
    else:
        batched_source = source
        batched_target = target

    # Zero center the coordinates
    centroid_source = torch.mean(batched_source, dim=1).unsqueeze(dim=1)
    centroid_target = torch.mean(batched_target, dim=1).unsqueeze(dim=1)
    zero_centered_source = batched_source - centroid_source
    zero_centered_target = batched_target - centroid_target

    # Covariance matrices
    covariance = (zero_centered_source.permute(0, 2, 1) @ zero_centered_target)

    # SVD decomposition
    U: torch.Tensor
    VT: torch.Tensor
    U, _, VT = torch.linalg.svd(covariance)
    V = VT.permute(0, 2, 1)

    # Handling reflection case!
    normalizer = torch.eye(source.shape[-1],
                           dtype=source.dtype,
                           device=source.device).tile(batched_source.shape[0], 1, 1)
    normalizer[:, -1, -1] = torch.linalg.det(V @ U.permute(0, 2, 1))

    # Compute relative transformation
    rotation_source_to_target = V @ normalizer @ U.permute(0, 2, 1)

    translation_source_to_taget = centroid_target.mT - (rotation_source_to_target @ centroid_source.mT)

    if len(source.shape) == 2 and len(target.shape) == 2:
        return rotation_source_to_target.squeeze(dim=0), translation_source_to_taget.squeeze(dim=0)
    else:
        return rotation_source_to_target, translation_source_to_taget
