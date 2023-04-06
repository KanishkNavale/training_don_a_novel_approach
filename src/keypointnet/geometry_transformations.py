import torch


def pixel_to_camera_coordinates(uv: torch.Tensor,
                                depth: torch.Tensor,
                                intrinsic: torch.Tensor) -> torch.Tensor:
    """
        Projects pixel coordinates to camera coordinates:
        f: I[u, v], D[u, v], Camera Intrinsics -> C[x, y, z].

        Supports batch-wise processing fashion and singleton tensor processing.

    Args:
        uv (torch.Tensor): pixel coordinates dims -> (*[optional], *[optional], 2 , 1)
        depth (torch.Tensor): depth of dims -> (*[optional], *[optional], 1)
        intrinsic (torch.Tensor): camera intrinsic parameter of dims -> (*[optional], *[optional], 3 , 3)

    Returns:
        torch.Tensor: camera coordinates (*[optional], *[optional], 3 , 1)
    """

    if uv.shape == (2, 1) and depth.shape == (1,) and intrinsic.shape == (3, 3):

        # Processing singleton tensor computation
        # Could have written it better code-wise! Leaving it as it is for math intuition.
        uv1 = torch.vstack((uv, torch.ones(1, device=uv.device, dtype=uv.dtype)))
        uvz = depth * uv1

        return torch.linalg.inv(intrinsic) @ uvz

    elif uv.shape[0] == depth.shape[0] and intrinsic.shape[0] == depth.shape[0]:
        # Processing batch tensor computation
        depth = depth.unsqueeze(dim=-2)
        uv1 = torch.cat([uv, torch.ones_like(depth)], dim=-2)
        uvz = torch.mul(uv1, depth)

        return torch.linalg.inv(intrinsic) @ uvz

    else:
        raise ValueError("Found mismatch in dimensions of tensors")


def camera_to_world_coordinates(cam_coords: torch.Tensor,
                                extrinsic: torch.Tensor) -> torch.Tensor:
    """
        Projects camera coordinates to world coordinates:
        f: C[x, y, z], Extrinsics(Cam -> World) -> W[x, y, z]

        Supports batch-wise processing fashion and singleton tensor processing.

    Args:
        cam_coords (torch.Tensor): camera coordinates of dims -> (*[optional], *[optional], 3 , 1)
        extrinsic (torch.Tensor): camera extrinsic (camera to world) of dims -> (*[optional], *[optional], 3 , 3)

    Returns:
        torch.Tensor: world coordinates of dims -> (*[optional], *[optional], 3 , 1)
    """
    if not cam_coords.shape[-2:] == (3, 1) and extrinsic.shape[-2:] == (3, 3):
        raise ValueError("The last dimensions of cam_coords must be (3, 1) and extrinsics must be (3, 3)")

    if len(cam_coords.shape) == 2 and len(extrinsic.shape) == 2:
        return (extrinsic @ torch.vstack([cam_coords, torch.ones_like(cam_coords[1])]))[:3]

    elif len(cam_coords.shape) == 3 and len(extrinsic.shape) == 3:
        return (extrinsic @ torch.cat([cam_coords, torch.ones_like(cam_coords[:, :1, :])], dim=1))[:, :3, :]

    elif len(cam_coords.shape) == 4 and len(extrinsic.shape) == 4:
        return (extrinsic @ torch.cat([cam_coords, torch.ones_like(cam_coords[:, :, :1, :])], dim=-2))[:, :, :3, :]

    else:
        raise ValueError("Found mismatch in dimensions of tensors")


def pixel_to_world_coordinates(uv: torch.Tensor,
                               depth: torch.Tensor,
                               intrinsic: torch.Tensor,
                               extrinsic: torch.Tensor) -> torch.Tensor:
    """
        Projects pixel coordinates to world coordinates:
        f: I[u, v], D[u, v], Camera Intrinsics, Camera Extrinsics (Cam -> World) -> W[x, y, z]

        Supports batch-wise processing fashion and singleton tensor processing.

    Args:
        uv (torch.Tensor): pixel coordinates dims -> (*[optional], *[optional], 2 , 1)
        depth (torch.Tensor): depth of dims -> (*[optional], *[optional], 1)
        intrinsic (torch.Tensor): camera intrinsic parameter of dims -> (*[optional], *[optional], 3 , 3)
        extrinsic (torch.Tensor): camera extrinsic (camera to world) of dims -> (*[optional], *[optional], 4 , 4)

    Returns:
        torch.Tensor: world coordinates of dims -> (*[optional], *[optional], 3 , 1)
    """
    return camera_to_world_coordinates(pixel_to_camera_coordinates(uv, depth, intrinsic), extrinsic)
