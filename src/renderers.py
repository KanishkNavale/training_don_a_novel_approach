import cv2
import numpy as np

import torch

from src import utils


def render_pose(img: np.ndarray, pose: np.ndarray, intrinsic: torch.Tensor, axis_lenght: float = 0.5) -> np.ndarray:
    """Renders the pose axes on the image.

    Args:
        img (np.ndarray): Image of cv2(uint8) datatype
        pose (Pose): Pose datatype
        intrinsic (torch.Tensor): camera intrinsic parameter
        axis_lenght (float, optional): axis lenght (units of intrinsic). Defaults to 0.05.

    Returns:
        np.ndarray: rotation axes annotated image of cv2 uint8 format
    """

    intrinsic = utils.convert_tensor_to_numpy(intrinsic)
    rotation = utils.convert_tensor_to_numpy(pose.rotation)

    center = pose.translation_as_numpy
    rotation, _ = cv2.Rodrigues(rotation)

    dist = np.zeros(4, dtype=center.dtype)

    # Handle distortion
    points = axis_lenght * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=dist.dtype)

    axis_points, _ = cv2.projectPoints(points, rotation, center, intrinsic, dist)
    axis_points = axis_points.astype(np.int64)

    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[0].ravel()), (255, 0, 0), 3)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[1].ravel()), (0, 255, 0), 3)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[2].ravel()), (0, 0, 255), 3)

    return img
