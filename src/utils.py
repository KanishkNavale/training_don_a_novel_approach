from typing import Any, Dict
import os

from omegaconf import OmegaConf
import cv2
import numpy as np
import torch
from torchvision import models


def initialize_config_file(path: str) -> Dict[str, Any]:
    config_path = os.path.abspath(path)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Could not find the specified config. file at: {config_path}")

    return OmegaConf.load(config_path)


def init_backbone(model_name: str) -> models.ResNet:

    # Init. parent model
    if model_name == "resnet_18":
        model = models.resnet18(pretrained=False)

    elif model_name == "resnet_34":
        model = models.resnet34(pretrained=False)

    elif model_name == "resnet_50":
        model = models.resnet50(pretrained=False)

    else:
        raise ValueError("Unidentified backbone found in the config!")

    return model


def convert_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a tensor of torch to numpy format

    Args:
        tensor (torch.Tensor): torch tensor

    Returns:
        np.ndarray: numpy tensor
    """
    if torch.is_tensor(tensor):
        with torch.no_grad():
            return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    else:
        return tensor


def convert_tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
    """Converts a tensor to cv2 format data

    Args:
        tensor (torch.Tensor): tensor

    Returns:
        np.ndarray: uint8 cv2 format
    """
    image = convert_tensor_to_numpy(tensor)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
