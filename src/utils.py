from typing import Any, List, Dict
import os
import json
from omegaconf import OmegaConf

import torch
import numpy as np
import cv2

import src.__models as models


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


def dump_as_json(path: str, filename: str, data: Any) -> None:
    """Dumps a json compatible data to .json file

    Args:
        path (str): path
        filename (str): filename with .json extension
        data (Any): json compatible data
    """
    path = os.path.abspath(path)
    file_path = os.path.join(path, filename)

    with open(file_path, 'w', encoding='utf8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def convert_tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
    """Converts a tensor to cv2 format data

    Args:
        tensor (torch.Tensor): tensor

    Returns:
        np.ndarray: uint8 cv2 format
    """
    image = convert_tensor_to_numpy(tensor)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def dump_as_image(path: str, filename: str, data: Any) -> None:
    """Dumps a cv2 compatible data to .png file

    Args:
        path (str): path
        filename (str): filename with .png extension
        data (Any): json compatible data
    """
    path = os.path.abspath(path)
    filepath = os.path.join(path, filename)

    cv2.imwrite(filepath, cv2.cvtColor(data, cv2.COLOR_BGR2RGB))


def merge_dictionaries(dictionary_list: List[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Merges a list of dictionaries"""

    dictionary = {}
    for dict in dictionary_list:
        dictionary.update(dict)

    return dictionary


def initialize_config_file(path: str) -> Dict[str, Any]:
    config_path = os.path.abspath(path)

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Could not find the specified config. file at: {config_path}")

    return OmegaConf.load(config_path)


def init_backbone(model_name: str) -> models.ResNet:

    # Init. parent model
    if model_name == "resnet_18":
        model = models.resnet18(fully_conv=True,
                                pretrained=False,
                                output_stride=8,
                                remove_avg_pool_layer=True)

    elif model_name == "resnet_34":
        model = models.resnet34(fully_conv=True,
                                pretrained=False,
                                output_stride=8,
                                remove_avg_pool_layer=True)
    else:
        model = models.resnet50(fully_conv=True,
                                pretrained=False,
                                output_stride=8,
                                remove_avg_pool_layer=True)

    return model
