from typing import Any, Dict
import os

from omegaconf import OmegaConf

import src.__models as models


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
