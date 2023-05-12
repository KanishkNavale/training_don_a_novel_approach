from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass

import torch


@dataclass
class KeypointNet:
    n_keypoints: int
    bottleneck_dimension: int
    backbone: str
    debug: str
    debug_path: str

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> KeypointNet:
        return cls(**dictionary)



@dataclass
class Loss:
    multiview_consistency: float
    spatial_distribution: float
    separation: float
    silhouette: float
    pose: float

    reduction: str
    margin: float

    @property
    def loss_ratios_as_tensor(self) -> torch.Tensor:
        return torch.as_tensor([self.multiview_consistency,
                                self.pose,
                                self.separation,
                                self.silhouette,
                                self.spatial_distribution])

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> Loss:
        return cls(**dictionary)


@dataclass
class KeypointNetConfig:

    keypointnet: KeypointNet
    loss: Loss

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> KeypointNetConfig:
        keypointnet = KeypointNet.from_dictionary(dictionary.keypointnet)
        loss = Loss.from_dictionary(dictionary.loss)

        return cls(keypointnet=keypointnet, loss=loss)
