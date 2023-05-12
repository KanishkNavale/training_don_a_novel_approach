from __future__ import annotations
from typing import Dict, Any, Union

from dataclasses import dataclass


@dataclass
class DON:
    descriptor_dimension: int
    backbone: str
    n_correspondence: int
    debug: bool = False
    debug_path: str = 'tmp'

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> DON:
        return cls(**dictionary)


@dataclass
class Loss:
    name: str
    reduction: str
    temperature: Union[float, None] = None

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> Loss:
        return cls(**dictionary)

    def __post_init__(self):
        if self.name not in ["pixelwise_correspondence_loss", "pixelwise_ntxent_loss"]:
            raise NotImplementedError(f"The specified loss function: {self.name} is not implemented")

        if self.name == "pixelwise_ntxent_loss" and self.temperature is None:
            raise ValueError(f"The loss function: {self.name} needs a temperatute scalar to be initialized")

        if self.reduction not in ["sum", "mean"]:
            raise NotImplementedError(f"The specified loss function: {self.reduction} is not implemented")


@dataclass
class DONConfig:

    don: DON
    loss: Loss

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> DONConfig:
        don = DON.from_dictionary(dictionary.don)
        loss = Loss.from_dictionary(dictionary.loss)

        return cls(don=don, loss=loss)
