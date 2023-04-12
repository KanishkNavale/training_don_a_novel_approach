from __future__ import annotations
from typing import Dict, Any, Union
from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    name: str
    learning_rate: float
    enable_schedule: bool
    schedular_step_size: float
    gamma: float
    weight_decay: Union[float, None, str] = None

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> OptimizerConfig:
        return cls(**dictionary.optimizer)

    def __post_init__(self):
        if self.name not in ["Adam", "Nadam"]:
            raise NotImplementedError(f"This optimizer: {self.name} is not implemented")

        if self.enable_schedule:
            if self.schedular_step_size <= 0 or self.schedular_step_size is None:
                raise ValueError(f"The schedular step size: {self.schedular_step_size} must be greater than 0")

            if self.gamma <= 0 or self.gamma is None:
                raise ValueError(f"The gamma: {self.gamma} must be greater than 0")

        if not isinstance(self.weight_decay, float):
            if self.weight_decay == 'None' or self.weight_decay == 0:
                self.weight_decay = 0
            else:
                raise ValueError(f"The weight decay: {self.weight_decay} must be greater than or equal to 0")
