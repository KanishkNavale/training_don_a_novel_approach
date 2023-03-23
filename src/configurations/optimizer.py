from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    name: str
    learning_rate: float
    enable_schedular: bool
    schedular_step_size: float

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> OptimizerConfig:
        return cls(**dictionary)
