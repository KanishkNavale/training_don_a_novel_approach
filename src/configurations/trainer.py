from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass
import os


@dataclass
class TrainerConfig:
    precision: str
    model_checkpoint_name: str
    epochs: int
    enable_logging: bool
    tensorboard_path: bool
    training_directory: str
    enable_checkpointing: bool
    model_path: str
    logging_frequency: bool
    validation_frequency: int

    def __post_init__(self):
        self.training_directory = os.path.abspath(self.training_directory)
        self.model_path = os.path.abspath(self.model_path)
        self.tensorboard_path = os.path.abspath(self.tensorboard_path)

        if self.precision not in ["medium", "high"]:
            raise ValueError("Precision must be in {medium, high}")

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> TrainerConfig:
        return cls(**dictionary.trainer)
