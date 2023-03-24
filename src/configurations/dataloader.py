from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass
import os


@dataclass
class DataLoaderConfig:
    directory: str

    depth_ratio: str
    test_size: float
    shuffle: int

    n_workers: int
    batch_size: int

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> DataLoaderConfig:
        return cls(**dictionary.dataloader)

    def __post_init__(self):
        self.directory = os.path.abspath(self.directory)
