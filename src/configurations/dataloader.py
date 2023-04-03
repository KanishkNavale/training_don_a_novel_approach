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

    random_hintergrund_probability: float
    noisy_hintergrund_probability: float
    masked_hintergrund_probability: float

    greyscale_probability: float
    colorjitter_probability: float
    gaussian_blur_probability: float

    @classmethod
    def from_dictionary(cls, dictionary: Dict[str, Any]) -> DataLoaderConfig:
        return cls(**dictionary.dataloader)

    def __post_init__(self):
        self.directory = os.path.abspath(self.directory)

        if self.random_hintergrund_probability + self.noisy_hintergrund_probability + self.masked_hintergrund_probability != 1.0:
            raise ValueError("The probabilities of the background augmentation must sum up to 1.")

        if self.greyscale_probability + self.colorjitter_probability + self.gaussian_blur_probability != 1.0:
            raise ValueError("The probabilities of the image augmentation must sum up to 1.")
