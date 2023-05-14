from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass
import os


@dataclass
class DataLoaderConfig:
    rgb_directory: str
    mask_directory: str
    depth_directory: str
    extrinsic_directory: str
    camera_intrinsics_numpy_text: str
    random_background_directory: str

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
        self.rgb_directory = os.path.abspath(self.rgb_directory)
        self.mask_directory = os.path.abspath(self.mask_directory)
        self.depth_directory = os.path.abspath(self.depth_directory)
        self.extrinsic_directory = os.path.abspath(self.extrinsic_directory)
        self.camera_intrinsics_numpy_text = os.path.abspath(self.camera_intrinsics_numpy_text)
        self.random_background_directory = os.path.abspath(self.random_background_directory)

        if self.random_hintergrund_probability + self.noisy_hintergrund_probability + self.masked_hintergrund_probability != 1.0:
            raise ValueError("The probabilities of the background augmentation must sum up to 1.")

        if self.greyscale_probability + self.colorjitter_probability + self.gaussian_blur_probability != 1.0:
            raise ValueError("The probabilities of the image augmentation must sum up to 1.")
