from typing import List, Dict, Any
import os

import numpy as np
import random
import cv2
from sklearn.model_selection import train_test_split
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pytorch_lightning as pl

from src.configurations import DataLoaderConfig

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class Dataset(Dataset):
    def __init__(self,
                 rgbs: List[str],
                 masks: List[str],
                 config: DataLoaderConfig) -> None:
        self.rgbs = rgbs
        self.masks = masks
        self.config = config
        self.random_backgrounds = [os.path.join(self.config.random_background_directory, file)
                                   for file in os.listdir(self.config.random_background_directory)]

        # deconstruct the config
        self.random_prob = int(1.0 / self.config.random_hintergrund_probability)
        self.noisy_prob = int(1.0 / self.config.noisy_hintergrund_probability)
        self.masked_prob = int(1.0 / self.config.masked_hintergrund_probability)

        self.greyscale_prob = int(1.0 / self.config.greyscale_probability)
        self.colorjitter_prob = int(1.0 / self.config.colorjitter_probability)
        self.gaussian_blur_prob = int(1.0 / self.config.gaussian_blur_probability)

    def __len__(self) -> int:
        return len(self.rgbs)

    def _augment_image(self,
                       image: torch.Tensor,
                       mask: torch.Tensor,
                       background: torch.Tensor) -> torch.Tensor:

        # Precomputations
        inverted_mask = torch.where(mask == 0.0, torch.ones_like(mask), torch.zeros_like(mask))
        inverted_mask = inverted_mask.unsqueeze(dim=-1).tile(1, 1, 3)
        tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, 3)
        masked_image = image * tiled_mask

        # Masked image
        if 0 == np.random.randint(0, self.masked_prob):
            background = torch.zeros_like(image)

            image = masked_image

        # Noisy background
        elif 0 == np.random.randint(0, self.noisy_prob):
            background = torch.rand_like(image)

            random_image = torch.rand_like(masked_image)
            noisy_hintergrund_image = random_image * inverted_mask

            image = noisy_hintergrund_image + masked_image

        # Random Background
        else:
            random_hintergrund_image = background * inverted_mask
            image = masked_image + random_hintergrund_image

        # Gaussian Blur
        if 0 == np.random.randint(0, self.gaussian_blur_prob):
            blurred_image: torch.Tensor = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(image.permute(2, 0, 1))
            image = blurred_image.permute(1, 2, 0)

        # Color augmentation
        elif 0 == np.random.randint(0, self.colorjitter_prob):
            image = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(image.permute(2, 0, 1))
            image = image.permute(1, 2, 0)

        else:
            grayscale: torch.Tensor = T.Grayscale()(image.permute(2, 0, 1))
            image = grayscale.tile(3, 1, 1).permute(1, 2, 0)

        return image, background

    @staticmethod
    def _add_mask(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, input.shape[-1])
        return input * tiled_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        rgb = cv2.imread(self.rgbs[idx])
        mask = cv2.imread(self.masks[idx])[..., 0]

        rgb = torch.as_tensor(rgb / 255, dtype=torch.float32)
        mask = torch.as_tensor(mask / 255, dtype=torch.float32)

        # Load a random background
        background = cv2.imread(random.choice(self.random_backgrounds)) / 255
        background = cv2.resize(background, (rgb.shape[1], rgb.shape[0]))
        background = torch.as_tensor(background, dtype=rgb.dtype)

        return {
            "RGBs": rgb.permute(2, 0, 1),
            "Masks": mask,
            "Random-Backgrounds": background.permute(2, 0, 1)
        }


class DataModule(pl.LightningDataModule):
    def __init__(self, config: DataLoaderConfig) -> None:

        self.config = config

        # Default values
        self._log_hyperparams = self.config.n_workers
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def prepare_data(self) -> None:
        # Reading RGBD data
        self.rgbs = sorted([os.path.join(self.config.rgb_directory, file) for file in os.listdir(self.config.rgb_directory)])
        self.masks = sorted([os.path.join(self.config.mask_directory, file) for file in os.listdir(self.config.mask_directory)])

    def setup(self, stage: str = None):
        # Create training, validation datasplits
        (train_rgs, val_rgbs, train_masks, val_masks) = train_test_split(self.rgbs,
                                                                         self.masks,
                                                                         shuffle=self.config.shuffle,
                                                                         test_size=self.config.test_size)

        if stage == 'fit':
            self.training_dataset = Dataset(train_rgs, train_masks, self.config)
            self.validation_dataset = Dataset(val_rgbs, val_masks, self.config)

    def train_dataloader(self):
        return DataLoader(self.training_dataset,
                          num_workers=self.config.n_workers,
                          batch_size=self.config.batch_size,
                          pin_memory=True)

    def val_dataloader(self, batch_size: int = None):
        return DataLoader(self.validation_dataset,
                          num_workers=self.config.n_workers,
                          batch_size=self.config.batch_size if batch_size is None else batch_size,
                          pin_memory=True)
