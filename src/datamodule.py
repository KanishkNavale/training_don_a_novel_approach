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
                 data: List[str],
                 config: DataLoaderConfig) -> None:
        self.data = data
        self.config = config
        self.random_backgrounds = [os.path.join("dataset/random_backgrounds/images", file)
                                   for file in os.listdir("dataset/random_backgrounds/images")]

        # deconstruct the config
        self.random_prob = int(1.0 / self.config.random_hintergrund_probability)
        self.noisy_prob = int(1.0 / self.config.noisy_hintergrund_probability)
        self.masked_prob = int(1.0 / self.config.masked_hintergrund_probability)

        self.greyscale_prob = int(1.0 / self.config.greyscale_probability)
        self.colorjitter_prob = int(1.0 / self.config.colorjitter_probability)
        self.gaussian_blur_prob = int(1.0 / self.config.gaussian_blur_probability)

    def __len__(self) -> int:
        return len(self.data)

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

        # Greyscale augmentation
        elif 0 == np.random.randint(0, self.colorjitter_prob):
            grayscale: torch.Tensor = T.Grayscale()(image.permute(2, 0, 1))
            image = grayscale.tile(3, 1, 1).permute(1, 2, 0)

        else:
            image = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(image.permute(2, 0, 1))
            image = image.permute(1, 2, 0)

        return image, background

    @staticmethod
    def _add_mask(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, input.shape[-1])
        return input * tiled_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        with open(self.data[idx], 'rb') as f:
            pickled = pickle.load(f)

        rgb_a = pickled["rgb_a"]
        depth_a = pickled["depth_a"]
        mask_a = pickled["mask_a"]
        extrinsic_a = pickled["pose_a"]
        intrinsic_a = pickled["intrinsics"]

        rgb_b = pickled["rgb_b"]
        depth_b = pickled["depth_b"]
        mask_b = pickled["mask_b"]
        extrinsic_b = pickled["pose_b"]
        intrinsic_b = pickled["intrinsics"]

        rgb_a = torch.as_tensor(rgb_a / 255, dtype=torch.float32)
        depth_a = torch.as_tensor(depth_a / self.config.depth_ratio, dtype=torch.float32)
        mask_a = torch.as_tensor(mask_a / 255, dtype=torch.float32)
        extrinsic_a = torch.as_tensor(extrinsic_a, dtype=torch.float32)
        intrinsic_a = torch.as_tensor(intrinsic_a, dtype=torch.float32)

        rgb_b = torch.as_tensor(rgb_b / 255, dtype=torch.float32)
        depth_b = torch.as_tensor(depth_b / self.config.depth_ratio, dtype=torch.float32)
        mask_b = torch.as_tensor(mask_b / 255, dtype=torch.float32)
        extrinsic_b = torch.as_tensor(extrinsic_b, dtype=torch.float32)
        intrinsic_b = torch.as_tensor(intrinsic_b, dtype=torch.float32)

        # Load a random background
        background = cv2.imread(random.choice(self.random_backgrounds)) / 255
        background = cv2.resize(background, (rgb_b.shape[1], rgb_b.shape[0]))
        background = torch.as_tensor(background, dtype=rgb_b.dtype)

        rgb_a, _ = self._augment_image(rgb_a, mask_a, background)
        rgb_b, background = self._augment_image(rgb_b, mask_b, background)

        return {
            "RGBs-A": rgb_a.permute(2, 0, 1),
            "RGBs-B": rgb_b.permute(2, 0, 1),
            "Depths-A": depth_a,
            "Depths-B": depth_b,
            "Intrinsics-A": intrinsic_a,
            "Intrinsics-B": intrinsic_b,
            "Extrinsics-A": extrinsic_a,
            "Extrinsics-B": extrinsic_b,
            "Masks-A": mask_a,
            "Masks-B": mask_b,
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
        directory = self.config.directory
        self.files = sorted([os.path.join(directory, file) for file in os.listdir(directory)])

    def setup(self, stage: str = None):
        # Create training, validation datasplits
        (train_files, val_files) = train_test_split(self.files,
                                                    shuffle=self.config.shuffle,
                                                    test_size=self.config.test_size)

        if stage == 'fit':
            self.training_dataset = Dataset(train_files, self.config)
            self.validation_dataset = Dataset(val_files, self.config)

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
