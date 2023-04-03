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
        self.random_backgrounds = [os.path.join("dataset/random_backgrounds", file)
                                   for file in os.listdir("dataset/random_backgrounds")]

    def __len__(self) -> int:
        return len(self.data)

    def _augment_image(self,
                       image: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:

        # Masked image
        if 0 == np.random.randint(0, 3):
            tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, 3)
            masked_image = image * tiled_mask
            image = masked_image

        # Noisy background
        elif 0 == np.random.randint(0, 3):
            tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, 3)
            masked_image = image * tiled_mask

            random_image = torch.rand_like(masked_image)
            masked_random_image = torch.where(masked_image != torch.zeros(3, dtype=masked_image.dtype),
                                              torch.zeros(3, dtype=masked_image.dtype),
                                              random_image)

            image = masked_random_image + masked_image

        # Random Background
        else:
            tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, 3)
            masked_image = image * tiled_mask

            random_image = cv2.imread(random.choice(self.random_backgrounds)) / 255
            random_image = torch.as_tensor(cv2.resize(random_image,
                                                      (image.shape[1], image.shape[0])),
                                           dtype=image.dtype)

            masked_random_image = torch.where(masked_image != torch.zeros(3, dtype=masked_image.dtype),
                                              torch.zeros(3, dtype=masked_image.dtype),
                                              random_image)

            image = masked_image + masked_random_image

        # Gaussian Blur
        if 0 == np.random.randint(0, 3):
            blurred_image: torch.Tensor = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(image.permute(2, 0, 1))
            image = blurred_image.permute(1, 2, 0)

        # Greyscale augmentation
        elif 0 == np.random.randint(0, 3):
            grayscale: torch.Tensor = T.Grayscale()(image.permute(2, 0, 1))
            image = grayscale.tile(3, 1, 1).permute(1, 2, 0)

        else:
            image = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(image.permute(2, 0, 1))
            image = image.permute(1, 2, 0)

        return image

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

        rgb_a = self._augment_image(rgb_a, mask_a)
        rgb_b = self._augment_image(rgb_b, mask_b)

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
            "Masks-B": mask_b
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

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          num_workers=self.config.n_workers,
                          batch_size=self.config.batch_size,
                          pin_memory=True)
