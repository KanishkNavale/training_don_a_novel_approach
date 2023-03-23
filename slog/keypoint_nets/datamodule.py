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

from slog.keypoint_nets.configurations.config import KeypointNetConfig
from slog.utils import initialize_config_file

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class KeypointNetDataset(Dataset):
    def __init__(self,
                 data: List[str],
                 config: KeypointNetConfig) -> None:
        self.data = data
        self.config = config
        self.random_backgrounds = [os.path.join("dataset/random_backgrounds", file) for file in os.listdir("dataset/random_backgrounds")]

    def __len__(self) -> int:
        return len(self.data)

    def _augment_image(self,
                       image: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:

        output = image

        # Random Background
        if 1 == np.random.randint(0, 5):
            tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, 3)
            masked_image = image * tiled_mask

            random_image = cv2.imread(random.choice(self.random_backgrounds)) / 255
            random_image = torch.as_tensor(cv2.resize(random_image, (image.shape[1], image.shape[0])), dtype=image.dtype)

            masked_random_image = torch.where(masked_image != torch.zeros(3, dtype=masked_image.dtype),
                                              torch.zeros(3, dtype=masked_image.dtype),
                                              random_image)

            output = masked_image + masked_random_image

            return output

        # Gaussian Blur
        elif 1 == np.random.randint(0, 5):
            blurred_image: torch.Tensor = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(image.permute(2, 0, 1))
            return blurred_image.permute(1, 2, 0)

        # Greyscale augmentation
        elif 1 == np.random.randint(0, 5):
            grayscale: torch.Tensor = T.Grayscale()(image.permute(2, 0, 1))
            output = grayscale.tile(3, 1, 1).permute(1, 2, 0)

        # Noisy background
        elif 1 == np.random.randint(0, 3):
            tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, 3)
            masked_image = image * tiled_mask

            random_image = torch.rand_like(masked_image)
            masked_random_image = torch.where(masked_image != torch.zeros(3, dtype=masked_image.dtype),
                                              torch.zeros(3, dtype=masked_image.dtype),
                                              random_image)

            output = masked_random_image + masked_image

        # Masked image
        elif 1 == np.random.randint(0, 5):
            tiled_mask = mask.unsqueeze(dim=-1).tile(1, 1, 3)
            masked_image = image * tiled_mask
            output = masked_image

        return output

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
        depth_a = torch.as_tensor(depth_a / self.config.datamodule.depth_ratio, dtype=torch.float32)
        mask_a = torch.as_tensor(mask_a / 255, dtype=torch.float32)
        extrinsic_a = torch.as_tensor(extrinsic_a, dtype=torch.float32)
        intrinsic_a = torch.as_tensor(intrinsic_a, dtype=torch.float32)

        rgb_b = torch.as_tensor(rgb_b / 255, dtype=torch.float32)
        depth_b = torch.as_tensor(depth_b / self.config.datamodule.depth_ratio, dtype=torch.float32)
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


class DataModuleKeypointNet(pl.LightningDataModule):
    def __init__(self, yaml_config_path: str) -> None:

        config_dictionary = initialize_config_file(yaml_config_path)
        config = KeypointNetConfig.from_dictionary(config_dictionary)
        self.config = config

        # Default values
        self._log_hyperparams = self.config.datamodule.n_workers
        self.prepare_data_per_node = True

    def prepare_data(self) -> None:
        # Reading RGBD data
        directory = self.config.datamodule.directory
        self.files = sorted([os.path.join(directory, file) for file in os.listdir(directory)])

    def setup(self, stage: str = None):
        # Create training, validation datasplits
        (train_files,
         val_files,) = train_test_split(self.files,
                                        shuffle=self.config.datamodule.shuffle,
                                        test_size=self.config.datamodule.test_size)

        if stage == 'fit':
            self.training_dataset = KeypointNetDataset(train_files,
                                                       self.config)

            self.validation_dataset = KeypointNetDataset(val_files,
                                                         self.config)

    def train_dataloader(self):
        return DataLoader(self.training_dataset,
                          num_workers=self.config.datamodule.n_workers,
                          batch_size=self.config.datamodule.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset,
                          num_workers=self.config.datamodule.n_workers,
                          batch_size=self.config.datamodule.batch_size,
                          pin_memory=True)
