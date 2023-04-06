from typing import List

import cv2
import numpy as np
import torch
import pygame

from src.utils import convert_tensor_to_cv2
from src.distances import compute_keypoint_confident_expectation
from src.renderers import render_spatial_distribution

pygame.init()


class PoseGraphGenerator:
    def __init__(self,
                 rgb_a: torch.Tensor,
                 descriptor_a: torch.Tensor,
                 rgb_b: torch.Tensor,
                 descriptor_b: torch.Tensor) -> None:

        self.rgb = torch.hstack([rgb_a, rgb_b])
        self.descriptors = torch.hstack([descriptor_a, descriptor_b])

        self.temp = torch.as_tensor(2.0, device=self.descriptors.device, dtype=self.descriptors.dtype)
        self.confidence = torch.as_tensor(0.1, device=self.descriptors.device, dtype=self.descriptors.dtype)

        # Set up display
        pygame.display.set_caption('Inspector de la Descriptors')
        width, height = self.descriptors.shape[1], self.descriptors.shape[0]
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.flip()

        # CPU copies of tensors
        self.cpu_temp = self.temp.item()
        self.cpu_conf = self.confidence.item()
        self.cpu_image = convert_tensor_to_cv2(self.rgb)

        # app defaults
        self.enable_spatial_expectation = False
        self.spat_probs = None

    def _update_display(self, image: np.ndarray) -> None:
        """Updates the pygame window with information

        Args:
            image (np.ndarray): image to render in np.uint8 format
        """
        pygame_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pygame_image = pygame.surfarray.make_surface(pygame_image.swapaxes(0, 1))

        self.screen.blit(pygame_image, (0, 0))
        pygame.display.update()

    def _compute_spatial_expectation(self) -> np.ndarray:
        """Computes spatial expecation of keypoint

        Returns:
            np.ndarray: expecation of keypoint in np.uint8 format
        """
        v, u = pygame.mouse.get_pos()
        keypoint = self.descriptors[u][v]

        weights = compute_keypoint_confident_expectation(self.descriptors, keypoint, self.temp, self.confidence)
        spat_probs = render_spatial_distribution(self.rgb, weights)

        return spat_probs

    def _toggle_spatial_expectation(self, event: pygame.event) -> None:
        """Toggles state to compute spatial expectation

        Args:
            event (pygame.event): pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                self.enable_spatial_expectation = not self.enable_spatial_expectation

    def _update_temperature(self, event: pygame.event) -> None:
        """Updates temperature parameter of the kernel function
        Args:
            event (pygame.event): pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_KP8:
                self.temp += 0.1
                self.cpu_temp += 0.1
            elif event.key == pygame.K_KP2:
                self.temp -= 0.1
                self.cpu_temp -= 0.1

    def _update_confidence(self, event: pygame.event) -> None:
        """Updates confidence parameter of the kernel function
        Args:
            event (pygame.event): pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_KP4:
                self.confidence += 0.1
                self.cpu_conf += 0.1
            elif event.key == pygame.K_KP6:
                self.confidence -= 0.1
                self.cpu_conf -= 0.1

        self.cpu_conf = np.clip(self.cpu_conf, 0.0, 1.0)
        self.confidence = torch.clamp(self.confidence, 0.0, 1.0)

    def run(self) -> None:
        running = True
        image = None

        while running:
            for event in pygame.event.get():
                self._toggle_spatial_expectation(event)

                self._update_confidence(event)
                self._update_temperature(event)

                if self.enable_spatial_expectation:
                    image = self._compute_spatial_expectation()
                if image is None or not self.enable_spatial_expectation:
                    image = self.cpu_image

                self._update_display(image)

                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

        pygame.quit()
