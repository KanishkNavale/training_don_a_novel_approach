import cv2
import numpy as np
import torch

from src.don.synthetizer import augment_images_and_compute_correspondences as sythethize
from src.keypointnet.geometry_transformations import kabsch_tranformation

if __name__ == "__main__":

    image = cv2.imread("/home/kanishk/sereact/training_don_while_not_training_don/dataset/tester.png")
    mask = cv2.imread("/home/kanishk/sereact/training_don_while_not_training_don/dataset/tester.png")[..., 0]

    image = torch.as_tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    mask = torch.as_tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0)

    image_a, matches_a, _, images_b, matches_b, _ = sythethize(image, mask, n_correspondences=25)

    rotation, translation = kabsch_tranformation(matches_a, matches_b)

    print(matches_b)

    print(rotation @ matches_a.type(torch.float32) + translation)
