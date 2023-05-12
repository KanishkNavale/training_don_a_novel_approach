from typing import Union
import os

import cv2
import numpy as np
from sklearn.decomposition import PCA
import torch


def _channel_first_tensor_to_channel_last_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.permute(0, 2, 3, 1).detach().cpu().numpy()


def _convert_numpy_image_to_cv2_image(numpy_image: np.ndarray) -> np.ndarray:
    return cv2.normalize(numpy_image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def debug_correspondences(images_a: torch.Tensor,
                          matches_a: torch.Tensor,
                          images_b: torch.Tensor,
                          matches_b: torch.Tensor,
                          debug_path: str) -> None:

    debug_image: Union[None, np.ndarray] = None

    with torch.no_grad():
        detached_image_a = _channel_first_tensor_to_channel_last_numpy(images_a)
        detached_image_b = _channel_first_tensor_to_channel_last_numpy(images_b)
        detached_matches_a = matches_a.detach().type(torch.int64).cpu().numpy()
        detached_matches_b = matches_b.detach().type(torch.int64).cpu().numpy()

    for (image_a,
         image_b,
         matches_a,
         matches_b) in zip(detached_image_a,
                           detached_image_b,
                           detached_matches_a,
                           detached_matches_b):
        image_a = _convert_numpy_image_to_cv2_image(image_a)
        image_b = _convert_numpy_image_to_cv2_image(image_b)
        for match_a, match_b in zip(matches_a, matches_b):
            random_color = np.random.randint(0, 255, size=(3,)).tolist()
            ua, va = match_a[1], match_a[0]
            ub, vb = match_b[1], match_b[0]
            cv2.circle(image_a, (ua, va), radius=1, color=random_color, thickness=2)
            cv2.circle(image_b, (ub, vb), radius=1, color=random_color, thickness=2)

        if debug_image is None:
            debug_image = np.hstack([image_a, image_b])
        else:
            debug_image = np.vstack([debug_image, np.hstack([image_a, image_b])])

    cv2.imwrite(os.path.join(debug_path, "debug_correspondences.png"),
                cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))


def debug_descriptors(images_a: torch.Tensor,
                      dense_descriptors_a: torch.Tensor,
                      images_b: torch.Tensor,
                      dense_descriptors_b: torch.Tensor,
                      debug_path: str) -> None:

    debug_image: Union[None, np.ndarray] = None

    with torch.no_grad():
        detached_image_a = _channel_first_tensor_to_channel_last_numpy(images_a)
        detached_image_b = _channel_first_tensor_to_channel_last_numpy(images_b)
        detached_descriptors_a = _channel_first_tensor_to_channel_last_numpy(dense_descriptors_a)
        detached_descriptors_b = _channel_first_tensor_to_channel_last_numpy(dense_descriptors_b)

    for (image_a,
         image_b,
         descriptors_a,
         descriptors_b) in zip(detached_image_a,
                               detached_image_b,
                               detached_descriptors_a,
                               detached_descriptors_b):
        image_a = _convert_numpy_image_to_cv2_image(image_a)
        image_b = _convert_numpy_image_to_cv2_image(image_b)

        if descriptors_a.shape[-1] > 3:
            pca_computer = PCA(n_components=3)

            flat_reduced_descriptors_a = pca_computer.fit_transform(descriptors_a.reshape(-1, descriptors_a.shape[-1]))
            flat_reduced_descriptors_b = pca_computer.transform(descriptors_b.reshape(-1, descriptors_a.shape[-1]))

            descriptors_a = flat_reduced_descriptors_a.reshape(descriptors_a.shape[0],
                                                               descriptors_a.shape[1],
                                                               3)
            descriptors_b = flat_reduced_descriptors_b.reshape(descriptors_b.shape[0],
                                                               descriptors_b.shape[1],
                                                               3)

        descriptors_a = _convert_numpy_image_to_cv2_image(descriptors_a)
        descriptors_b = _convert_numpy_image_to_cv2_image(descriptors_b)

        image_slice = np.hstack([image_a, image_b])
        descriptor_slice = np.hstack([descriptors_a, descriptors_b])
        debug_slice = np.hstack([image_slice, descriptor_slice])

        if debug_image is None:
            debug_image = debug_slice
        else:
            debug_image = np.vstack([debug_image, debug_slice])

    cv2.imwrite(os.path.join(debug_path, "debug_descriptors.png"),
                cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
