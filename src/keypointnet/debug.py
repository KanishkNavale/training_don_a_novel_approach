from typing import Union, List
import os

import cv2
import torch
import numpy as np

from src.don.debug import _channel_first_tensor_to_channel_last_numpy, _convert_numpy_image_to_cv2_image


def debug_keypoints(images_a: torch.Tensor,
                    matches_a: torch.Tensor,
                    images_b: torch.Tensor,
                    matches_b: torch.Tensor,
                    colors: List[np.ndarray],
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

        for (match_a,
             match_b,
             color) in zip(matches_a,
                           matches_b,
                           colors):
            ua, va = match_a[1], match_a[0]
            ub, vb = match_b[1], match_b[0]

            cv2.circle(image_a, (ua, va), radius=1, color=color, thickness=2)
            cv2.circle(image_b, (ub, vb), radius=1, color=color, thickness=2)

        if debug_image is None:
            debug_image = np.hstack([image_a, image_b])
        else:
            debug_image = np.vstack([debug_image, np.hstack([image_a, image_b])])

    cv2.imwrite(os.path.join(debug_path, "debug_keypoints.png"),
                cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
