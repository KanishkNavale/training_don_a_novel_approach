import cv2
import numpy as np

from src.don import load_trained_don_model
from src.inspector import DescritporInspectorApp

if __name__ == "__main__":
    image = cv2.imread("dataset/298_rgb.png")
    trained_model = load_trained_don_model("sandbox/don/don-config.yaml")

    descriptors = trained_model.compute_descriptors_from_numpy_image(image)

    app = DescritporInspectorApp(rgb_a=image,
                                 rgb_b=image,
                                 descriptor_a=descriptors,
                                 descriptor_b=descriptors)
    app.run()
