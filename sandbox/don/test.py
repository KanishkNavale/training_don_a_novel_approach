import cv2
import numpy as np

from src.don import load_trained_don_model

if __name__ == "__main__":
    image = cv2.imread("dataset/298_rgb.png")
    trained_model = load_trained_don_model("sandbox/don/don-config.yaml")

    descriptors = trained_model.compute_descriptors_from_numpy(image)
    descriptors = cv2.normalize(descriptors, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow("Image-Descriptor Pair", np.hstack([image, descriptors]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
