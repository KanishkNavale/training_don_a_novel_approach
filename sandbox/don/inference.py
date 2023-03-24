import cv2
import matplotlib.pylab as plt

from src.don import load_trained_don_model

if __name__ == "__main__":
    image = cv2.imread("dataset/298_rgb.png") / 255
    trained_model = load_trained_don_model("/tmp/dense_mf.ckpt", "sandbox/don/don-config.yaml")

    descriptors = trained_model.compute_descriptors_from_numpy(image)

    plt.imshow(descriptors)
    plt.show()