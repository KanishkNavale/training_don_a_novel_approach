import cv2

from src.keypointnet import load_trained_don_model
from src.inspector import DescritporInspectorApp

if __name__ == "__main__":
    image = cv2.imread("dataset/298_rgb.png")
    trained_model = load_trained_don_model("sandbox/keypointnet/keypointnet-config.yaml",
                                           "/home/kanishk/Desktop/trained_keynet.ckpt",)

    descriptors = trained_model.compute_dense_local_descriptors(image)

    app = DescritporInspectorApp(rgb_a=image,
                                 rgb_b=image,
                                 descriptor_a=descriptors,
                                 descriptor_b=descriptors)
    app.run()
