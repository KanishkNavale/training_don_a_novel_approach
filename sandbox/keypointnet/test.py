import cv2

from src.keypointnet import load_trained_keypoint_model
from src.inspector import DescritporInspectorApp

if __name__ == "__main__":
    image = cv2.imread("dataset/rgbs/01_rgb.png")
    trained_model = load_trained_keypoint_model("sandbox/keypointnet/keypointnet-config.yaml",
                                                "trained_keynet_d64.ckpt",)
    

    descriptors = trained_model.compute_descriptors_from_numpy_image(image)

    app = DescritporInspectorApp(rgb_a=image,
                                 rgb_b=image,
                                 descriptor_a=descriptors,
                                 descriptor_b=descriptors)
    app.run()
