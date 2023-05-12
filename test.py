import numpy as np
import cv2
import pickle


if __name__ == "__main__":

    with open("dataset/dataset/298.pkl", 'rb') as f:
        pickled = pickle.load(f)

    rgb_a = pickled["rgb_a"]

    cv2.imshow("rgb_a", rgb_a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
