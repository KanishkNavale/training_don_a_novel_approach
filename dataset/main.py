import blenderproc as bproc

from typing import List
import numpy as np
import cv2
import pickle

bproc.init()


def generate_erode_mask(mask: np.ndarray):
    kernel = np.ones((5, 5), np.uint8)
    erode = cv2.erode(mask, kernel, cv2.BORDER_REFLECT)
    mask = np.where(erode >= np.max(erode), 255, 0)
    return mask


if __name__ == "__main__":

    # Load object
    objs = bproc.loader.load_obj("models/9c225cd8f50b7c353d9199581d0f4b4/models/model_normalized.obj")
    object = objs[0]
    object.set_cp("category_id", 0)
    object.set_location([0.0, 0.0, 0.0])
    object.set_rotation_euler([-np.pi / 2, 0.0, 0.0])

    # Freeze lights
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([0, 0, 15])
    light.set_energy(3000)

    # Freeze camera location
    bproc.camera.set_resolution(640, 480)
    K = np.array([[638.0, 0.0, 300],
                  [0.0, 637.0, 295],
                  [0.0, 0.0, 1.0]])

    bproc.camera.set_intrinsics_from_K_matrix(K, 640, 480)
    bproc.renderer.enable_depth_output(activate_antialiasing=True)

    poi = bproc.object.compute_poi([object])

    list_of_poses: List[np.ndarray] = []
    for i in range(100):
        # Move and place the camera
        random_transformation = np.random.uniform([-0.1, -0.1, 2], [0.1, 0.1, 3])
        random_rotation = np.random.uniform([0, 0, -np.pi / 3], [0, 0, np.pi / 3])

        cam_pose = bproc.math.build_transformation_mat(random_transformation, random_rotation)
        bproc.camera.add_camera_pose(cam_pose)

        cam_pose = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam_pose, ["-Y", "X", "-Z"])

        list_of_poses.append(cam_pose)

    data = bproc.renderer.render()
    data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))

    for i in range(100):
        idx = np.random.randint(0, len(data) - 1)
        next_idx = np.random.randint(0, len(data) - 1)

        data_slice = {"rgb_a": data["colors"][idx],
                      "depth_a": data["depth"][idx],
                      "mask_a": generate_erode_mask(data["instance_segmaps"][idx]),
                      "pose_a": list_of_poses[idx],
                      "rgb_b": data["colors"][next_idx],
                      "depth_b": data["depth"][next_idx],
                      "mask_b": generate_erode_mask(data["instance_segmaps"][next_idx]),
                      "pose_b": list_of_poses[next_idx],
                      "intrinsics": K
                      }

        with open(f'dataset/{i+199}.pkl', 'wb') as f:
            pickle.dump(data_slice, f)
