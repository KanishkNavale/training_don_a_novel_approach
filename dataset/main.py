import blenderproc as bproc

from typing import List
import numpy as np
import cv2
import pickle

bproc.init()


if __name__ == "__main__":

    # Load object
    objs = bproc.loader.load_obj("models/357c2a333ffefc3e90f80ab08ae6ce2/models/model_normalized.obj")
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

    cam_pose = bproc.math.build_transformation_mat([0.1, 0.1, 2.5], [0, 0, 0])
    bproc.camera.add_camera_pose(cam_pose)
    bproc.camera.set_intrinsics_from_K_matrix(K, 640, 480)
    bproc.renderer.enable_depth_output(activate_antialiasing=True)

    poi = bproc.object.compute_poi([object])

    data = bproc.renderer.render()
    data.update(bproc.renderer.render_segmap(map_by=["class", "instance", "name"]))

    cv2.imwrite("12_rgb.png", data["colors"][0])
    cv2.imwrite("12_mask.png", data["instance_segmaps"][0] * 255)
