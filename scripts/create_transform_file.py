import sys
import shutil
import os
import json
import numpy as np

def main(argv):
    dir_name = argv
    camera_pathes_path = os.path.join(dir_name, 'camera_path.json')
    dataparser_transforms_path = os.path.join(dir_name, 'dataparser_transforms.json')
    transforms_path = os.path.join(dir_name, 'transforms.json')
    backup_file_path = os.path.join(dir_name, 'transforms_backup')
    if not os.path.exists(backup_file_path):
        os.mkdir(backup_file_path)
    shutil.copy(camera_pathes_path, backup_file_path)
    shutil.copy(dataparser_transforms_path, backup_file_path)
    shutil.copy(transforms_path, backup_file_path)

    with open(os.path.join(camera_pathes_path), 'r') as f:
        camera_pathes = json.load(f)
    with open(os.path.join(dataparser_transforms_path), 'r') as f:
        dataparser_transforms = json.load(f)
    with open(os.path.join(transforms_path), 'r') as f:
        transforms = json.load(f)

    new_transforms = transforms
    new_frames = []

    new_camera_pathes = camera_pathes
    new_camera_path = []

    fl_y = transforms["fl_y"]
    h = transforms["h"]

    fov = np.rad2deg(2*np.arctan(h/(2*fl_y)))

    new_transforms["fl_x"] = transforms["fl_y"]
    new_transforms["cx"] = transforms["w"] / 2
    new_transforms["cy"] = transforms["h"] / 2
    new_transforms["k1"] = 0
    new_transforms["k2"] = 0
    new_transforms["p1"] = 0
    new_transforms["p2"] = 0


    for i, camera in enumerate(camera_pathes["keyframes"]):
        frame = {}
        frame["file_path"] = f"images/{i:05d}.png"
        transform_matrix = np.array(eval(camera["matrix"])).reshape((4, 4)).T
        frame["transform_matrix"] = transform_matrix.tolist()
        new_frames.append(frame)

        new_camera = {}
        new_camera["camera_to_world"] = transform_matrix.flatten().tolist()
        new_camera["fov"] = fov
        new_camera["aspect"] = camera["aspect"]
        new_camera_path.append(new_camera)

    new_transforms["frames"] = new_frames
    new_transforms["transform"] = dataparser_transforms["transform"]
    new_transforms["registration_matrix"] = dataparser_transforms["registration_matrix"]
    new_transforms["registration_rot_euler"] = dataparser_transforms["registration_rot_euler"]
    new_transforms["registration_translation"] = dataparser_transforms["registration_translation"]

    new_camera_pathes["camera_path"] = new_camera_path

    print(f"[INFO] writing {len(new_transforms['frames'])} frames to {transforms_path}")
    with open(transforms_path, "w") as outfile:
        json.dump(new_transforms, outfile, indent=2)

    print(f"[INFO] writing {len(new_camera_pathes['camera_path'])} frames to {camera_pathes_path}")
    with open(camera_pathes_path, "w") as outfile:
        json.dump(new_camera_pathes, outfile, indent=2)

if __name__ == "__main__":
   main(sys.argv[1])