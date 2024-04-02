import sys
import shutil
import os
import json
import numpy as np

def main(base_directory, min_bound, max_bound, even_odd):
    if even_odd:
        exp_name = f"{min_bound}_{max_bound}_even_odd"
    else:
        exp_name = f"{min_bound}_{max_bound}"
    transforms_path = os.path.join(base_directory, 'transforms.json')
    transforms1_path = os.path.join(base_directory, f'transforms_{exp_name}_1.json')
    transforms2_path = os.path.join(base_directory, f'transforms_{exp_name}_2.json')
    backup_file_path = os.path.join(base_directory, 'transforms_backup')
    if not os.path.exists(backup_file_path):
        os.mkdir(backup_file_path)
    shutil.copy(transforms_path, backup_file_path)

    with open(os.path.join(transforms_path), 'r') as f:
        transforms = json.load(f)

    transforms1 = transforms.copy()
    transforms2 = transforms.copy()
    frames1 = []
    frames2 = []

    for i, frame in enumerate(transforms["frames"]):
        frames_len = len(transforms["frames"])
        if frame["colmap_im_id"] <= frames_len * (int(max_bound) / 100):
            if even_odd:
                if frame["colmap_im_id"] % 2:
                    frames1.append(frame)
            else:
                frames1.append(frame)
        if frame["colmap_im_id"] > frames_len * (int(min_bound) / 100):
            if even_odd:
                if frame["colmap_im_id"] % 2 == 0:
                    frames2.append(frame)
            else:
                frames2.append(frame)

    transforms1["frames"] = frames1
    transforms2["frames"] = frames2

    print(f"[INFO] writing {len(transforms1['frames'])} frames to {transforms1_path}")
    with open(transforms1_path, "w") as outfile:
        json.dump(transforms1, outfile, indent=2)

    print(f"[INFO] writing {len(transforms2['frames'])} frames to {transforms2_path}")
    with open(transforms2_path, "w") as outfile:
        json.dump(transforms2, outfile, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python process_data.py <base_directory> <min_bound> <max_bound> <even/odd>")
    else:
        base_directory = sys.argv[1]
        min_bound = sys.argv[2]
        max_bound = sys.argv[3]
        even_odd = sys.argv[4] == "True"
        if not os.path.isdir(base_directory):
            print(f"Error: {base_directory} is not a valid directory.")
        else:
            main(base_directory, min_bound, max_bound, even_odd)