import sys
import shutil
import os
import json
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from pathlib import Path
from PIL import Image

def clustering(poses: np.ndarray, num_clusters: int = 2, method: str = "KMeans") -> np.array:
    """
    Args:
        params poses: camera poses from camera frame to world frame, [N, 4, 4]
        params num_cluster: number of clusters to partition
        params method: use 'KMeans' or 'Spectral'
    Return:
        cluster labels for corresponding camera poses.
    """
    centers = poses[..., :3, -1]

    if method == 'KMeans':
        clustering = KMeans(
            n_clusters=num_clusters,
            random_state=0,
            n_init="auto"
        ).fit(centers)
    elif method == 'Spectral':
        clustering = SpectralClustering(
            n_clusters=num_clusters,
            assign_labels='discretize',
            random_state=0
        ).fit(centers)
    else:
        raise NotImplementedError

    return clustering.labels_


def main(base_directory):
    transforms_path = os.path.join(base_directory, 'transforms.json')
    transforms1_path = os.path.join(base_directory, f'transforms_1.json')
    transforms2_path = os.path.join(base_directory, f'transforms_2.json')
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
    camtoworlds = []

    mask_path = Path(base_directory) / "masks"
    mask_path.mkdir(exist_ok=True)

    for i, frame in enumerate(transforms["frames"]):
        camtoworlds.append(frame["transform_matrix"])
        image_path = Path(frame["file_path"])
        frame["mask_path"] = "./masks/" + image_path.name
        pil_image = Image.open(base_directory + frame["file_path"] + ".png")
        image = np.array(pil_image, dtype="uint8")
        mask = image[:, :, -1:].squeeze() > 128
        Image.fromarray(mask).save(base_directory + frame["mask_path"] + ".png")


    camtoworlds = np.stack(camtoworlds, axis=0)
    labels = clustering(camtoworlds, num_clusters=2, method="KMeans")

    for i, frame in enumerate(transforms["frames"]):
        transform_matrix = np.array(frame["transform_matrix"])
        transform_matrix[:3, :3] = transform_matrix[:3, :3] * 1e2
        frame["transform_matrix"] = transform_matrix.tolist()

        if labels[i] == 0:
            frames1.append(frame)
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
    if len(sys.argv) != 2:
        print("Usage: python process_data.py <base_directory>")
    else:
        base_directory = sys.argv[1]
        if not os.path.isdir(base_directory):
            print(f"Error: {base_directory} is not a valid directory.")
        else:
            main(base_directory)