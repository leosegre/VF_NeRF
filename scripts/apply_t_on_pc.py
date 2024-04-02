import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import json
import open3d as o3d
from nerfstudio.utils import poses as pose_utils
import torch
import copy
import random
import os
import cv2
from scipy import spatial
import random
import math
from scipy.spatial.transform import Rotation as R


def main(path_A, path_B, output_path, transform_path, scene_name):
    # Read point cloud data from ply files
    A = o3d.io.read_point_cloud(path_A)
    B = o3d.io.read_point_cloud(path_B)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(os.path.join(transform_path), 'r') as f:
        stats_json = json.load(f)

    scene_stats = stats_json[scene_name]["best"]
    t_final = np.array(scene_stats["t_final"]).squeeze()
    t_final = np.vstack((t_final, np.array([[0, 0, 0, 1]])))

    transformed_A = copy.deepcopy(A).transform(t_final)
    # transformed_B = copy.deepcopy(B).transform(registration_matrix)
    output_path = Path(output_path)
    output_path_our = str(output_path / "our_method.ply")
    output_path_A = str(output_path / "A.ply")
    output_path_B = str(output_path / "B.ply")
    # output_path_B = str(output_path.parent / "Transformed_B.ply")
    o3d.io.write_point_cloud(output_path_our, transformed_A)
    o3d.io.write_point_cloud(output_path_A, A)
    o3d.io.write_point_cloud(output_path_B, B)
    # o3d.io.write_point_cloud(output_path_B, transformed_B)
    # o3d.visualization.draw_geometries([transformed_A, B])


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ICP with ply point clouds')
    parser.add_argument('path_A', type=str, help='Path to ply file A')
    parser.add_argument('path_B', type=str, help='Path to ply file B')
    parser.add_argument('transform_path', type=str, help='transform path for JSON file')
    parser.add_argument('output_path', type=str, help='Output path for JSON file')
    parser.add_argument('scene_name', type=str, help='scene name in JSON file')

    args = parser.parse_args()

    # Call main function with provided arguments
    main(args.path_A, args.path_B, args.output_path, args.transform_path, args.scene_name)
