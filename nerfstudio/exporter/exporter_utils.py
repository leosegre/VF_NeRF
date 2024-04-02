# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export utils such as structs, point cloud generation, and rendering code.
"""

# pylint: disable=no-member

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pymeshlab
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torchtyping import TensorType

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.rich_utils import ItersPerSecColumn

from nerfstudio.fields.base_field import shift_directions_for_tcnn
from nerfstudio.field_components.activations import trunc_exp
import cv2 as cv

from scipy.spatial.transform import Rotation
import torch.nn.functional as F
import torchgeometry as tgm
from nerfstudio.cameras import camera_utils
from nerfstudio.utils import poses as pose_utils
from nerfstudio.data.scene_box import SceneBox

CONSOLE = Console(width=120)


@dataclass
class Mesh:
    """Class for a mesh."""

    vertices: TensorType["num_verts", 3]
    """Vertices of the mesh."""
    faces: TensorType["num_faces", 3]
    """Faces of the mesh."""
    normals: TensorType["num_verts", 3]
    """Normals of the mesh."""
    colors: Optional[TensorType["num_verts", 3]] = None
    """Colors of the mesh."""


def get_mesh_from_pymeshlab_mesh(mesh: pymeshlab.Mesh) -> Mesh:
    """Get a Mesh from a pymeshlab mesh.
    See https://pymeshlab.readthedocs.io/en/0.1.5/classes/mesh.html for details.
    """
    return Mesh(
        vertices=torch.from_numpy(mesh.vertex_matrix()).float(),
        faces=torch.from_numpy(mesh.face_matrix()).long(),
        normals=torch.from_numpy(np.copy(mesh.vertex_normal_matrix())).float(),
        colors=torch.from_numpy(mesh.vertex_color_matrix()).float(),
    )


def get_mesh_from_filename(filename: str, target_num_faces: Optional[int] = None) -> Mesh:
    """Get a Mesh from a filename."""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    if target_num_faces is not None:
        CONSOLE.print("Running meshing decimation with quadric edge collapse")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_num_faces)
    mesh = ms.current_mesh()
    return get_mesh_from_pymeshlab_mesh(mesh)


def generate_point_cloud(
    pipeline: Pipeline,
    num_points: int = 1000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    normal_output_name: Optional[str] = None,
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    std_ratio: float = 10.0,
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    # pylint: disable=too-many-statements

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    points = []
    rgbs = []
    normals = []
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_train(0)
                outputs = pipeline.model(ray_bundle)
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            rgb = outputs[rgb_output_name]
            depth = outputs[depth_output_name]
            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                normal = outputs[normal_output_name]
                assert (
                    torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                ), "Normal values from method output must be in [0, 1]"
                normal = (normal * 2.0) - 1.0
            point = ray_bundle.origins + ray_bundle.directions * depth

            if use_bounding_box:
                comp_l = torch.tensor(bounding_box_min, device=point.device)
                comp_m = torch.tensor(bounding_box_max, device=point.device)
                assert torch.all(
                    comp_l < comp_m
                ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
                mask = torch.all(torch.concat([point > comp_l, point < comp_m], dim=-1), dim=-1)
                point = point[mask]
                rgb = rgb[mask]
                if normal_output_name is not None:
                    normal = normal[mask]

            points.append(point)
            rgbs.append(rgb)
            if normal_output_name is not None:
                normals.append(normal)
            progress.advance(task, point.shape[0])
    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())

    ind = None
    if remove_outliers:
        CONSOLE.print("Cleaning Point Cloud")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")

    # either estimate_normals or normal_output_name, not both
    if estimate_normals:
        if normal_output_name is not None:
            CONSOLE.rule("Error", style="red")
            CONSOLE.print("Cannot estimate normals and use normal_output_name at the same time", justify="center")
            sys.exit(1)
        CONSOLE.print("Estimating Point Cloud Normals")
        pcd.estimate_normals()
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
    elif normal_output_name is not None:
        normals = torch.cat(normals, dim=0)
        if ind is not None:
            # mask out normals for points that were removed with remove_outliers
            normals = normals[ind]
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    return pcd

def generate_point_cloud_nf(
    pipeline: Pipeline,
    num_points: int = 100000,
    remove_outliers: bool = True,
    use_bounding_box: bool = True,
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    std_ratio: float = 10.0,
    threshold: float = 10.0,
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    # pylint: disable=too-many-statements
    with torch.no_grad():
        d_cat_x, log_prob = pipeline.model.nf_field.nf_model.sample(num_points, sample_scale=1)
        # d_cat_x = sample(pipeline.model.nf_field.nf_model, 1, num_points, truncate_std=1, gpu=0).squeeze()
        print(d_cat_x.shape)
        points = d_cat_x[..., :3]
        directions = shift_directions_for_tcnn(d_cat_x[..., 3:])


        # points = 4 * (torch.rand((num_points, 3)) - 0.5)
        # directions = 2 * (torch.rand((num_points, 3)) - 0.5)

        print(points)


        # Get density
        if pipeline.model.field.spatial_distortion is not None:
            positions = pipeline.model.field.spatial_distortion(points)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(points, pipeline.model.field.aabb)

        positions_flat = positions.view(-1, 3)
        h = pipeline.model.field.mlp_base(positions_flat)
        density_before_activation, base_mlp_out = torch.split(h, [1, pipeline.model.field.geo_feat_dim], dim=-1)
        density = trunc_exp(density_before_activation.to(positions)).squeeze()

        # Get RGB
        d = pipeline.model.field.direction_encoding(directions)
        h = torch.cat(
            [
                d,
                base_mlp_out.view(-1, pipeline.model.field.geo_feat_dim)
            ],
            dim=-1,
        )
        rgbs = pipeline.model.field.mlp_head(h)

        # rgbs = torch.ones_like(rgbs)
        # rgbs[..., 0] = 0
        # rgbs[..., 2] = 0

        log_prob[torch.isnan(log_prob)] = 0
        sorted, indices = torch.sort(log_prob)
        # mask = indices[-10:]
        # mask = log_prob > 8
        # print(log_prob.min())
        # print(log_prob.max())

    mask = (density > threshold)
    points = points[mask]
    rgbs = rgbs[mask]


    if use_bounding_box:
        comp_l = torch.tensor(bounding_box_min, device=points.device)
        comp_m = torch.tensor(bounding_box_max, device=points.device)
        assert torch.all(
            comp_l < comp_m
        ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
        mask = torch.all(torch.concat([points > comp_l, points < comp_m], dim=-1), dim=-1)
        points = points[mask]
        rgbs = rgbs[mask]


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())

    if remove_outliers:
        CONSOLE.print("Cleaning Point Cloud")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")

    return pcd

def generate_cameras_from_nf(
    pipeline: Pipeline,
    dataparser_transforms: dict,
    num_points: int = 10,
    sample_ratio = 2,
    # sample_ratio = 100,
    min_depth: float = 0.8,
    max_depth: float = 1.0,
    generate_masks = True,
    num_depth_points = 5,

) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        depth: The depth of the camera.

    Returns:
        List of transform matrices.
    """
    points_to_sample = num_points * sample_ratio

    # pylint: disable=too-many-statements
    with torch.no_grad():
        d_cat_x, log_prob = pipeline.model.nf_field.nf_model.sample(points_to_sample, sample_scale=1)
        # d_cat_x = sample(pipeline.model.nf_field.nf_model, 1, num_points, truncate_std=1, gpu=0).squeeze()
        print(d_cat_x.shape)

        log_prob[torch.isnan(log_prob)] = -torch.inf
        sorted, indices = torch.sort(log_prob)
        mask = indices[-num_points:]
        d_cat_x = d_cat_x[mask]

        points = d_cat_x[..., :3]
        # points[0] = torch.zeros(3)
        directions = d_cat_x[..., 3:]
        directions_shape = directions.shape

        print("points", points)
        print("directions", directions)

    total_num_points = num_points * num_depth_points
    points = points.repeat_interleave(num_depth_points, dim=0)
    directions = directions.repeat_interleave(num_depth_points, dim=0)
    # depth = torch.FloatTensor(directions.size()).uniform_(min_depth, max_depth).to(directions.device)
    depth = torch.FloatTensor(np.linspace(min_depth, max_depth, num=num_depth_points)).to(directions.device)[..., None]
    depth = depth.repeat((num_points, 1))
    c2w = torch.zeros((total_num_points, 4, 4))
    my_c2w = torch.zeros((total_num_points, 4, 4))
    print(directions.shape)
    print(depth.shape)
    origins = points - directions * depth
    print(origins.shape)


    my_c2w[..., :3, 3] = origins  # (..., 3)

    def align_vectors(a, b):
        b = b / np.linalg.norm(b)  # normalize a
        a = a / np.linalg.norm(a)  # normalize b
        v = np.cross(a, b)
        # s = np.linalg.norm(v)
        c = np.dot(a, b)

        v1, v2, v3 = v
        h = 1 / (1 + c)

        Vmat = np.array([[0, -v3, v2],
                         [v3, 0, -v1],
                         [-v2, v1, 0]])

        R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
        return R

    directions = directions
    directions_numpy = directions.cpu().numpy()

    for i in range(total_num_points):
        my_c2w[i, :3, :3] = torch.from_numpy(align_vectors(np.array([0, 0, -1]), directions_numpy[i]))

    my_c2w[..., 3, 3] = 1

    up = torch.randn(directions_shape).repeat_interleave(num_depth_points, dim=0).to(device=directions.device)
    up = torch.cross(up, directions, dim=1)


    for i in range(total_num_points):
        c2w_temp = camera_utils.viewmatrix(-directions[i], up[i], origins[i])
        c2w_temp = pose_utils.to4x4(c2w_temp)
        c2w[i] = c2w_temp

    # c2w = my_c2w

    # import ipdb; ipdb.set_trace()
    c2w_list = c2w.tolist()

    ## Make full transform file with fixed parameters
    transforms = {}
    transforms["w"] = dataparser_transforms["width"]
    transforms["h"] = dataparser_transforms["height"]
    transforms["fl_x"] = dataparser_transforms["fx"]
    transforms["fl_y"] = dataparser_transforms["fy"]
    transforms["cx"] = dataparser_transforms["cx"]
    transforms["cy"] = dataparser_transforms["cy"]
    transforms["k1"] = 0
    transforms["k2"] = 0
    transforms["p1"] = 0
    transforms["p2"] = 0

    frames = []
    for i, camera in enumerate(c2w_list):
        frame = {}
        frame["file_path"] = f"images/rgb_{int(i/num_depth_points)}.png"
        if generate_masks:
            frame["mask_path"] = f"masks/mask_{int(i/num_depth_points)}.png"
        # transform_matrix = np.array(eval(camera["matrix"])).reshape((4, 4)).T
        frame["transform_matrix"] = camera
        frames.append(frame)

    transforms["frames"] = frames

    # train_dataparser_outputs = pipeline.datamanager.train_dataparser_outputs.as_dict()

    transforms["transform"] = dataparser_transforms["transform"]
    transforms["registration_matrix"] = dataparser_transforms["registration_matrix"]
    if "objaverse_transform_matrix_json" in dataparser_transforms:
        if dataparser_transforms["objaverse_transform_matrix_json"] is not None:
            transform_a = dict(dataparser_transforms["objaverse_transform_matrix_json"])["0"]
            transform_a = torch.tensor(np.array(transform_a))
            transform_b = dict(dataparser_transforms["objaverse_transform_matrix_json"])["1"]
            transform_b = torch.tensor(np.array(transform_b))
            # print(transform_a, transform_b)
            transforms["registration_matrix"] = pose_utils.to4x4(pose_utils.multiply(transform_a, pose_utils.inverse(transform_b))).tolist()
    transforms["registration_rot_euler"] = dataparser_transforms["registration_rot_euler"]
    transforms["registration_translation"] = dataparser_transforms["registration_translation"]
    transforms["scale"] = dataparser_transforms["scale"]


    return transforms, c2w


def render_trajectory(
    pipeline: Pipeline,
    cameras: Cameras,
    rgb_output_name: str,
    depth_output_name: str,
    view_likelihood_output_name: str = None,
    rendered_resolution_scaling_factor: float = 1.0,
    disable_distortion: bool = False,
    camera_opt_to_camera: Optional[TensorType["num_rays":..., 3, 4]] = None,
    camera_index = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Helper function to create a video of a trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        disable_distortion: Whether to disable distortion.

    Returns:
        List of rgb images, list of depth images.
    """
    images = []
    depths = []
    view_likelihood = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    # camera_opt_to_camera = pipeline.datamanager.train_camera_optimizer([0]).to(device="cpu")

    progress = Progress(
        TextColumn(":cloud: Computing rgb and depth images :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    if camera_index is not None:
        camera_indices = [camera_index]
    else:
        camera_indices = range(cameras.size)
    with progress:
        for camera_idx in progress.track(camera_indices, description=""):
            camera_ray_bundle = cameras.generate_rays(
                camera_indices=camera_idx, disable_distortion=disable_distortion, camera_opt_to_camera=camera_opt_to_camera
            ).to(pipeline.device)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if view_likelihood_output_name is not None:
                if view_likelihood_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {view_likelihood_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --view_likelihood_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)

            images.append(outputs[rgb_output_name].cpu().numpy())
            depths.append(outputs[depth_output_name].cpu().numpy())
            if view_likelihood_output_name is not None:
                view_likelihood.append(outputs[view_likelihood_output_name].cpu().numpy())
        if view_likelihood_output_name is not None:
            return images, depths, view_likelihood
        else:
            return images, depths


def collect_camera_poses_for_dataset(dataset: Optional[InputDataset]) -> List[Dict[str, Any]]:
    """Collects rescaled, translated and optimised camera poses for a dataset.

    Args:
        dataset: Dataset to collect camera poses for.

    Returns:
        List of dicts containing camera poses.
    """

    if dataset is None:
        return []

    cameras = dataset.cameras
    image_filenames = dataset.image_filenames

    frames: List[Dict[str, Any]] = []

    # new cameras are in cameras, whereas image paths are stored in a private member of the dataset
    for idx in range(len(cameras)):
        image_filename = image_filenames[idx]
        transform = cameras.camera_to_worlds[idx].tolist()
        frames.append(
            {
                "file_path": str(image_filename),
                "transform": transform,
            }
        )

    return frames


def collect_camera_poses(pipeline: VanillaPipeline) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collects camera poses for train and eval datasets.

    Args:
        pipeline: Pipeline to evaluate with.

    Returns:
        List of train camera poses, list of eval camera poses.
    """

    train_dataset = pipeline.datamanager.train_dataset
    assert isinstance(train_dataset, InputDataset)

    eval_dataset = pipeline.datamanager.eval_dataset
    assert isinstance(eval_dataset, InputDataset)

    train_frames = collect_camera_poses_for_dataset(train_dataset)
    eval_frames = collect_camera_poses_for_dataset(eval_dataset)

    return train_frames, eval_frames


def get_mask_from_view_likelihood(images, colormap_normalize=True, threshold=0.05):
    # Normalize
    colormap_max = 1
    colormap_min = 0
    # print(colormap_normalize)
    eps = 1e-20
    output = images
    # output = torch.nan_to_num(output)
    # Find the minimum non-NaN value
    min_value = torch.min(output[~torch.isnan(output)])
    # Replace NaN values with the minimum non-NaN value
    output[torch.isnan(output)] = min_value
    # print("before exp:", output.min(), output.max())
    # print(torch.amax(output, (1, 2, 3)))
    output = torch.exp(output)
    # print(torch.amax(output, (1, 2, 3)))

    # output = torch.clip(output, 0, 100)
    # print(torch.max(output))
    # print("after exp:", output.min(), output.max())
    if colormap_normalize:
        output = output - torch.min(output)
        output = output / (torch.max(output) + eps)
        # output = output / 2400
    output = output * (colormap_max - colormap_min) + colormap_min
    output = torch.nan_to_num(output)
    output_colormap_flat = torch.clip(output, 0, 1)
    output_colormap_flat = torch.sqrt(output_colormap_flat)
    output_colormap_flat = output_colormap_flat.cpu().numpy()
    output_colormap_flat = (output_colormap_flat * 255).astype(np.uint8)
    output_colormap = np.zeros((output_colormap_flat.shape[0], output_colormap_flat.shape[1], output_colormap_flat.shape[2], 3))
    for i in range(output_colormap_flat.shape[0]):
        output_colormap[i] = cv.applyColorMap(output_colormap_flat[i], cv.COLORMAP_TURBO)

    # threshold = 0.1
    mask_output = (255 * (output >= threshold)).cpu().numpy()

    return mask_output, output_colormap
