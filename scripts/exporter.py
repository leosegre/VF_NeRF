"""
Script for exporting NeRF into other formats.
"""

# pylint: disable=no-member

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import pathlib
import torch
import tyro
from rich.console import Console
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import (
    collect_camera_poses,
    generate_point_cloud,
    generate_point_cloud_nf,
    generate_cameras_from_nf,
    get_mesh_from_filename,
    render_trajectory,
)
from nerfstudio.exporter.marching_cubes import (
    generate_mesh_with_multires_marching_cubes,
)
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup

from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
import cv2 as cv

from nerfstudio.utils.io import load_from_json
from nerfstudio.utils import colormaps

from nerfstudio.exporter.exporter_utils import get_mask_from_view_likelihood


CONSOLE = Console(width=120)


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    estimate_normals: bool = False
    """Estimate normals for the point cloud."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Maximum of the bounding box, used if use_bounding_box is True."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            estimate_normals=self.estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")
@dataclass
class ExportPointCloudNF(Exporter):
    """Export NeRF as a point cloud from Normalizing flow."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Maximum of the bounding box, used if use_bounding_box is True."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    seed: int = 42
    threshold: float = 10.0

    def main(self) -> None:
        """Export point cloud."""

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # # Increase the batchsize to speed up the evaluation.
        # pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        pcd = generate_point_cloud_nf(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            std_ratio=self.std_ratio,
            threshold=self.threshold,
        )
        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

@dataclass
class ExportTransformsNF(Exporter):
    """Export transform matrices from Normalizing flow."""

    num_points: int = 10
    """Number of points to generate. May result in less if outlier removal is used."""
    min_depth: float = 0.7
    """depth: The depth of the camera."""
    max_depth: float = 1.2
    """depth: The depth of the camera."""
    downscale_factor: int = 1
    depth_output_name: str = "depth"
    rgb_output_name: str = "rgb"
    mask_output_name: str = "mask"
    view_likelihood_output_name: str = "view_log_likelihood"
    generate_masks: bool = True
    num_depth_points: int = 1
    seed: int = 42
    near_plane: float = None
    far_plane: float = None
    threshold: float = 0.05
    sample_ratio: int = 10


    def main(self) -> None:
        """Export transform matrices."""

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config, near_plane=self.near_plane, far_plane=self.far_plane)



        device = pipeline.device

        # # Increase the batchsize to speed up the evaluation.
        # pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch
        dataparser_transforms = load_from_json(pathlib.Path(os.path.join(os.path.dirname(self.load_config), "dataparser_transforms.json")))
        # print(dataparser_transforms)

        transforms, c2w = generate_cameras_from_nf(
            pipeline=pipeline,
            dataparser_transforms=dataparser_transforms,
            num_points=self.num_points,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            generate_masks=self.generate_masks,
            num_depth_points=self.num_depth_points,
            sample_ratio=self.sample_ratio,
        )
        torch.cuda.empty_cache()


        distortion_params = torch.zeros((self.num_points*self.num_depth_points, 6))
        cameras = Cameras(
            fx=transforms["fl_x"],
            fy=transforms["fl_y"],
            cx=transforms["cx"],
            cy=transforms["cy"],
            distortion_params=distortion_params,
            height=transforms["h"],
            width=transforms["w"],
            camera_to_worlds=c2w[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        color_images, depth_images, view_likelihood_images = render_trajectory(
            pipeline,
            cameras,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            view_likelihood_output_name=self.view_likelihood_output_name,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            disable_distortion=True,
        )

        color_images = 255 * torch.tensor(np.array(color_images), device=device).cpu().numpy()  # shape (N, 3, H, W)
        depth_images = 255 * torch.tensor(np.array(depth_images), device=device).cpu().numpy()  # shape (N, 1, H, W)
        view_likelihood_images = torch.tensor(np.array(view_likelihood_images), device=device)  # shape (N, 1, H, W)

        images_dir = "images"
        if self.downscale_factor > 1:
            images_dir = images_dir + "_" + str(self.downscale_factor)
        images_dir = os.path.join(self.output_dir, images_dir)
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)

        masks_dir = "masks"
        if self.downscale_factor > 1:
            masks_dir = masks_dir + "_" + str(self.downscale_factor)
        masks_dir = os.path.join(self.output_dir, masks_dir)
        if not os.path.exists(masks_dir):
            os.mkdir(masks_dir)

        best_viewshed = 0
        depth_list = []
        best_depth = 0


        for i in range(view_likelihood_images.shape[0]):
            output_viewshed = view_likelihood_images[i]
            min_value = torch.min(output_viewshed[~torch.isnan(output_viewshed)])
            # Replace NaN values with the minimum non-NaN value
            output_viewshed[torch.isnan(output_viewshed)] = min_value
            output_viewshed = torch.exp(output_viewshed)

            viewshed = output_viewshed.sum()

            print("viewshed", viewshed)
            if viewshed > best_viewshed:
                best_viewshed = viewshed
                best_depth = i
            if (i % self.num_depth_points) == (self.num_depth_points - 1):
                depth_list.append(best_depth)
                best_viewshed = 0

        print(depth_list)

        transforms["frames"] = [transforms["frames"][i] for i in depth_list]
        color_images = color_images[depth_list]
        depth_images = depth_images[depth_list]
        view_likelihood_images = view_likelihood_images[depth_list]


        mask_output, output_colormap = get_mask_from_view_likelihood(view_likelihood_images, threshold=self.threshold)

        for i in reversed(range(self.num_points)):

            # if (mask_output.sum() / (255 * mask_output.size)) <= 0.5:
            #     transforms["frames"].pop(i)

            cv.imwrite(f"{masks_dir}/{self.mask_output_name}_{i}.png", mask_output[i])
            cv.imwrite(f"{images_dir}/{self.view_likelihood_output_name}_{i}.png", output_colormap[i])

            color_images[i] = cv.cvtColor(color_images[i], cv.COLOR_BGR2RGB)
            cv.imwrite(f"{images_dir}/{self.rgb_output_name}_{i}.png", color_images[i])
            cv.imwrite(f"{images_dir}/{self.depth_output_name}_{i}.png", depth_images[i])



        for file_name, frames in [("transforms.json", transforms)]:
            if len(frames) == 0:
                CONSOLE.print(f"[bold yellow]No frames found for {file_name}. Skipping.")
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")



@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_point_cloud: bool = False
    """Whether to save the point cloud."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def validate_pipeline(self, pipeline: Pipeline) -> None:
        """Check that the pipeline is valid for this exporter."""
        if self.normal_method == "model_output":
            CONSOLE.print("Checking that the pipeline has a normal output.")
            origins = torch.zeros((1, 3), device=pipeline.device)
            directions = torch.ones_like(origins)
            pixel_area = torch.ones_like(origins[..., :1])
            camera_indices = torch.zeros_like(origins[..., :1])
            ray_bundle = RayBundle(
                origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
            )
            outputs = pipeline.model(ray_bundle)
            if self.normal_output_name not in outputs:
                CONSOLE.print(
                    f"[bold yellow]Warning: Normal output '{self.normal_output_name}' not found in pipeline outputs."
                )
                CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
                CONSOLE.print(
                    "[bold yellow]Warning: Please train a model with normals "
                    "(e.g., nerfacto with predicted normals turned on)."
                )
                CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
                CONSOLE.print("[bold yellow]Exiting early.")
                sys.exit(1)

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        self.validate_pipeline(pipeline)

        # Increase the batchsize to speed up the evaluation.
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()
        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")

        if self.save_point_cloud:
            CONSOLE.print("Saving Point Cloud...")
            o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

        CONSOLE.print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

        CONSOLE.print("Saving Mesh...")
        o3d.io.write_triangle_mesh(str(self.output_dir / "poisson_mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

        # This will texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the poisson reconstruction
            mesh = get_mesh_from_filename(
                str(self.output_dir / "poisson_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """Export a mesh using marching cubes."""

    isosurface_threshold: float = 0.0
    """The isosurface threshold for extraction. For SDF based methods the surface is the zero level set."""
    resolution: int = 1024
    """Marching cube resolution."""
    simplify_mesh: bool = False
    """Whether to simplify the mesh."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Minimum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Maximum of the bounding box."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Main function."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # TODO: Make this work with Density Field
        assert hasattr(pipeline.model.config, "sdf_field"), "Model must have an SDF field."

        CONSOLE.print("Extracting mesh with marching cubes... which may take a while")

        assert (
            self.resolution % 512 == 0
        ), f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the mimimum resolution is 512."""

        # Extract mesh using marching cubes for sdf at a multi-scale resolution.
        multi_res_mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=lambda x: pipeline.model.field.forward_geonetwork(x)[:, 0].contiguous(),
            resolution=self.resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            isosurface_threshold=self.isosurface_threshold,
            coarse_mask=None,
        )
        filename = self.output_dir / "sdf_marching_cubes_mesh.ply"
        multi_res_mesh.export(filename)

        # load the mesh from the marching cubes export
        mesh = get_mesh_from_filename(str(filename), target_num_faces=self.target_num_faces)
        CONSOLE.print("Texturing mesh with NeRF...")
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            self.output_dir,
            px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


@dataclass
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        train_frames, eval_frames = collect_camera_poses(pipeline)

        for file_name, frames in [("transforms_train.json", train_frames), ("transforms_eval.json", eval_frames)]:
            if len(frames) == 0:
                CONSOLE.print(f"[bold yellow]No frames found for {file_name}. Skipping.")
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportPointCloudNF, tyro.conf.subcommand(name="nf-pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
        Annotated[ExportTransformsNF, tyro.conf.subcommand(name="nf-cameras")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
