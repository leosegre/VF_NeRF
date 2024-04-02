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
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

import nerfstudio.utils.poses as pose_utils

from scipy.spatial.transform import Rotation

import imageio


CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class NerfstudioDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nerfstudio)
    """target class to instantiate"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    registration: bool = False
    """Whether to apply registration transform."""
    optimize_camera_registration: bool = False
    """Whether to apply registration transform."""
    scale_opt: bool = False
    """Whether to optimize scale factor."""
    load_registration: bool = False
    """Whether to load registration data from json."""
    max_angle_factor: float = 12
    """Max Angle to rotate for registration test (for example - 4 -> pi/4)."""
    max_translation: float = 0.2
    """Max translation for registration test."""
    blender: bool = False
    """dataset from blender."""
    registration_data: Path = None
    """Directory or explicit json file path specifying location of registration data."""
    inerf: bool = False
    """load a aframe and transform it."""
    objaverse: bool = False
    """dataset from objaverse."""
    alpha_color: str = None
    """alpha color of background"""
    objaverse_transform_matrix: str = None
    """matrix number of registration data."""





@dataclass
class Nerfstudio(DataParser):
    """Nerfstudio DatasetParser"""

    config: NerfstudioDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        # import ipdb; ipdb.set_trace()

        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        if self.config.registration_data is not None:
            assert self.config.registration_data.exists(), f"Data directory {self.config.registration_data} does not exist."
            registration_data = load_from_json(self.config.registration_data / f"dataparser_transforms.json")


        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        elif self.config.blender:
            meta = load_from_json(self.config.data / f"transforms_{split}.json")
            data_dir = self.config.data
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []
        num_skipped_image_filenames = 0

        if self.config.blender or self.config.objaverse:
            image_0_filename = meta["frames"][0]["file_path"]
            if self.config.blender:
                image_0_filename = self.config.data / Path(image_0_filename.replace("./", "") + ".png")
            else:
                image_0_filename = data_dir / Path(image_0_filename.replace("./", "") + ".png")
            img_0 = imageio.v2.imread(image_0_filename)
            meta["h"], meta["w"] = img_0.shape[:2]
            camera_angle_x = float(meta["camera_angle_x"])
            meta["fl_x"] = 0.5 * meta["w"] / np.tan(0.5 * camera_angle_x)
            meta["fl_y"] = meta["fl_x"]
            meta["cx"] = meta["w"] / 2.0
            meta["cy"] = meta["h"] / 2.0
            height_fixed, width_fixed, fx_fixed, fy_fixed, cx_fixed, cy_fixed = True, True, True, True, True, True

            # camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform
        else:
            fx_fixed = "fl_x" in meta
            fy_fixed = "fl_y" in meta
            cx_fixed = "cx" in meta
            cy_fixed = "cy" in meta
            height_fixed = "h" in meta
            width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        if self.config.objaverse:
            camera_utils.get_distortion_params(
                k1=float(meta["fl_x"]),
                k2=float(0),
                k3=float(meta["cx"]),
                k4=float(0),
                p1=float(meta["fl_y"]),
                p2=float(meta["cy"]),
            )

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            if self.config.blender:
                fname = self.config.data / Path(frame["file_path"].replace("./", "") + ".png")
            elif self.config.objaverse:
                fname = data_dir / Path(frame["file_path"].replace("./", "") + ".png")
                self.downscale_factor = self.config.downscale_factor
            else:
                filepath = PurePath(frame["file_path"])
                fname = self._get_fname(filepath, data_dir)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = PurePath(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                if self.config.objaverse:
                    mask_fname = data_dir / Path(frame["mask_path"].replace("./", "") + ".png")
                mask_filenames.append(mask_fname)

            if "depth_file_path" in frame:
                depth_filepath = PurePath(frame["depth_file_path"])
                depth_fname = self._get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
                depth_filenames.append(depth_fname)

        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(self._get_fname(PurePath(x), data_dir) for x in meta[f"{split}_filenames"])
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # filter image_filenames and poses based on train/eval split percentage
            num_images = len(image_filenames)
            num_train_images = math.ceil(num_images * self.config.train_split_fraction)
            num_eval_images = num_images - num_train_images
            i_all = np.arange(num_images)
            i_train = np.linspace(
                0, num_images - 1, num_train_images, dtype=int
            )  # equally spaced training images starting and ending at 0 and num_images-1
            i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
            assert len(i_eval) == num_eval_images
            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        if self.config.objaverse_transform_matrix is not None:
            objaverse_transform_matrix_json = data_dir / f"world_frame_transforms.json"
            assert objaverse_transform_matrix_json.exists(), f"objaverse_transform_matrix  {objaverse_transform_matrix_json} does not exist."
            objaverse_transform_matrix_json = load_from_json(objaverse_transform_matrix_json)
            objaverse_transform_matrix = torch.tensor(objaverse_transform_matrix_json[self.config.objaverse_transform_matrix])
            poses = objaverse_transform_matrix @ poses
        else:
            objaverse_transform_matrix_json = None
        if self.config.registration_data is not None:
            transform_matrix = torch.tensor(registration_data["transform"])
            poses = transform_matrix @ poses
        elif self.config.load_registration:
            transform_matrix = torch.tensor(meta["transform"])
            # transform_matrix = torch.eye(4)
        else:
            poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
                poses,
                method=orientation_method,
                center_method=self.config.center_method,
            )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        elif self.config.registration_data is not None:
            scale_factor = torch.tensor(registration_data["scale"])
        scale_factor *= self.config.scale_factor
        # if self.config.load_registration:
        #     scale_factor *= torch.tensor(meta["scale"])
        if self.config.load_registration and not self.config.inerf:
            scale_factor = 1.0

        poses[:, :3, 3] *= scale_factor


        if self.config.registration:
            if self.config.load_registration:
                registration_matrix = torch.tensor(meta["registration_matrix"])
                if meta["registration_rot_euler"] is not None:
                    registration_rot_euler = torch.tensor(meta["registration_rot_euler"])
                    registration_translation = torch.tensor(meta["registration_translation"])
                else:
                    registration_rot_euler = meta["registration_rot_euler"]
                    registration_translation = meta["registration_translation"]
            else:
                max_angle_factor = self.config.max_angle_factor
                max_translation = self.config.max_translation

                anglex = np.random.uniform() * np.pi * max_angle_factor
                angley = np.random.uniform() * np.pi * max_angle_factor
                anglez = np.random.uniform() * np.pi * max_angle_factor


                registration_rot_euler = torch.rad2deg(torch.tensor([anglex, angley, anglez]))
                r = Rotation.from_euler('xyz', registration_rot_euler, degrees=True)
                rotation_ab = r.as_matrix()

                translation_ab = np.array([np.random.uniform(-max_translation, max_translation), np.random.uniform(-max_translation, max_translation),
                                           np.random.uniform(-max_translation, max_translation)])

                registration_matrix = np.zeros((4, 4), dtype=np.float32)
                registration_matrix[:3, :3] = rotation_ab
                registration_matrix[3, 3] = 1
                registration_matrix[:3, -1] = translation_ab.T
                registration_translation = torch.from_numpy(translation_ab)
                CONSOLE.log(f"[yellow] registration_matrix is {registration_matrix}")
                registration_matrix = torch.from_numpy(registration_matrix)

                unregistration_matrix = pose_utils.inverse(registration_matrix)
                poses = pose_utils.multiply(unregistration_matrix, poses)

                CONSOLE.log(f"[yellow] rotation is {r}")
                CONSOLE.log(f"[yellow] translation is {translation_ab}")

            if self.config.inerf:
                unregistration_matrix = pose_utils.inverse(registration_matrix)
                poses = pose_utils.multiply(unregistration_matrix, poses)

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        poses = poses[indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)
        # print(cameras)

        # if "applied_transform" in meta:
        #     if not self.config.load_registration:
        #         applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
        #         transform_matrix = transform_matrix @ torch.cat(
        #             [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
        #         )
        # if "applied_scale" in meta:
        #     applied_scale = float(meta["applied_scale"])
        #     scale_factor *= applied_scale

        if self.config.alpha_color == "white":
            alpha_color = 1
        else:
            alpha_color = None
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                "registration_matrix": registration_matrix if self.config.registration else None,
                "registration_rot_euler": registration_rot_euler if self.config.registration else None,
                "registration_translation": registration_translation if self.config.registration else None,
                "objaverse_transform_matrix_json": objaverse_transform_matrix_json if self.config.objaverse else None,
                "height": height,
                "width": width,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            },
            alpha_color=alpha_color,
        )
        return dataparser_outputs

    def _get_fname(self, filepath: PurePath, data_dir: PurePath, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath
