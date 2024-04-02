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
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast

import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler

from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from scipy.spatial.transform import Rotation
import numpy as np
import nerfstudio.utils.poses as pose_utils



def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


class Pipeline(nn.Module):
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization. Each model
    class should be 1:1 with a pipeline that can act as a standardized interface and hide
    differences in how each model takes in and outputs data.

    This class's function is to hide the data manager and model classes from the trainer,
    worrying about:
    1) Fetching data with the data manager
    2) Feeding the model the data and fetching the loss
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes, which each may have different forward() methods
    and so on.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'train': loads train/eval datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    # pylint: disable=abstract-method

    datamanager: DataManager
    _model: Model
    world_size: int

    @property
    def model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._model)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}
        self.model.load_state_dict(model_state, strict=strict)
        super().load_state_dict(pipeline_state, strict=False)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.world_size > 1:
            assert self.datamanager.eval_sampler is not None
            self.datamanager.eval_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @abstractmethod
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """

    @abstractmethod
    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average."""

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """


@dataclass
class VanillaPipelineConfig(cfg.InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VanillaPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = VanillaDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""
    registration: bool = False
    """Registration mode is on."""
    objaverse: bool = False
    """objaverse mode is on."""

class VanillaPipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: VanillaDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    def rotation_distance(self, R1, R2, eps=1e-7):
        """
        Args:
            R1: rotation matrix from camera 1 to world
            R2: rotation matrix from camera 2 to world
        Return:
            angle: the angular distance between camera 1 and camera 2.
        """
        # http://www.boris-belousov.net/2016/12/01/quat-dist/
        # R_diff = R1 @ R2.transpose(-2, -1)
        # R_diff = R1.transpose(-2, -1) @ R2
        R_diff = pose_utils.multiply(R1.transpose(-2, -1).to(torch.float64), R2.to(torch.float64))

        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]

        # numerical stability near -1/+1
        angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()
        angle = torch.rad2deg(angle)

        return angle

    @torch.no_grad()
    def evaluate_camera_alignment(self, pred_poses, poses_gt):
        """
        Args:
            pred_poses: [B, 3/4, 4]
            poses_gt: [B, 3/4, 4]
        """
        # measure errors in rotation and translation
        R_pred, t_pred = pred_poses.split([3, 1], dim=-1)
        R_gt, t_gt = poses_gt.split([3, 1], dim=-1)

        R_error = self.rotation_distance(R_pred[..., :3, :3], R_gt[..., :3, :3])
        t_error = (t_pred[..., :3, -1] - t_gt[..., :3, -1])[..., 0].norm(dim=-1)
        mean_rotation_error = R_error.mean()
        mean_position_error = t_error.mean()

        return mean_rotation_error, mean_position_error

    @profiler.time_function
    def get_train_loss_dict(self, step: int, full_images=False):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        if full_images:
            _, ray_bundle, batch = self.datamanager.next_train_image(step)
            model_outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)
        else:
            ray_bundle, batch = self.datamanager.next_train(step)
            model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch, full_images)

        if self.config.datamanager.camera_optimizer is not None:
            # camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            # if camera_opt_param_group in self.datamanager.get_param_groups():
            #     camera_opt_params = self.datamanager.get_param_groups()[camera_opt_param_group][0].data
            #     # Report the camera optimization metrics
            #     metrics_dict["camera_opt_translation"] = (
            #         camera_opt_params[:, :3].norm()
            #     )
            #     metrics_dict["camera_opt_rotation"] = (
            #         camera_opt_params[:, 3:].norm()
            #     )
            #
            #     # Apply learned transformation delta.
            #     if self.config.datamanager.camera_optimizer.mode == "off":
            #         pass
            #     elif self.config.datamanager.camera_optimizer.mode == "SO3xR3":
            #         camera_opt_transform_matrix = exp_map_SO3xR3(camera_opt_params)
            #     elif self.config.datamanager.camera_optimizer.mode == "SE3":
            #         camera_opt_transform_matrix = exp_map_SE3(camera_opt_params)
                # print("camera_opt_transform_matrix", camera_opt_transform_matrix)
                # print("unregistration_matrix", unregistration_matrix)
            if self.config.registration:
                # camera_opt_transform_matrix = pose_utils.multiply(self.datamanager.train_camera_optimizer([0]),
                #                                                   self.datamanager.train_camera_optimizer.t0)
                camera_opt_transform_matrix = self.datamanager.train_camera_optimizer([0])
                metrics_dict["t_final"] = camera_opt_transform_matrix
                registration_matrix = torch.tensor(self.datamanager.train_dataparser_outputs.metadata["registration_matrix"], device=self.device)
                if self.config.objaverse:
                    # unreg_pose = self.datamanager.train_dataparser_outputs.cameras.camera_to_worlds.to(device=self.device)
                    # unreg_pose = pose_utils.to4x4(unreg_pose)
                    # reg_pose_pred = pose_utils.multiply(camera_opt_transform_matrix, unreg_pose)
                    # reg_pose = pose_utils.multiply(registration_matrix, unreg_pose)
                    # rotation_rmse, translation_rmse = self.evaluate_camera_alignment(reg_pose_pred, reg_pose)
                    rotation_rmse, translation_rmse = self.evaluate_camera_alignment(camera_opt_transform_matrix, registration_matrix)
                    translation_rmse_100 = translation_rmse * 100
                else:
                    # unregistration_matrix = self.datamanager.train_dataparser_outputs.metadata["unregistration_matrix"]
                    # print(unregistration_matrix.shape)
                    # print(camera_opt_transform_matrix.shape)
                    # print(unregistration_matrix.to(self.device) @ torch.cat((camera_opt_transform_matrix.squeeze(), torch.tensor([[0, 0, 0, 1]], device=self.device)), dim=0))

                    def npmat2euler(mat, seq='xyz'):
                            eulers = []
                            r = Rotation.from_matrix(mat.cpu().detach().numpy())
                            eulers.append(r.as_euler(seq, degrees=True))
                            return torch.tensor(np.array(eulers), dtype=torch.float32)

                    camera_opt_rot_euler = npmat2euler(camera_opt_transform_matrix[:, :3, :3])
                    registration_rot_euler = torch.tensor(self.datamanager.train_dataparser_outputs.metadata["registration_rot_euler"])
                    camera_opt_translation = camera_opt_transform_matrix[:, :, 3].cpu()
                    registration_translation = torch.tensor(self.datamanager.train_dataparser_outputs.metadata["registration_translation"])

                    rotation_mse = torch.mean((camera_opt_rot_euler - registration_rot_euler).pow(2))
                    translation_mse = torch.mean((camera_opt_translation - registration_translation).pow(2))

                    rotation_rmse = torch.sqrt(rotation_mse)
                    translation_rmse = torch.sqrt(translation_mse)
                    translation_rmse_100 = translation_rmse * 100

                # metrics_dict["rotation_mse"] = (rotation_mse)
                metrics_dict["rotation_rmse"] = (rotation_rmse)
                # metrics_dict["translation_mse"] = (translation_mse)
                metrics_dict["translation_rmse_100"] = translation_rmse_100


        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict, full_images)

        return model_outputs, loss_dict, metrics_dict


    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    def get_train_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.train()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_train_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        return metrics_dict, images_dict


    def get_average_train_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.train()
        metrics_dict_list = []
        num_images = len(self.datamanager.fixed_indices_train_dataloader)
        # with Progress(
        #     TextColumn("[progress.description]{task.description}"),
        #     BarColumn(),
        #     TimeElapsedColumn(),
        #     MofNCompleteColumn(),
        #     transient=True,
        # ) as progress:
            # task = progress.add_task("[green]Evaluating all train images for registration...", total=num_images)

        for camera_ray_bundle, batch in self.datamanager.fixed_indices_train_dataloader:
            # time this the following line
            inner_start = time()
            height, width = camera_ray_bundle.shape
            num_rays = height * width
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch, step)
            assert "num_rays_per_sec" not in metrics_dict
            metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
            fps_str = "fps"
            assert fps_str not in metrics_dict
            metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
            metrics_dict_list.append(metrics_dict)
            # progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.median(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
            # if key == "loss":
            #     print(metrics_dict_list[0][key])
            #     print(metrics_dict_list[1][key])
        self.train()
        return metrics_dict
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch, step)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state, strict=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
