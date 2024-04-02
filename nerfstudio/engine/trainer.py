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
Code to train model.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from rich.console import Console
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils
from tqdm import tqdm
from nerfstudio.viewer.server.viewer_state import ViewerState
import json
from nerfstudio.exporter.exporter_utils import (
    render_trajectory,
    get_mask_from_view_likelihood,
)
import numpy as np
import cv2 as cv

CONSOLE = Console(width=120)

TRAIN_INTERATION_OUTPUT = Tuple[  # pylint: disable=invalid-name
    torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
]
TORCH_DEVICE = Union[torch.device, str]  # pylint: disable=invalid-name


@dataclass
class TrainerConfig(ExperimentConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: Trainer)
    """target class to instantiate"""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 500
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 500
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""
    pretrain_iters: int = 0
    """number of pretrain iterations for registration."""
    nf_first_iter: int = 10000
    """number of pretrain iterations for registration."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    use_grad_scaler: bool = False
    """Use gradient scaler even if the automatic mixed precision is disabled."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    log_gradients: bool = False
    """Optionally log gradients during training"""
    start_step: Optional[int] = None
    """Optionally specify start step to load from."""
    t0: Optional[Path] = None
    """load JSON file of t0."""
    downscale_init: int = 4
    """Downscale the initial VF image H and W."""
    render_images_on_train: bool = False
    """Render images while training to visualize the progress."""
    rendered_images_dir: Optional[Path] = None
    """Render images while training to visualize the progress to this directory."""



class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        self.train_lock = Lock()
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device: TORCH_DEVICE = "cpu" if world_size == 0 else f"cuda:{local_rank}"
        self.mixed_precision: bool = self.config.mixed_precision
        self.use_grad_scaler: bool = self.mixed_precision or self.config.use_grad_scaler
        self.training_state: Literal["training", "paused", "completed"] = "training"

        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step: int = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler)
        self.nf_grad_scaler = GradScaler(enabled=self.use_grad_scaler)


        self.base_dir: Path = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir: Path = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")

        self.viewer_state = None
        self.pretrain_iters = config.pretrain_iters
        self.nf_first_iter = config.nf_first_iter
        self.downscale_init = config.downscale_init

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
            registration:
                Whether to register or train
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device, test_mode=test_mode, world_size=self.world_size, local_rank=self.local_rank
        )
        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Viewer at: {self.viewer_state.viewer_url}"]
        self._check_viewer_warnings()

        self._load_checkpoint()

        if self.pipeline.config.registration:
            self.optimizers = self.setup_optimizers_for_registration()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,  # type: ignore
                grad_scaler=self.grad_scaler,  # type: ignore
                nf_grad_scaler=self.nf_grad_scaler,  # type: ignore
                pipeline=self.pipeline,  # type: ignore
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(), self.config.is_tensorboard_enabled(), log_dir=writer_log_path
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()
        camera_optimizer_config = self.config.pipeline.datamanager.camera_optimizer
        if camera_optimizer_config is not None and camera_optimizer_config.mode != "off":
            assert camera_optimizer_config.param_group not in optimizer_config
            optimizer_config[camera_optimizer_config.param_group] = {
                "optimizer": camera_optimizer_config.optimizer,
                "scheduler": camera_optimizer_config.scheduler,
            }
        # if self.pipeline.model.config.predict_view_likelihood:
        #     for param_group in param_groups.values():
        #         for param in param_group:
        #             param.requires_grad = False
        # for param in self.pipeline.model.field.nf_model.parameters():
        #     param.requires_grad = True
        return Optimizers(optimizer_config, param_groups)

    def setup_optimizers_for_registration(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = {}
        param_groups = self.pipeline.get_param_groups()
        for param_group in param_groups.values():
            for param in param_group:
                param.requires_grad = False
        for param in param_groups["camera_opt"]:
            param.requires_grad = True
        camera_optimizer_config = self.config.pipeline.datamanager.camera_optimizer
        if camera_optimizer_config is not None and camera_optimizer_config.mode != "off":
            assert camera_optimizer_config.param_group not in optimizer_config
            optimizer_config[camera_optimizer_config.param_group] = {
                "optimizer": camera_optimizer_config.optimizer,
                "scheduler": camera_optimizer_config.scheduler,
            }
        camera_param_groups = {'camera_opt': param_groups["camera_opt"]}
        return Optimizers(optimizer_config, camera_param_groups)

    def equally_divided_cube(self, num_samples):
        # Calculate the number of points along each axis
        num_points_per_axis = int(torch.ceil(torch.pow(torch.tensor(num_samples, dtype=torch.float), 1 / 3)))

        # Generate equally spaced coordinates along each axis
        x = torch.linspace(0, 2, num_points_per_axis)
        y = torch.linspace(0, 2, num_points_per_axis)
        z = torch.linspace(0, 2, num_points_per_axis)

        # Create a meshgrid from the coordinates
        xx, yy, zz = torch.meshgrid(x, y, z)

        # Reshape the meshgrid into a list of coordinates
        xx = xx.flatten()[:num_samples]
        yy = yy.flatten()[:num_samples]
        zz = zz.flatten()[:num_samples]

        return xx, yy, zz

    def train(self) -> None:
        """Train the model."""

        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
            self.base_dir / "dataparser_transforms.json"
        )

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):


            num_pretrain = self._start_step + self.pretrain_iters

            num_iterations = self.config.max_num_iterations
            step = 0
            pretrain_flag = True
            for step in range(self._start_step, self._start_step + num_iterations):
                while self.training_state == "paused":
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        if self.pipeline.config.registration:
                            if self.config.t0 is not None and pretrain_flag:
                                with open(os.path.join(self.config.t0), 'r') as f:
                                    t0_json = json.load(f)
                                # t0 = torch.tensor(t0_json["t0"], device=self.device)
                                # with torch.no_grad():
                                #     self.pipeline.datamanager.train_camera_optimizer.pose_adjustment[[0], :] = t0
                                t0_matrix = torch.tensor(t0_json["t0_matrix"], device=self.device)
                                # t0_matrix = t0_matrix[torch.tensor([1, 0, 2, 3]), :]
                                # t0_matrix[2, :] *= -1
                                self.pipeline.datamanager.train_camera_optimizer.t0 = t0_matrix
                                print(t0_matrix)
                                pretrain_flag = False
                                best_psnr = 0
                                _, _, metrics_dict_train = self.pipeline.get_train_loss_dict(step)
                                t0_rotation_rmse = metrics_dict_train["rotation_rmse"]
                                t0_translation_rmse_100 = metrics_dict_train["translation_rmse_100"]
                            elif pretrain_flag:
                                if step == self._start_step:
                                    self.pipeline.datamanager.fixed_indices_train_dataloader.cameras.rescale_output_resolution(
                                        1.0 / self.downscale_init)
                                    metrics_dict = self.pipeline.get_average_train_image_metrics(step)
                                    best_viewshed_score = metrics_dict["viewshed_score"]
                                    _, _, metrics_dict_train = self.pipeline.get_train_loss_dict(step)
                                    best_rotation = metrics_dict_train["rotation_rmse"]
                                    best_6dof = self.pipeline.datamanager.train_camera_optimizer.pose_adjustment[[0], :]
                                    if self.pretrain_iters == 0:
                                        pretrain_flag = False
                                        step = self._start_step
                                        best_psnr = 0
                                elif step < num_pretrain:
                                    print("Pretrain step:", step)
                                    if self.pipeline.config.objaverse:
                                        # min_rand_rot = 0
                                        # max_rand_rot = 1 * torch.pi
                                        # min_rand_trans = -0.2
                                        # max_rand_trans = 0.2
                                        # random_6dof_rot = torch.randn(3).to(
                                        #     device=self.device) - torch.randn(3).to(
                                        #     device=self.device)
                                        # random_6dof_trans = torch.clamp(torch.randn(3), min=-0.2, max=0.2).to(
                                        #     device=self.device) - torch.clamp(torch.randn(3), min=-0.2, max=0.2).to(
                                        #     device=self.device)
                                        x, y, z = self.equally_divided_cube(num_pretrain)
                                        random_6dof_rot = torch.tensor([x[step]*torch.pi, y[step]*torch.pi, z[step]*torch.pi]).to(
                                            device=self.device)
                                        random_6dof_trans = torch.tensor([0, 0, 0]).to(
                                            device=self.device)
                                        random_6dof = torch.concat((random_6dof_trans, random_6dof_rot))
                                    else:
                                        min_rand_rot = 0
                                        max_rand_rot = 0.25*torch.pi
                                        min_rand_trans = -0.5
                                        max_rand_trans = 0.5
                                        # random_6dof_rot = torch.deg2rad(torch.tensor((-33.70861053466797, 85.56428527832031, 65.87945556640625-15)).to(
                                        #     device=self.device))
                                        # random_6dof_trans = torch.tensor((0.0986584841970366, -0.3439813595575635, -0.34400547966379735)).to(
                                        #     device=self.device)
                                        random_6dof_rot = (min_rand_rot - max_rand_rot) * torch.rand(3).to(
                                            device=self.device) + max_rand_rot
                                        random_6dof_trans = (min_rand_trans - max_rand_trans) * torch.rand(3).to(
                                            device=self.device) + max_rand_trans
                                        random_6dof = torch.concat((random_6dof_trans, random_6dof_rot))

                                    with torch.no_grad():
                                        self.pipeline.datamanager.train_camera_optimizer.pose_adjustment[[0], :] = random_6dof

                                    metrics_dict = self.pipeline.get_average_train_image_metrics(step)
                                    _, _, metrics_dict_train = self.pipeline.get_train_loss_dict(step)
                                    print("step:", step)
                                    print("best_VF:", best_viewshed_score)
                                    print("step_VF:", metrics_dict["viewshed_score"])
                                    print("step_rotation:", metrics_dict_train["rotation_rmse"])
                                    print("best_rotation:", best_rotation)

                                    if metrics_dict["viewshed_score"] > best_viewshed_score:
                                    # if metrics_dict_train["rotation_rmse"] < best_rotation:
                                        best_viewshed_score = metrics_dict["viewshed_score"]
                                        best_rotation = metrics_dict_train["rotation_rmse"]
                                        best_6dof = random_6dof
                                        print("step:", step, ", viewshed_score:", best_viewshed_score)
                                        print(best_6dof)
                                        self._update_register_cameras(step=step, pre_train=True)

                                    # ## If found match for more than 70% of the images - stop
                                    # if best_loss < (0.3 * 10):
                                    #     with torch.no_grad():
                                    #         self.pipeline.datamanager.train_camera_optimizer.pose_adjustment[[0], :] = best_6dof
                                    #     pretrain_flag = False
                                    #     step = self._start_step
                                    #     best_psnr = 0
                                elif step == num_pretrain:
                                    with torch.no_grad():
                                        self.pipeline.datamanager.train_camera_optimizer.pose_adjustment[[0], :] = best_6dof
                                    pretrain_flag = False
                                    step = self._start_step
                                    best_psnr = 0
                                    # self.pipeline.model.config.predict_view_likelihood = False

                            else:
                                loss, loss_dict, metrics_dict = self.train_iteration(step)
                                t_final = metrics_dict.pop("t_final")
                                if metrics_dict["psnr"] > best_psnr:
                                    best_metrics_dict = metrics_dict
                                    best_psnr = metrics_dict["psnr"]
                                    best_t_final = t_final
                                # print(best_psnr)
                            if self.config.render_images_on_train:
                                # if pretrain_flag or ((step-self.config.pretrain_iters)%100==0):
                                if step < 300 or (step % 100 == 0):
                                    self.pipeline.eval()
                                    # print(self.pipeline.datamanager.train_camera_optimizer.pose_adjustment[[0], :])
                                    # print(self.pipeline.datamanager.dataparser.get_dataparser_outputs().cameras)
                                    color_images, depth_images, view_likelihood_images = render_trajectory(
                                        self.pipeline,
                                        self.pipeline.datamanager.dataparser.get_dataparser_outputs().cameras,
                                        rgb_output_name="rgb",
                                        depth_output_name="depth",
                                        view_likelihood_output_name="view_log_likelihood",
                                        rendered_resolution_scaling_factor=1.0,
                                        disable_distortion=True,
                                        camera_opt_to_camera=self.pipeline.datamanager.train_camera_optimizer([0]).to(device="cpu"),
                                        camera_index=9
                                    )
                                    color_images = 255 * torch.tensor(np.array(color_images),
                                                                      device=self.device).cpu().numpy()  # shape (N, 3, H, W)
                                    depth_images = 255 * torch.tensor(np.array(depth_images),
                                                                      device=self.device).cpu().numpy()  # shape (N, 1, H, W)
                                    view_likelihood_images = torch.tensor(np.array(view_likelihood_images),
                                                                          device=self.device)  # shape (N, 1, H, W)
                                    _, output_colormap = get_mask_from_view_likelihood(view_likelihood_images)

                                    images_dir = self.config.rendered_images_dir


                                    for i in range(len(color_images)):
                                        if step == 0:
                                            for file in ["rgb", "vf", "depth"]:
                                                path = Path(f"{images_dir}/{i}/{file}")
                                                path.mkdir(parents=True, exist_ok=True)
                                        cv.imwrite(f"{images_dir}/{i}/vf/{step}.png",
                                                   output_colormap[i])

                                        color_images[i] = cv.cvtColor(color_images[i], cv.COLOR_BGR2RGB)
                                        cv.imwrite(f"{images_dir}/{i}/rgb/{step}.png", color_images[i])
                                        cv.imwrite(f"{images_dir}/{i}/depth/{step}.png", depth_images[i])
                                    self.pipeline.train()
                        else:
                            # time the forward pass
                            loss, loss_dict, metrics_dict = self.train_iteration(step)



                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.pipeline.datamanager.get_train_rays_per_batch() / train_t.duration,
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)
                if self.pipeline.config.registration and (step % 100) == 0:
                    self._update_register_cameras(step)
                    # self.pipeline.get_average_train_image_metrics(step)

                # a batch of train rays
                if (not self.pipeline.config.registration) or (self.pipeline.config.registration and step > num_pretrain):
                    if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                        writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                        writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                        writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                        # The actual memory allocated by Pytorch. This is likely less than the amount
                        # shown in nvidia-smi since some unused memory can be held by the caching
                        # allocator and some context needs to be created on GPU. See Memory management
                        # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                        # for more details about GPU memory management.
                        writer.put_scalar(
                            name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                        )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        if self.pipeline.config.registration:
            # Save the last iteration stats as txt file
            stats_json = best_metrics_dict
            stats_json_path = self.base_dir / "stats.json"
            # Iterate through the keys and values in the input dictionary
            for key, tensor in stats_json.items():
                # Convert the tensor to a scalar or list by extracting the value
                scalar_value = tensor.item() if isinstance(tensor, torch.Tensor) else tensor
                # Add the scalar to the new dictionary
                stats_json[key] = scalar_value
            stats_json["t_final"] = best_t_final.tolist()
            if self.config.t0 is not None:
                stats_json["t0_rotation_rmse"] = t0_rotation_rmse.tolist()
                stats_json["t0_translation_rmse_100"] = t0_translation_rmse_100.tolist()

            with open(stats_json_path, "w") as outfile:
                json.dump(stats_json, outfile, indent=2)

        CONSOLE.rule()
        CONSOLE.print("[bold green]:tada: :tada: :tada: Training Finished :tada: :tada: :tada:", justify="center")
        if not self.config.viewer.quit_on_train_completion:
            self.training_state = "completed"
            self._train_complete_viewer()
            CONSOLE.print("Use ctrl+c to quit", justify="center")
            while True:
                time.sleep(0.01)

    @check_main_thread
    def _check_viewer_warnings(self) -> None:
        """Helper to print out any warnings regarding the way the viewer/loggers are enabled"""
        if (
            self.config.is_viewer_enabled()
            and not self.config.is_tensorboard_enabled()
            and not self.config.is_wandb_enabled()
        ):
            string: str = (
                "[NOTE] Not running eval iterations since only viewer is enabled.\n"
                "Use [yellow]--vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard}[/yellow] to run with eval."
            )
            CONSOLE.print(f"{string}")

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state and self.pipeline.datamanager.train_dataset
        self.viewer_state.init_scene(
            dataset=self.pipeline.datamanager.train_dataset,
            # start_train=self.config.viewer.start_train,
            registration=self.pipeline.config.registration,
            train_state="training",
        )

    @check_viewer_enabled
    def _update_viewer_state(self, step: int) -> None:
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        try:
            self.viewer_state.update_scene(step, num_rays_per_batch)
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            assert self.viewer_state.vis is not None
            self.viewer_state.vis["renderingState/log_errors"].write(
                "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
            )

    @check_viewer_enabled
    def _update_register_cameras(self, step: int, pre_train=False) -> None:
        """Updates the Camera to includes registered cameras
        Returns the time taken to render scene.
        num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        try:
            self.viewer_state.update_scene(step, num_rays_per_batch)
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")

        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        try:
            self.viewer_state.update_register_cameras(self.pipeline.datamanager, step, pre_train=pre_train)
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            assert self.viewer_state.vis is not None
            self.viewer_state.vis["renderingState/log_errors"].write(
                "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
            )

    @check_viewer_enabled
    def _train_complete_viewer(self) -> None:
        """Let the viewer know that the training is complete"""
        assert self.viewer_state is not None
        try:
            self.viewer_state.training_complete()
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")

    @check_viewer_enabled
    def _update_viewer_rays_per_sec(self, train_t: TimeWriter, vis_t: TimeWriter, step: int) -> None:
        """Performs update on rays/sec calculation for training

        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=train_num_rays_per_batch / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
        )

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir: Path = self.config.load_dir
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest checkpoint from load_dir")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            if self.config.start_step is not None:
                self._start_step = self.config.start_step
            else:
                self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            self.nf_grad_scaler.load_state_dict(loaded_state["nf_scalers"])
            CONSOLE.print(f"done loading checkpoint from {load_path}")
        else:
            CONSOLE.print("No checkpoints to load, training from scratch")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "scalers": self.grad_scaler.state_dict(),
                "nf_scalers": self.nf_grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

        # ckpt_path_pt: Path = self.checkpoint_dir / f"step-{step:09d}.pt"
        # example_forward_input = torch.rand(1, 6)
        # # trace
        # print("tracing..")
        # pipeline_traced = torch.jit.trace(self.pipeline, example_forward_input)
        # pipeline_traced.eval()  # type: ignore
        # print("done tracing")
        #
        # # save the traced model
        # pipeline_traced.save(unet_traced_filename)  # type: ignore




    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        param_groups = self.pipeline.get_param_groups()
        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]
        # nf_total_params = sum(p.numel() for p in param_groups["nf_field"])
        # print(f"Number of parameters: {nf_total_params}")

        # if self.pipeline.model.config.predict_view_likelihood:
        #     _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
        #     loss = loss_dict["view_log_likelihood_loss"]
        #     loss.backward()
        #     self.optimizers.optimizer_step_all()
        # else:
        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            if self.pipeline.model.config.predict_view_likelihood:
                nf_loss = loss_dict.pop("view_log_likelihood_loss")
            loss = functools.reduce(torch.add, loss_dict.values())
        if self.pipeline.model.config.predict_view_likelihood and not self.pipeline.config.registration:
            if ~(torch.isnan(nf_loss) | torch.isinf(nf_loss) | torch.numel(nf_loss)==0) and step >= self.nf_first_iter:
                nf_field_params = param_groups.pop("nf_field")
                self.nf_grad_scaler.scale(nf_loss).backward()  # type: ignore
                self.nf_grad_scaler.unscale_(self.optimizers.optimizers["nf_field"])
                torch.nn.utils.clip_grad_norm_(nf_field_params, max_norm=5000.0)
                # self.optimizers.optimizer_step_all(exclude=param_groups.keys())
                self.optimizers.optimizer_scaler_step_all(self.nf_grad_scaler, exclude=param_groups.keys())
                self.nf_grad_scaler.update()
                self.optimizers.scheduler_step_all(step, exclude=param_groups.keys())
                # self.optimizers.optimizers["nf_field"].zero_grad()
            self.grad_scaler.scale(loss).backward()  # type: ignore
            self.optimizers.optimizer_scaler_step_all(self.grad_scaler, exclude=["nf_field"])
            self.grad_scaler.update()
            self.optimizers.scheduler_step_all(step, exclude=["nf_field"])
            loss_dict["view_log_likelihood_loss"] = nf_loss
        else:
            # if True:
            #     self.nf_grad_scaler.scale(nf_loss).backward()
            #     # self.nf_grad_scaler.unscale_(self.optimizers)
            #     self.optimizers.optimizer_scaler_step_all(self.nf_grad_scaler)
            #     self.nf_grad_scaler.update()
            #     self.optimizers.scheduler_step_all(step)
            #     loss_dict["view_log_likelihood_loss"] = nf_loss
            # else:
            self.grad_scaler.scale(loss).backward()  # type: ignore
            self.optimizers.optimizer_scaler_step_all(self.grad_scaler)
            self.grad_scaler.update()
            self.optimizers.scheduler_step_all(step)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad
                    total_grad += grad

            metrics_dict["Gradients/Total"] = total_grad


        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)
