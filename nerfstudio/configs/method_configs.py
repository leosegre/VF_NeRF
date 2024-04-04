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
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig, SGDOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.registry import discover_methods

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "nerfacto": "Recommended real-time model tuned for real captures. This model will be continually updated.",
    "depth-nerfacto": "Nerfacto with depth supervision.",
    "volinga": "Real-time rendering model from Volinga. Directly exportable to NVOL format at https://volinga.ai/",
    "instant-ngp": "Implementation of Instant-NGP. Recommended real-time model for unbounded scenes.",
    "instant-ngp-bounded": "Implementation of Instant-NGP. Recommended for bounded real and synthetic scenes",
    "mipnerf": "High quality model for bounded scenes. (slow)",
    "semantic-nerfw": "Predicts semantic segmentations and filters out transient objects.",
    "vanilla-nerf": "Original NeRF model. (slow)",
    "tensorf": "tensorf",
    "dnerf": "Dynamic-NeRF model. (slow)",
    "phototourism": "Uses the Phototourism data.",
    "nerfplayer-nerfacto": "NeRFPlayer with nerfacto backbone.",
    "nerfplayer-ngp": "NeRFPlayer with InstantNGP backbone.",
    "neus": "Implementation of NeuS. (slow)",
}

method_configs["nerfacto"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3",
                optimizer=SGDOptimizerConfig(lr=6e-3, eps=1e-8),
                # scheduler=CosineDecaySchedulerConfig(max_steps=10000),
                # optimizer=AdamOptimizerConfig(lr=6e-3, eps=1e-8, weight_decay=1e-2),
                # scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-4, max_steps=10000),
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "nf_field": {
            "optimizer": RAdamOptimizerConfig(lr=5e-5, eps=0.1),
            "scheduler": None,
            # "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=100000),
            # "scheduler": CosineDecaySchedulerConfig(max_steps=30000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["register-nerfacto"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                # mode="SO3xR3",
                mode="SE3",
                optimizer=SGDOptimizerConfig(lr=5e-3, eps=1e-8),
                # optimizer=AdamOptimizerConfig(lr=5e-3, eps=1e-8),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=5e-4, max_steps=15000),
                # scheduler=CosineDecaySchedulerConfig(max_steps=10000),
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15, register=True),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "nf_field": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["objaverse-nerfacto"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3",
                optimizer=SGDOptimizerConfig(lr=6e-3, eps=1e-8),
                # scheduler=CosineDecaySchedulerConfig(max_steps=10000),
                # optimizer=AdamOptimizerConfig(lr=6e-3, eps=1e-8, weight_decay=1e-2),
                # scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-4, max_steps=10000),
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
                                  near_plane=2.0,
                                  far_plane=6.0,
                                  disable_scene_contraction=True,
                                  ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=20000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=20000),
        },
        "nf_field": {
            "optimizer": RAdamOptimizerConfig(lr=5e-5, eps=0.1),
            "scheduler": None,
            # "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=100000),
            # "scheduler": CosineDecaySchedulerConfig(max_steps=30000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["register-objaverse-nerfacto"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                # mode="SO3xR3",
                mode="SE3",
                optimizer=SGDOptimizerConfig(lr=1e-3, eps=1e-8),
                # optimizer=AdamOptimizerConfig(lr=5e-3, eps=1e-8),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=2500),
                # scheduler=CosineDecaySchedulerConfig(max_steps=10000),
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
                                  near_plane=0.0,
                                  far_plane=20.0,
                                  disable_scene_contraction=True,
                                  ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "nf_field": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)



external_methods, external_descriptions = discover_methods()
method_configs.update(external_methods)
descriptions.update(external_descriptions)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
