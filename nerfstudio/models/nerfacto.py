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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.fields.nf_field import NFField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    view_likelihood_loss,
    weighted_mse_loss,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
    UncertaintyRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

from nerfstudio.fields.base_field import shift_directions_for_tcnn


from torchvision.utils import save_image
from scipy.ndimage import gaussian_filter
import cv2 as cv
import torchvision

from nerfstudio.exporter.exporter_utils import get_mask_from_view_likelihood



@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    rgb_loss_mult: float = 1.0
    """rgb loss multiplier."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    pred_directions_loss_mult: float = 0.001
    """Predicted directions loss multiplier."""
    view_likelihood_loss_mult: float = 1
    """View_likelihood loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    predict_directions: bool = False
    """Whether to predict directions or not."""
    predict_view_likelihood: bool = False
    """Whether to predict uncertainty or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    register: bool = False
    """Whether to register scene or not."""
    mse_init: bool = False
    """Whether to register using mse loss, otherwise use viewshed."""
    weighted_loss: bool = False
    """Whether to register using weightef loss."""
    nf_loss_on_mask_only: bool = False
    """Apply nf_loss only where the image is masked."""
    noise_oriented_points: bool = False
    """Apply noise to the oriented point (nf field)."""
    noise_level_oriented_points: float = 0.0
    """Noise level - percentage of scaled scene to the oriented point (nf field)."""


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.register = self.config.register
        self.mse_init = self.config.mse_init

        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_pred_directions=self.config.predict_directions,
            use_view_likelihood=False,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )
        if self.config.predict_view_likelihood:
            self.nf_field = NFField()

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()
        self.renderer_uncertainty = UncertaintyRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.weighted_rgb_loss = weighted_mse_loss

        self.weighted_loss = self.config.weighted_loss

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        if self.config.predict_view_likelihood:
            param_groups["nf_field"] = list(self.nf_field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            # print(field_outputs[FieldHeadNames.NORMALS])
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        if self.config.predict_directions:
            # code = torch.sin(ray_samples.frustums.directions.detach().pow(2).sum(dim=-1) +  ray_samples.frustums.get_positions().detach().pow(2).sum(dim=-1) + field_outputs[FieldHeadNames.RGB].detach().pow(2).sum(dim=-1)).unsqueeze(-1)
            code = torch.sin(shift_directions_for_tcnn(ray_samples.frustums.directions).detach().pow(2).sum(dim=-1) +  ray_samples.frustums.get_positions().detach().pow(2).sum(dim=-1)).unsqueeze(-1)
            directions = self.renderer_uncertainty(betas=code, weights=weights)
            pred_directions = self.renderer_uncertainty(betas=field_outputs[FieldHeadNames.DIRECTIONS], weights=weights)
            # outputs["directions"] = self.normals_shader(directions)
            # outputs["pred_directions"] = self.normals_shader(pred_directions)
            outputs["directions"] = directions
            outputs["pred_directions"] = pred_directions

        if self.config.predict_view_likelihood:
            # print("check")
            # cpu_or_cuda_str: str = str(self.device).split(":")[0]
            # with torch.autocast(device_type=cpu_or_cuda_str, enabled=False):

            # cloned_weights = weights.detach()
            # argmax_weights = cloned_weights.max(axis=1)[1].squeeze()
            # nf_field_outputs = self.nf_field.get_outputs(ray_samples[np.arange(len(ray_samples)), argmax_weights])
            # outputs["view_log_likelihood"] = nf_field_outputs[FieldHeadNames.VIEW_LOG_LIKELIHOOD]

            def ray_sphere_intersection_batch(origins, directions, centers, radii):
                # Calculate the coefficients of the quadratic equation
                a = torch.sum(directions ** 2, dim=-1)
                oc = origins - centers
                b = 2 * torch.sum(oc * directions, dim=-1)
                c = torch.sum(oc ** 2, dim=-1) - radii ** 2

                # Calculate the discriminant
                discriminant = b ** 2 - 4 * a * c

                # Find the values of t where the ray intersects the sphere
                t1 = (-b - torch.sqrt(discriminant)) / (2 * a)
                t2 = (-b + torch.sqrt(discriminant)) / (2 * a)

                # Choose the intersection points corresponding to the smaller t value
                t_min = torch.min(t1, t2)
                t_max = torch.max(t1, t2)
                intersection_points = origins + t_min.unsqueeze(-1) * directions
                return intersection_points

            def ray_plane_intersection_batch(origins, directions, plane_axis, plane_coordinate):
                # Ensure the direction is normalized
                directions = directions / torch.norm(directions, dim=-1, keepdim=True)

                # Calculate the parameter t for the intersection with the plane
                t = (plane_coordinate - origins[:, plane_axis]) / directions[:, plane_axis]

                # Calculate the intersection points
                intersection_points = origins + t.unsqueeze(-1) * directions

                return intersection_points

            points = ray_bundle.origins + ray_bundle.directions * depth
            # intersection_points_circle = ray_sphere_intersection_batch(ray_bundle.origins, ray_bundle.directions,
            #                                                     torch.tensor([0.0, 0.0, 0.0], device=ray_bundle.origins.device)
            #                                                     , 0.5)
            # intersection_points_plane = ray_plane_intersection_batch(ray_bundle.origins, ray_bundle.directions, 2,
            #                                                     -0.5)

            # max_abs_value_of_points = torch.max(points.abs(), dim=1).values
            # eps = 0.2
            # assuming aabb is a symethric box
            # point_on_boundary = max_abs_value_of_points >= (torch.max(self.scene_box.aabb) - eps)

            if self.config.noise_oriented_points:
                # Assuming scene scale is 2
                points = points + 2 * (2*torch.rand_like(points) - 1) * self.config.noise_level_oriented_points

            nf_field_outputs = self.nf_field.get_outputs(points, ray_bundle.directions)
            outputs["view_log_likelihood"] = nf_field_outputs[FieldHeadNames.VIEW_LOG_LIKELIHOOD]
            # print(outputs["view_log_likelihood"])
            # outputs["view_log_likelihood"][point_on_boundary] = -100
            # print(outputs["view_log_likelihood"])





            # nf_field_outputs = self.nf_field.get_outputs(ray_samples)
            # outputs["view_log_likelihood"] = self.renderer_uncertainty(betas=nf_field_outputs[FieldHeadNames.VIEW_LOG_LIKELIHOOD], weights=torch.clip((cloned_weights-0.1), -1/48, 1))
            # if outputs["view_log_likelihood"].isnan().any():
            #     import ipdb; ipdb.set_trace()

            # nan_mask = ~outputs["view_log_likelihood"].isnan()
            # outputs["view_log_likelihood"] = outputs["view_log_likelihood"][nan_mask]

            # outputs["view_log_likelihood"][outputs["view_log_likelihood"].isnan()] = 0



            # if outputs["view_log_likelihood"].max() > 1:
            # print("max weight", cloned_weights.max())
            # print("max log_likelihood", field_outputs[FieldHeadNames.VIEW_LOG_LIKELIHOOD].max())

            # outputs["view_log_likelihood"] = self.renderer_uncertainty(betas=field_outputs[FieldHeadNames.VIEW_LOG_LIKELIHOOD], weights=cloned_weights)
            # outputs["view_log_likelihood"] = view_log_likelihood
            # with torch.no_grad():
            #     # cloned_weights = weights.detach()
            #     outputs["view_likelihood"] = self.renderer_uncertainty(betas=field_outputs[FieldHeadNames.VIEW_LIKELIHOOD], weights=cloned_weights)
            #     # cloned_weights = weights.detach()
            #     outputs["max_density"] = cloned_weights.squeeze().max(dim=-1)[0]
            #     # print(outputs["max_density"].shape)
            #     # outputs["view_likelihood"] = view_log_likelihood
            # with torch.no_grad():
            #     outputs["view_likelihood_exp"] = torch.exp(self.renderer_uncertainty(betas=field_outputs[FieldHeadNames.VIEW_LOG_LIKELIHOOD].detach(), weights=weights.detach()))


        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        if self.training and self.config.predict_directions:
            outputs["rendered_pred_directions_loss"] = pred_normal_loss(
                weights.detach(),
                code,
                field_outputs[FieldHeadNames.DIRECTIONS],
            )

        # if self.training and self.config.predict_view_likelihood:
        #     outputs["rendered_view_log_likelihood_loss"] = view_likelihood_loss(
        #         weights.detach(),
        #         field_outputs[FieldHeadNames.VIEW_LOG_LIKELIHOOD],
        #     )


        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch, full_images=False):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training and not full_images:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None, full_images=False):
        loss_dict = {}
        image = batch["image"].to(self.device)

        if self.register:
            # yuv_transform = torch.tensor([[0.2126, 0.7152, 0.0722], [-0.09991, -0.33609, 0.436], [0.615, -0.55861, -0.05639]], device=self.device)
            yuv_transform = torch.tensor([[0.0722, 0.7152, 0.2126], [0.436, -0.33609, -0.09991], [-0.05639, -0.55861, 0.615]], device=self.device)

            image_yuv = torch.matmul(yuv_transform, image.unsqueeze(-1)).squeeze()
            rgb_yuv = torch.matmul(yuv_transform, outputs["rgb"].unsqueeze(-1)).squeeze()

            if self.weighted_loss:
                viewshed = outputs["view_log_likelihood"]
                # Find the minimum non-NaN value
                if not torch.isnan(viewshed).all():
                    min_value = torch.min(viewshed[~torch.isnan(viewshed)])
                    # Replace NaN values with the minimum non-NaN value
                    viewshed[torch.isnan(viewshed)] = min_value
                    # print("before exp:", output.min(), output.max())
                    viewshed = torch.exp(viewshed)
                    viewshed = torch.nan_to_num(viewshed)

                    # viewshed = viewshed >= 10
                    # if not viewshed.any():
                    #     viewshed = torch.ones_like(viewshed)

                    # viewshed = viewshed.clamp(min=0, max=1)
                    # viewshed_score = viewshed[image_mask].sum()
                else:
                    viewshed = torch.ones_like(viewshed)

                viewshed = 100 * (viewshed / viewshed.sum())

                # print(image_yuv.shape)
                # print(rgb_yuv.shape)
                # loss_dict["rgb_loss"] = self.config.rgb_loss_mult * self.rgb_loss(image_yuv[..., 1:], rgb_yuv[..., 1:])
                loss_dict["rgb_loss"] = self.config.rgb_loss_mult * self.weighted_rgb_loss(viewshed.detach(), image_yuv[..., 1:], rgb_yuv[..., 1:])
            else:
                loss_dict["rgb_loss"] = self.config.rgb_loss_mult * self.rgb_loss(image_yuv[..., 1:], rgb_yuv[..., 1:])
        else:
            loss_dict["rgb_loss"] = self.config.rgb_loss_mult * self.rgb_loss(image, outputs["rgb"])

        if self.training and not full_images:
            # print(outputs["depth"])
            # loss_dict["depth_loss"] = self.rgb_loss(outputs["depth"], torch.tensor([0.5], device=self.device))
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            if self.config.predict_directions:
                loss_dict["pred_directions_loss"] = self.config.pred_directions_loss_mult * torch.mean(
                    outputs["rendered_pred_directions_loss"]
                )
            if self.config.predict_view_likelihood:
                # cpu_or_cuda_str: str = str(self.device).split(":")[0]
                # with torch.autocast(device_type=cpu_or_cuda_str, enabled=False):
                if self.config.nf_loss_on_mask_only and "mask" in batch:
                    mask = batch["mask"].to(self.device)
                    loss_dict["view_log_likelihood_loss"] = self.config.view_likelihood_loss_mult * -torch.mean(
                        outputs["view_log_likelihood"][mask]
                    )
                else:
                    loss_dict["view_log_likelihood_loss"] = self.config.view_likelihood_loss_mult * -torch.mean(
                        outputs["view_log_likelihood"]
                    )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], step
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        if "mask" in batch.keys():
            image_mask = batch["mask"].to(self.device)
        else:
            image_mask = torch.ones_like(image, device=self.device, dtype=torch.bool)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        if image.shape != rgb.shape:
            torch_image = torch.moveaxis(image, -1, 0)
            torch_image_mask = torch.moveaxis(image_mask, -1, 0)
            torch_rgb = torch.moveaxis(rgb, -1, 0)
            image = torchvision.transforms.functional.resize(torch_image, torch_rgb.shape[1])
            image = torchvision.transforms.functional.resize(torch_image, torch_rgb.shape[1])
            image = torch.moveaxis(image, 0, -1)
            image_mask = torchvision.transforms.functional.resize(torch_image_mask, torch_rgb.shape[1])
            image_mask = torch.moveaxis(image_mask, 0, -1)

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        torch_image = torch.moveaxis(image, -1, 0)[None, ...]
        torch_rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        torch_rgb = torch.clamp(torch_rgb, 0, 1)
        psnr = self.psnr(torch_image, torch_rgb)
        ssim = self.ssim(torch_image, torch_rgb)
        lpips = self.lpips(torch_image, torch_rgb)

        if False:
            image_numpy = image.cpu().numpy()
            rgb_numpy = rgb.cpu().numpy()
            image_mask_numpy = image_mask.cpu().numpy().astype(np.uint8)
            rgb_mask_numpy = rgb_mask_numpy.astype(np.uint8)

            # gray_image = cv.cvtColor(image_numpy, cv.COLOR_BGR2GRAY)
            # gray_image = cv.normalize(gray_image, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            # gray_rgb = cv.cvtColor(rgb_numpy, cv.COLOR_BGR2GRAY)
            # gray_rgb = cv.normalize(gray_rgb, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            color_image = cv.cvtColor(image_numpy, cv.COLOR_BGR2RGB)
            color_image = cv.normalize(color_image, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            color_rgb = cv.cvtColor(rgb_numpy, cv.COLOR_BGR2RGB)
            color_rgb = cv.normalize(color_rgb, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

            # rgb_numpy_check = cv.cvtColor(rgb_numpy, cv.COLOR_BGR2RGB)
            # rgb_numpy_check = cv.normalize(rgb_numpy, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            #
            # cv.imwrite(
            #     f"/home/leo/nerfstudio_reg/nerfstudio/check/render_step_{step}_{random.randint(0, 100)}.png",
            #     rgb_numpy_check)
            # import ipdb; ipdb.set_trace()

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            MIN_MATCH_COUNT = 10

            if rgb_mask_numpy.any() and image_mask_numpy.any():
                # Calculate Homography
                sift = cv.SIFT_create()
                kp_image, des_image = sift.detectAndCompute(color_image, image_mask_numpy)
                kp_rgb, des_rgb = sift.detectAndCompute(color_rgb, None)

                # image_numpy = cv.normalize(image_numpy, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
                # rgb_numpy = cv.normalize(rgb_numpy, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
                # # Calculate Homography
                # sift = cv.SIFT_create()
                # kp_image, des_image = sift.detectAndCompute(image_numpy, None)
                # kp_rgb, des_rgb = sift.detectAndCompute(rgb_numpy, None)

                # img2 = cv.drawKeypoints(image_numpy, kp_image, None)
                # cv.imwrite(f"/home/leo/nerfstudio_reg/nerfstudio/check/step_{step}_image_siftkpgray.jpg", img2)
                # img2 = cv.drawKeypoints(rgb_numpy, kp_rgb, None)
                # cv.imwrite(f"/home/leo/nerfstudio_reg/nerfstudio/check/step_{step}_rgb_siftkpgray.jpg", img2)


                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)  # or pass empty dictionary
                flann = cv.FlannBasedMatcher(index_params, search_params)

                bf = cv.BFMatcher(crossCheck=True)
            else:
                kp_rgb = []
                print("Skipped - All image masked")

            if len(kp_rgb) == 0:
                matches = []
            else:
                # matches = flann.knnMatch(des_image, des_rgb, k=2)
                matches = bf.knnMatch(des_image, des_rgb, k=1)
            # print("step:", step, "matches:", len(matches))

            # Need to draw only good matches, so create a mask
            good = []
            # ratio test as per Lowe's paper
            for i, m in enumerate(matches):
                if len(m) > 0:
                    good.append(m[0])
            # print("step:", step, "good:", len(good))


            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp_image[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_rgb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 7.0)
                matchesMask = mask.ravel().tolist()
                if np.array(matchesMask).sum() == 0:
                    print("matchesMask is 0 for in all entries")
                    matchesMask = None
                else:
                    h, w = color_image.shape[:-1]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv.perspectiveTransform(pts, M)
                    color_rgb_poly = cv.polylines(color_rgb, [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)

                    draw_params = dict(matchColor=(0, 255, 0),
                                       singlePointColor=None,
                                       matchesMask=None,
                                       flags=2)
                    # img3 = cv.drawMatches(color_image, kp_image, color_rgb_poly, kp_rgb, good, None, **draw_params)
                    # print("Found - {} matches".format(len(good)))
                    # print(np.linalg.det(M))
                    # img2 = cv.drawKeypoints(gray_rgb, kp_rgb, None)
                    # cv.imwrite(f"/home/leo/nerfstudio_reg/nerfstudio/check/step_{step}_rgb_siftkpgray_{len(good)}.jpg", img3)
            else:
                # img2 = cv.drawKeypoints(rgb_numpy, kp_rgb, None)
                # cv.imwrite(f"/home/leo/nerfstudio_reg/nerfstudio/check/step_{step}_rgb_siftkpgray_{len(good)}.jpg", img2)
                print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
                matchesMask = None

            # image_numpy2 = cv.cvtColor(image_numpy, cv.COLOR_BGR2RGB)
            # image_numpy2 = cv.normalize(image_numpy2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            # rgb_numpy2 = cv.cvtColor(rgb_numpy, cv.COLOR_BGR2RGB)
            # rgb_numpy2 = cv.normalize(rgb_numpy2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            #
            # cv.imwrite(
            #     f"/home/leo/nerfstudio_reg/nerfstudio/check/homography_step_{step}_random_{random.randint(0, 100)}.png",
            #     np.concatenate((image_numpy2, rgb_numpy2), axis=1))

            if matchesMask == None:
                loss = -10
            else:

                # map src to dst
                im_image_dst = cv.warpPerspective(image_numpy, M, (w, h))
                mask = np.ones(image_numpy.shape, dtype=np.uint8)
                mask = cv.warpPerspective(mask, M, (w, h))

                threshold = 0.5
                precentage = mask.sum() / (mask.shape[0] * mask.shape[1] * 3)


                det = np.linalg.det(M)
                if 0.1 < det < 10 and precentage > threshold:
                    image_numpy = cv.cvtColor(image_numpy, cv.COLOR_BGR2RGB)
                    image_numpy = cv.normalize(image_numpy, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
                    im_image_dst = cv.cvtColor(im_image_dst, cv.COLOR_BGR2RGB)
                    im_image_dst = cv.normalize(im_image_dst, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
                    rgb_numpy = cv.cvtColor(rgb_numpy, cv.COLOR_BGR2RGB)
                    rgb_numpy = cv.normalize(rgb_numpy, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

                    rgb_for_loss = torch.from_numpy(np.moveaxis(rgb_numpy*mask, -1, 0)).to(dtype=float)
                    image_for_loss = torch.from_numpy(np.moveaxis(im_image_dst, -1, 0)).to(dtype=float)
                    loss = -self.rgb_loss(image_for_loss.unsqueeze(0), rgb_for_loss.unsqueeze(0)) / mask.sum()

                    # cv.imwrite(f"/home/leo/nerfstudio_reg/nerfstudio/check/homography_step_{step}_loss_{loss:.7f}_det_{det:.4f}.png",
                    #            np.concatenate((image_numpy, im_image_dst, rgb_numpy), axis=1))
                else:
                    loss = -10

            viewshed_score = loss


                # cv.imwrite(f"/home/leo/nerfstudio_reg/nerfstudio/check/image_loss_{loss:.0f}.png", img3)
        else:
            if self.mse_init:
                # viewshed_score = -self.rgb_loss(image.unsqueeze(0), rgb.unsqueeze(0))
                viewshed_score = -self.rgb_loss(image[image_mask.squeeze()].unsqueeze(0), rgb[image_mask.squeeze()].unsqueeze(0))
            else:
                rgb_mask_numpy, rgb_mask_colormap = get_mask_from_view_likelihood(
                    torch.moveaxis(outputs["view_log_likelihood"], -1, 0), True)
                rgb_mask_colormap = rgb_mask_colormap.squeeze()

                viewshed = outputs["view_log_likelihood"]
                # Find the minimum non-NaN value
                min_value = torch.min(viewshed[~torch.isnan(viewshed)])
                # Replace NaN values with the minimum non-NaN value
                viewshed[torch.isnan(viewshed)] = min_value
                # print("before exp:", output.min(), output.max())
                viewshed = torch.exp(viewshed)
                viewshed = torch.nan_to_num(viewshed)

                viewshed_score = viewshed.sum()
                # viewshed_score = viewshed[image_mask].sum()

                # rgb_numpy = rgb.cpu().numpy()
                # rgb_numpy_check = cv.cvtColor(rgb_numpy, cv.COLOR_BGR2RGB)
                # rgb_numpy_check = cv.normalize(rgb_numpy_check, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
                #
                # depth_numpy = outputs["depth"].cpu().numpy()
                # depth_numpy_check = cv.cvtColor(depth_numpy, cv.COLOR_BGR2RGB)
                # depth_numpy_check = cv.normalize(depth_numpy_check, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
                #
                # # viewshed_numpy = rgb_mask_colormap.cpu().numpy()
                # # viewshed_numpy_check = cv.cvtColor(viewshed_numpy, cv.COLOR_BGR2RGB)
                # # viewshed_numpy_check = cv.normalize(rgb_numpy_check, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
                #
                # # img_numpy = image.cpu().numpy()
                # # img_numpy_check = cv.cvtColor(img_numpy, cv.COLOR_BGR2RGB)
                # # img_numpy_check = cv.normalize(img_numpy_check, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
                # #
                # # print(rgb_mask_colormap.shape)
                # # print(depth.shape)
                # cv.imwrite(f"/home/leo/nerfstudio_reg/nerfstudio/check/rgb_step_{step}_loss_{viewshed_score:.0f}.png", rgb_numpy_check)
                # cv.imwrite(f"/home/leo/nerfstudio_reg/nerfstudio/check/vf_step_{step}_loss_{viewshed_score:.0f}.png", rgb_mask_colormap)
                # cv.imwrite(f"/home/leo/nerfstudio_reg/nerfstudio/check/depth_step_{step}_loss_{viewshed_score:.0f}.png", depth_numpy_check)
                # # cv.imwrite(f"/home/leo/nerfstudio_reg/nerfstudio/check/image_step_{step}_loss_{viewshed_score:.0f}.png", img_numpy_check)
                # # print(image_mask.shape)
                # # viewshed_score = -self.rgb_loss(image[image_mask.repeat(1, 1, 3)].unsqueeze(0), rgb[image_mask.repeat(1, 1, 3)].unsqueeze(0))

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)
        # metrics_dict["loss"] = float(loss)
        metrics_dict["viewshed_score"] = float(viewshed_score)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
