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
Field for compound Normalizing Flow
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import Encoding, HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    PredNormalsFieldHead,
    DirectionsFieldHead,
    RGBFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
    # ViewLikelihoodFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
import normflows as nf



class NFField(Field):
    """Compound Field

    Args:
        num_layers: number of NF layers
        num_dims: dimension of Gaussian
        hidden_dim: dimension of hidden layers
    """

    def __init__(
        self,
        num_layers: int = 4,
        num_dims: int = 6,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()


        # view likelihood
        # num_dims = self.direction_encoding.n_output_dims + self.position_encoding.n_output_dims

        # Define 6D Gaussian base distribution
        base = nf.distributions.base.DiagGaussian(num_dims)

        # Define list of flows
        flows = []
        for i in range(num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            # param_map = nf.nets.MLP([3, 64, 64, 6], init_zeros=True)
            param_map = nf.nets.MLP([int(num_dims/2), hidden_dim, hidden_dim, num_dims], init_zeros=True, leaky=0.1, dropout=0.2)
            # param_map = nf.nets.MLP([int(num_dims/2), hidden_dim, hidden_dim, num_dims], init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map="exp"))
            # Swap dimensions
            flows.append(nf.flows.Permute(num_dims, mode='swap'))

        self.nf_model = nf.NormalizingFlow(base, flows)

        # nf_args = get_args()
        # self.nf_model = get_point_cnf(nf_args)
        # print(self.nf_model)


    def get_outputs(
        # self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
        self, positions, directions
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs = {}
        # if ray_samples.camera_indices is None:
        #     raise AttributeError("Camera indices are not provided.")

        # outputs_shape = ray_samples.frustums.directions.shape[:-1]
        outputs_shape = directions.shape[:-1]
        # directions = shift_directions_for_tcnn(ray_samples.frustums.directions)

        # predicted view likelihood

        # positions = ray_samples.frustums.get_positions().view(-1, 3)
        positions = positions.view(-1, 3)

        # positions = self.position_encoding(positions).to(dtype=torch.float32)

        # dirs = shift_directions_for_tcnn(ray_samples.frustums.directions).contiguous().view(-1, 3)

        # dirs = shift_directions_for_tcnn(directions).contiguous().view(-1, 3)
        dirs = directions.contiguous().view(-1, 3)

        # dirs = self.direction_encoding(dirs).to(dtype=torch.float32)
        # print(positions.shape)
        # print(dirs.shape)
        # positions_flat = self.position_encoding(positions.view(-1, 3))
        # view_likelihood_inp = torch.cat([d, positions_flat], dim=-1)
        view_likelihood_inp = torch.cat([positions, dirs], dim=-1)

        # x = self.mlp_view_likelihood(view_likelihood_inp).view(*outputs_shape, -1).to(directions)
        # print(view_likelihood_inp.shape)
        # import ipdb; ipdb.set_trace()
        # print(view_likelihood_inp.shape)


        outputs[FieldHeadNames.VIEW_LOG_LIKELIHOOD] = self.nf_model.log_prob(view_likelihood_inp).view(*outputs_shape, -1).to(directions)

        # # Compute the reconstruction likelihood P(X|z)
        # batch_size = view_likelihood_inp.size(0)
        # num_points = 1
        # y, delta_log_py = self.nf_model(view_likelihood_inp.unsqueeze(1), None, torch.zeros(batch_size, num_points, 1).to(view_likelihood_inp))
        # # log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        # # delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        # log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        # delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        # log_px = log_py - delta_log_py

        # outputs[FieldHeadNames.VIEW_LOG_LIKELIHOOD] = log_px.view(*outputs_shape, -1).to(directions)


        # with torch.no_grad():
        #     outputs[FieldHeadNames.VIEW_LIKELIHOOD] = torch.exp(outputs[FieldHeadNames.VIEW_LOG_LIKELIHOOD])
        # print(x)
        # .view(*outputs_shape, -1).to(directions)

        return outputs
