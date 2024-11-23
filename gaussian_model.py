#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn

from gaussian_splatting.utils.general_utils import (
    inverse_sigmoid,
    strip_symmetric,
    build_scaling_rotation,
)


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_exposure(self):
        return self._exposure

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(self, fused_point_cloud, features, scales, rots, opacities):
        self._xyz = nn.Parameter(fused_point_cloud)
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous()
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous()
        )
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(opacities)

    def extend_from_pcd(self, fused_point_cloud, features, scales, rots, opacities):
        self._xyz = nn.Parameter(torch.cat((self._xyz, fused_point_cloud), dim=0))
        self._features_dc = nn.Parameter(
            torch.cat((self._features_dc, features[:, :, 0:1]), dim=0)
        )
        self._features_rest = nn.Parameter(
            torch.cat((self._features_rest, features[:, :, 1:]), dim=0)
        )
        self._scaling = nn.Parameter(torch.cat((self._scaling, scales), dim=0))
        self._rotation = nn.Parameter(torch.cat((self._rotation, rots), dim=0))
        self._opacity = nn.Parameter(torch.cat((self._opacity, opacities), dim=0))
