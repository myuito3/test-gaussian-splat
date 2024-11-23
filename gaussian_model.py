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
from simple_knn._C import distCUDA2

from gaussian_splatting.utils.general_utils import (
    inverse_sigmoid,
    strip_symmetric,
    build_scaling_rotation,
)
from gaussian_splatting.utils.sh_utils import RGB2SH


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

    def create_from_params(self, fused_point_cloud, features, scales, rots, opacities):
        self._xyz = fused_point_cloud.clone()
        self._features_dc = features[:, :, 0:1].clone().transpose(1, 2).contiguous()
        self._features_rest = features[:, :, 1:].clone().transpose(1, 2).contiguous()
        self._scaling = scales.clone()
        self._rotation = rots.clone()
        self._opacity = opacities.clone()


def convert_colors_to_features(colors, sh_degree=3, device="cuda"):
    sh_colors = RGB2SH(colors.float())
    features = torch.zeros(
        (sh_colors.shape[0], 3, (sh_degree + 1) ** 2), device=device
    ).float()
    features[:, :3, 0] = sh_colors
    features[:, 3:, 1:] = 0.0
    return features


def compute_scales(fused_point_cloud, device="cuda"):
    dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    return scales


def get_gaussian_rots(num_points, device="cuda"):
    rots = torch.zeros((num_points, 4), device=device).float()
    rots[:, 0] = 1
    return rots


def get_gaussian_opacities(num_points, initial_opacity=1.0, device="cuda"):
    opacities = inverse_sigmoid(
        initial_opacity * torch.ones((num_points, 1), device=device).float()
    )
    return opacities


def create_gaussian_params(fused_point_cloud, colors, sh_degree=3, device="cuda"):
    features = convert_colors_to_features(colors)
    scales = compute_scales(fused_point_cloud)
    rots = get_gaussian_rots(fused_point_cloud.shape[0])
    opacities = get_gaussian_opacities(fused_point_cloud.shape[0])
    return features, scales, rots, opacities
