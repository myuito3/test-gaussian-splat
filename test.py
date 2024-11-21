import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gaussian_splatting.utils.general_utils import (
    inverse_sigmoid,
    strip_symmetric,
    build_scaling_rotation,
    PILtoTorch,
)
from gaussian_splatting.utils.graphics_utils import (
    getWorld2View2,
    getProjectionMatrix,
    focal2fov,
    fov2focal,
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

    def extend_from_pcd(self, fused_point_cloud, features_dc, scales, rots, opacities):
        self._xyz = torch.cat((self._xyz, fused_point_cloud), dim=0)
        self._features_dc = torch.cat((self._features_dc, features_dc), dim=0)
        self._features_rest = torch.cat((self._features_rest, features_rest), dim=0)
        self._scaling = torch.cat((self._scaling, scales), dim=0)
        self._rotation = torch.cat((self._rotation, rots), dim=0)
        self._opacity = torch.cat((self._opacity, opacities), dim=0)


class Camera:

    def __init__(
        self,
        R,
        T,
        FovX,
        FovY,
        image,
        invdepthmap,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        device="cuda",
    ) -> None:

        self.R = R
        self.T = T
        self.FovX = FovX
        self.FovY = FovY

        self.gt_image = image.to(device)
        self.invdepthmap = invdepthmap.to(device)
        self.height = self.gt_image.shape[1]
        self.width = self.gt_image.shape[2]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FovX, fovY=self.FovY
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def load_cam(data_dir):
    import json

    data_dir = Path(data_dir)
    with open(data_dir / "transforms.json", mode="r", encoding="utf-8") as f:
        meta = json.load(f)

    camera_list = []
    for frame in meta["frames"]:

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        resolution = (meta["h"], meta["w"])
        image_path = data_dir / frame["file_path"]
        image = PILtoTorch(Image.open(image_path), resolution)

        depth_path = data_dir / frame["depth_path"]
        invdepthmap = PILtoTorch(Image.open(depth_path), resolution) / 512

        FovY = focal2fov(meta["fl_y"], image.shape[2])
        FovX = focal2fov(meta["fl_x"], image.shape[1])

        camera_list.append(
            Camera(R=R, T=T, FovX=FovX, FovY=FovY, image=image, invdepthmap=invdepthmap)
        )

    return camera_list


def get_grid_points_2d(
    y_linspace_args=(0, 1, 100), x_linspace_args=(0, 1, 100), device: str = "cuda"
):
    """Returns a set of 2d coordinates representing the grid points.

    Args:
        y_linspace_args: A pair of start and end points and number of steps in the
            y-axis direction coordinates.
        x_linspace_args: A pair of start and end points and number of steps in the
            x-axis direction coordinates.
        device: Device to place elements on.
    """
    y_start, y_end, y_steps = y_linspace_args
    points_y = torch.linspace(y_start, y_end, steps=y_steps)
    x_start, x_end, x_steps = x_linspace_args
    points_x = torch.linspace(x_start, x_end, steps=x_steps)

    grid_points = torch.tensor(
        [[x, y] for y in points_y for x in points_x], dtype=torch.float, device=device
    )
    return grid_points


def create_pcd_from_image_and_depth(
    rgb, depth, num_verts_h, num_verts_w, device="cuda", sh_degree=3
):
    _, h, w = rgb.shape

    grid_points = get_grid_points_2d(
        y_linspace_args=(0, h - 1, num_verts_h),
        x_linspace_args=(0, w - 1, num_verts_w),
        device=device,
    )
    """
    tensor([[   0.0000,    0.0000],
            [   2.0010,    0.0000],
            [   4.0021,    0.0000],
            ...,
            [1914.9979,  959.0000],
            [1916.9989,  959.0000],
            [1919.0000,  959.0000]], device='cuda:0')
    """
    theta = grid_points[:, 1] * math.pi / (h - 1)
    phi = grid_points[:, 0] * 2 * math.pi / (w - 1)
    radius = depth[grid_points[:, 1].long(), grid_points[:, 0].long()]

    fused_point_cloud = torch.tensor(
        [[0.0, 0.0, 0.0] for _ in range(num_verts_h) for _ in range(num_verts_w)],
    ).to(device)
    fused_point_cloud[:, 0] = radius * torch.sin(theta) * torch.sin(phi)
    fused_point_cloud[:, 1] = radius * torch.sin(theta) * torch.cos(phi)
    fused_point_cloud[:, 2] = radius * torch.cos(theta)

    colors = rgb[:, grid_points[:, 1].long(), grid_points[:, 0].long()].T
    colors = RGB2SH(colors.float().cuda())
    features = torch.zeros((colors.shape[0], 3, (sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = colors
    features[:, 3:, 1:] = 0.0

    scales = torch.ones((fused_point_cloud.shape[0], 1), device=device) * torch.log(
        torch.sqrt(torch.tensor([0.05], device=device))
    )

    rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
    rots[:, 0] = 1
    opacities = inverse_sigmoid(
        0.7
        * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device)
    )

    return fused_point_cloud, features, scales, rots, opacities


if __name__ == "__main__":
    camera_list = load_cam(data_dir="data/test")

    camera: Camera = camera_list[0]
    num_verts_h = camera.height // 4
    num_verts_w = camera.width // 4

    fused_point_cloud, features, scales, rots, opacities = (
        create_pcd_from_image_and_depth(
            rgb=camera.gt_image,
            depth=camera.invdepthmap,
            num_verts_h=num_verts_h,
            num_verts_w=num_verts_w,
        )
    )
