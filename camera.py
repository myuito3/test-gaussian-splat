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

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gaussian_splatting.utils.general_utils import PILtoTorch
from gaussian_splatting.utils.graphics_utils import (
    getWorld2View2,
    getProjectionMatrix,
    focal2fov,
)


class Camera:

    def __init__(
        self,
        R,
        T,
        FovX,
        FovY,
        image,
        depth,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        device="cuda",
    ) -> None:

        self.R = R
        self.T = T
        self.FoVx = FovX
        self.FoVy = FovY

        self.image = image.to(device)
        self.depth = depth.to(device)
        self.image_height = self.image.shape[1]
        self.image_width = self.image.shape[2]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(device)
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def load_cam(data_dir, device="cuda"):
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

        resolution = (meta["w"], meta["h"])
        image_path = data_dir / frame["file_path"]
        image = PILtoTorch(Image.open(image_path), resolution)

        depth_path = data_dir / frame["depth_path"]
        depth = PILtoTorch(Image.open(depth_path), resolution)
        depth *= 255
        depth = (depth * 5).clamp(0, 200)

        FovY = focal2fov(meta["fl_y"], image.shape[2])
        FovX = focal2fov(meta["fl_x"], image.shape[1])

        camera_list.append(
            Camera(
                R=R, T=T, FovX=FovX, FovY=FovY, image=image, depth=depth, device=device
            )
        )

    return camera_list
