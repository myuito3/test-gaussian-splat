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

import numpy as np
import torch

from gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera:

    def __init__(
        self,
        R,
        T,
        FovX,
        FovY,
        image,
        depth=None,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        device="cuda",
    ) -> None:

        self.R = R
        self.T = T
        self.FoVx = FovX
        self.FoVy = FovY

        self.image = image.to(device)
        self.image_height = self.image.shape[1]
        self.image_width = self.image.shape[2]

        self.depth = depth
        if depth is not None:
            self.depth = self.depth.to(device)

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
