import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from gaussian_splatting.utils.graphics_utils import focal2fov
from camera import Camera
from gaussian_model import GaussianModel, create_gaussian_params
from viewer.viewer import Viewer


def get_grid_points_2d(height, width, interval=1):
    points_y = torch.linspace(0, height - 1, steps=height // interval)
    points_x = torch.linspace(0, width - 1, steps=width // interval)
    grid_points = torch.tensor(
        [[x, y] for y in points_y for x in points_x], dtype=torch.float
    )
    return grid_points


def create_pcd_from_equirectangular_image(rgb, depth=None, device="cuda"):
    _, h, w = rgb.shape

    if depth is None:
        depth_scale = 10
        depth = torch.ones((1, h, w), device=device) * depth_scale

    grid_points = get_grid_points_2d(h, w, interval=2).to(device)

    fused_point_cloud = torch.zeros((grid_points.shape[0], 3), device=device).float()
    radius = depth[0, grid_points[:, 1].long(), grid_points[:, 0].long()]
    theta = grid_points[:, 1] * math.pi / (h - 1)
    phi = grid_points[:, 0] * 2 * math.pi / (w - 1)
    fused_point_cloud[:, 0] = radius * torch.sin(theta) * torch.sin(phi)
    fused_point_cloud[:, 1] = radius * torch.sin(theta) * torch.cos(phi)
    fused_point_cloud[:, 2] = radius * torch.cos(theta)

    colors = rgb[:3, grid_points[:, 1].long(), grid_points[:, 0].long()].T  # [N, 3]
    return fused_point_cloud, colors


def get_camera_list_from_transforms(transforms_data: dict, data_dir: Path):
    camera_list = []

    for frame in transforms_data["frames"]:
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

        image_path = data_dir / frame["file_path"]
        image = pil_to_tensor(Image.open(image_path)).float() / 255.0

        depth_path = data_dir / frame["depth_path"]
        depth = pil_to_tensor(Image.open(depth_path)).float()
        depth = (depth * 5).clamp(0, 1000)

        FovY = focal2fov(transforms_data["fl_y"], image.shape[2])
        FovX = focal2fov(transforms_data["fl_x"], image.shape[1])

        camera_list.append(
            Camera(R=R, T=T, FovX=FovX, FovY=FovY, image=image, depth=depth)
        )

    return camera_list


def main():
    if not torch.cuda.is_available():
        print("runs only with cuda device, exit")
        return

    data_dir = Path("data/test")
    with open(data_dir / "transforms.json", mode="r", encoding="utf-8") as f:
        meta = json.load(f)

    camera_list = get_camera_list_from_transforms(meta, data_dir)
    camera: Camera = camera_list[0]

    fused_point_cloud, colors = create_pcd_from_equirectangular_image(
        rgb=camera.image, depth=camera.depth
    )
    features, scales, rots, opacities = create_gaussian_params(
        fused_point_cloud, colors
    )

    gaussians = GaussianModel(sh_degree=3)
    gaussians.create_from_params(fused_point_cloud, features, scales, rots, opacities)

    _ = Viewer(gaussians=gaussians)

    while True:
        time.sleep(999)


if __name__ == "__main__":
    main()
