import math

import torch

from gaussian_splatting.utils.general_utils import inverse_sigmoid
from gaussian_splatting.utils.sh_utils import RGB2SH
from camera import Camera, load_cam
from gaussian_model import GaussianModel
from render import render


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
    rgb, depth, num_verts_h, num_verts_w, sh_degree=3, device="cuda"
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
    radius = depth[0, grid_points[:, 1].long(), grid_points[:, 0].long()]

    fused_point_cloud = torch.tensor(
        [[0.0, 0.0, 0.0] for _ in range(num_verts_h) for _ in range(num_verts_w)],
        device=device,
    ).float()
    fused_point_cloud[:, 0] = radius * torch.sin(theta) * torch.sin(phi)
    fused_point_cloud[:, 1] = radius * torch.sin(theta) * torch.cos(phi)
    fused_point_cloud[:, 2] = radius * torch.cos(theta)

    colors = rgb[:3, grid_points[:, 1].long(), grid_points[:, 0].long()].T
    colors = RGB2SH(colors.float())
    features = (
        torch.zeros((colors.shape[0], 3, (sh_degree + 1) ** 2)).float().to(device)
    )
    features[:, :3, 0] = colors
    features[:, 3:, 1:] = 0.0

    scales = torch.ones((fused_point_cloud.shape[0], 1), device=device) * torch.log(
        torch.sqrt(torch.tensor([0.05], device=device))
    )
    scales = scales.float()

    rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device).float()
    rots[:, 0] = 1
    opacities = inverse_sigmoid(
        0.7 * torch.ones((fused_point_cloud.shape[0], 1), device=device).float()
    )

    return fused_point_cloud, features, scales, rots, opacities


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    camera_list = load_cam(data_dir="data/test")

    camera: Camera = camera_list[0]

    fused_point_cloud, features, scales, rots, opacities = (
        create_pcd_from_image_and_depth(
            rgb=camera.image,
            depth=camera.depth,
            num_verts_h=camera.image_height // 4,
            num_verts_w=camera.image_width // 4,
            device=device,
        )
    )
    print(
        fused_point_cloud.shape,
        fused_point_cloud.device,
        features.shape,
        features.device,
        scales.shape,
        scales.device,
        rots.shape,
        rots.device,
        opacities.shape,
        opacities.device,
    )

    gaussians = GaussianModel(sh_degree=3)
    gaussians.create_from_pcd(fused_point_cloud, features, scales, rots, opacities)

    import numpy as np
    from gaussian_splatting.utils.graphics_utils import focal2fov

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = np.array([0, 0, 0])
    image = torch.zeros((3, 240, 480))

    render_camera = Camera(
        R=R,
        T=T,
        FovX=focal2fov(500, image.shape[2]),
        FovY=focal2fov(500, image.shape[1]),
        image=image,
        depth=image,
    )
    output = render(render_camera, gaussians)
    image = output["render"]
