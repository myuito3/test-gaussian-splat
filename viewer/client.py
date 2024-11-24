import threading
import time

import numpy as np
import torch
import viser
import viser.transforms as vtf

from camera import Camera
from gaussian_splatting.utils.graphics_utils import focal2fov
from render import render


class ClientThread(threading.Thread):
    def __init__(self, viewer, gaussians, client: viser.ClientHandle):
        super().__init__()
        self.viewer = viewer
        self.gaussians = gaussians
        self.client = client

    def get_camera(self, client: viser.ClientHandle):
        R = vtf.SO3(wxyz=client.camera.wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = R.as_matrix()
        pos = client.camera.position
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        image = torch.zeros((3, 480, 960))
        render_camera = Camera(
            R=R,
            T=T,
            FovX=focal2fov(250, image.shape[2]),
            FovY=focal2fov(250, image.shape[1]),
            image=image,
            depth=image,
        )
        return render_camera

    def render_and_send(self):
        with self.client.atomic():
            camera = self.get_camera(self.client)
            with torch.no_grad():
                output = render(camera, self.gaussians)
            image = output["render"]
            self.client.set_background_image(image.permute(1, 2, 0).cpu().numpy())

    def run(self):
        while True:
            time.sleep(0.1)

            try:
                self.render_and_send()
            except Exception:
                print("error occurred when rendering for client")
                break

        self._destroy()

    def _destroy(self):
        self.viewer = None
        self.renderer = None
        self.client = None
