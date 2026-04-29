import math

import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from utils import GaussianModel

from .models import MiniCam


def render_official_gaussian_frame(
    gaussian: GaussianModel,
    mini_cam: MiniCam,
    background: torch.Tensor,
    scaling_modifier: float = 1.0,
) -> np.ndarray:
    screenspace_points = torch.zeros_like(
        gaussian.get_xyz,
        dtype=gaussian.get_xyz.dtype,
        device=gaussian.get_xyz.device,
        requires_grad=True,
    )
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    tanfovx = math.tan(mini_cam.FoVx * 0.5)
    tanfovy = math.tan(mini_cam.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(mini_cam.image_height),
        image_width=int(mini_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=float(scaling_modifier),
        viewmatrix=mini_cam.world_view_transform,
        projmatrix=mini_cam.full_proj_transform,
        sh_degree=int(gaussian.active_sh_degree),
        campos=mini_cam.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, _ = rasterizer(
        means3D=gaussian.get_xyz,
        means2D=screenspace_points,
        shs=gaussian.get_features,
        colors_precomp=None,
        opacities=gaussian.get_opacity,
        scales=gaussian.get_scaling,
        rotations=gaussian.get_rotation,
        cov3D_precomp=None,
    )

    image = rendered_image.detach().clamp(0.0, 1.0).permute(1, 2, 0).contiguous().cpu().numpy()
    return image
