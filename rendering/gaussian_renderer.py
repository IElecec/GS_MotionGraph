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
        requires_grad=True,
        device="cuda",
    ) + 0

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
        scale_modifier=scaling_modifier,
        viewmatrix=mini_cam.world_view_transform,
        projmatrix=mini_cam.full_proj_transform,
        sh_degree=int(gaussian.active_sh_degree),
        campos=mini_cam.camera_center,
        prefiltered=False,
        debug=False,
        pixel_weights=None,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussian.get_xyz
    means2D = screenspace_points
    opacity = gaussian.get_opacity
    scales = gaussian.get_scaling
    rotations = gaussian.get_rotation
    cov3D_precomp = None
    colors_precomp = None

    # separate SH:
    # full features: [N, 16, 3] when sh_degree=3
    # dc:            [N, 1, 3]
    # shs:           [N, 15, 3]
    features = gaussian.get_features
    dc = features[:, :1, :].contiguous()
    shs = features[:, 1:, :].contiguous()

    rendered_image, radii, counts, lists, listsRender, listsDistance, centers, depths, my_radii, accum_weights, accum_count, accum_blend, accum_dist = rasterizer(
        means3D=means3D.contiguous(),
        means2D=means2D.contiguous(),
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity.contiguous(),
        scales=scales.contiguous(),
        rotations=rotations.contiguous(),
        cov3D_precomp=cov3D_precomp,
    )

    image = (
        rendered_image
        .detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )

    return image