import math
from typing import Optional

import numpy as np
import torch

from utils.graphics_utils import getProjectionMatrix

from .models import FixedCamera, MiniCam


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v.copy()
    return v / norm


def build_fixed_camera(
    reference_xyz: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    azimuth_deg: float,
    elevation_deg: float,
    distance_scale: float,
) -> FixedCamera:
    mins = reference_xyz.min(axis=0)
    maxs = reference_xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    extent = float(np.max(maxs - mins))
    extent = max(extent, 1e-3)

    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    direction = np.array(
        [
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.cos(azimuth),
        ],
        dtype=np.float32,
    )

    position = center + direction * (extent * distance_scale)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return FixedCamera(
        width=width,
        height=height,
        position=position.astype(np.float32),
        target=center.astype(np.float32),
        up=up,
        fov_deg=float(fov_deg),
    )


def build_orbit_camera(
    center: np.ndarray,
    extent: float,
    width: int,
    height: int,
    fov_deg: float,
    azimuth_deg: float,
    elevation_deg: float,
    distance_scale: float,
) -> FixedCamera:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)

    direction = np.array(
        [
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.cos(azimuth),
        ],
        dtype=np.float32,
    )

    distance = max(extent, 1e-3) * distance_scale
    position = center + direction * distance

    return FixedCamera(
        width=width,
        height=height,
        position=position.astype(np.float32),
        target=center.astype(np.float32),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        fov_deg=float(fov_deg),
    )


def parse_vector3(text: Optional[str]) -> Optional[np.ndarray]:
    if text is None:
        return None
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {text}")
    return np.asarray(values, dtype=np.float32)


def build_world_to_view_lookat(
    camera_position: np.ndarray,
    target: np.ndarray,
    up_hint: np.ndarray,
) -> np.ndarray:
    camera_position = camera_position.astype(np.float32)
    target = target.astype(np.float32)
    up_hint = normalize(up_hint.astype(np.float32))

    forward = normalize(target - camera_position)
    # Match the renderer's camera handedness so the image is not mirrored.
    right = normalize(np.cross(up_hint, forward))
    true_up = normalize(np.cross(forward, right))

    if np.linalg.norm(right) < 1e-8 or np.linalg.norm(true_up) < 1e-8:
        raise ValueError("Invalid look-at camera configuration.")

    R_w2c = np.stack([right, true_up, forward], axis=0).astype(np.float32)
    t_w2c = (-R_w2c @ camera_position).astype(np.float32)

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c
    return w2c


def build_mini_cam(camera: FixedCamera, device: torch.device) -> MiniCam:
    fov_y = math.radians(camera.fov_deg)
    fov_x = 2.0 * math.atan(math.tan(fov_y * 0.5) * (camera.width / max(camera.height, 1)))

    w2c = build_world_to_view_lookat(
        camera_position=camera.position,
        target=camera.target,
        up_hint=camera.up,
    )

    world_view = torch.tensor(w2c, dtype=torch.float32, device=device).transpose(0, 1)
    projection = getProjectionMatrix(
        znear=camera.znear,
        zfar=camera.zfar,
        fovX=fov_x,
        fovY=fov_y,
    ).to(device=device, dtype=torch.float32).transpose(0, 1)

    full_proj = world_view @ projection
    camera_center = torch.tensor(camera.position, dtype=torch.float32, device=device)

    return MiniCam(
        image_width=camera.width,
        image_height=camera.height,
        FoVx=fov_x,
        FoVy=fov_y,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
        camera_center=camera_center,
    )
