from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FixedCamera:
    width: int
    height: int
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_deg: float
    znear: float = 0.05
    zfar: float = 1000.0


@dataclass
class MiniCam:
    image_width: int
    image_height: int
    FoVx: float
    FoVy: float
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
