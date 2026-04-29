import torch

from utils import GaussianModel

from .rotation import rotate_points
from typing import List

def compute_frame_distance(g1: GaussianModel, g2: GaussianModel, theta: float = 0.0) -> float:
    xyz1 = g1.get_xyz
    xyz2 = g2.get_xyz

    center1 = xyz1.mean(dim=0, keepdim=True)
    center2 = xyz2.mean(dim=0, keepdim=True)

    xyz1_centered = xyz1 - center1
    xyz2_centered = xyz2 - center2

    xyz2_centered = rotate_points(xyz2_centered, theta)
    point_distances = torch.norm(xyz1_centered - xyz2_centered, dim=1)
    return point_distances.mean().item()

def compute_sequence_distance(
    gaussians_a: List[GaussianModel],
    gaussians_b: List[GaussianModel],
    theta: float = 0.0,
) -> float:
    total_distance = 0.0

    for g1, g2 in zip(gaussians_a, gaussians_b):
        total_distance += compute_frame_distance(g1, g2, theta=theta)

    return total_distance / len(gaussians_a)
