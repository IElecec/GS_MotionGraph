import math
import torch
from typing import List
from utils import GaussianModel


def estimate_sequence_rotation(
    gaussians_a: List[GaussianModel],
    gaussians_b: List[GaussianModel],
) -> float:
    sequence_length = len(gaussians_a)

    center = (sequence_length - 1) * 0.5
    sigma = max(sequence_length / 3.0, 1e-6)

    a_sum = 0.0
    b_sum = 0.0

    for offset, (ga, gb) in enumerate(zip(gaussians_a, gaussians_b)):
        xyz_a = ga.get_xyz.detach()
        xyz_b = gb.get_xyz.detach()

        xz_a = xyz_a[:, [0, 2]]
        xz_b = xyz_b[:, [0, 2]]

        xz_a = xz_a - xz_a.mean(dim=0, keepdim=True)
        xz_b = xz_b - xz_b.mean(dim=0, keepdim=True)

        n = min(xz_a.shape[0], xz_b.shape[0])
        if n == 0:
            continue

        xz_a = xz_a[:n]
        xz_b = xz_b[:n]

        px, pz = xz_a[:, 0], xz_a[:, 1]
        qx, qz = xz_b[:, 0], xz_b[:, 1]

        w = math.exp(-((offset - center) ** 2) / (2 * sigma ** 2))

        a_sum += w * torch.sum(px * qx + pz * qz).item()
        b_sum += w * torch.sum(px * qz - pz * qx).item()

    return float(math.atan2(b_sum, a_sum))

def build_y_rotation_matrix(
    theta: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        device=device,
        dtype=dtype,
    )

def rotate_points(xyz: torch.Tensor, theta: float) -> torch.Tensor:
    if theta == 0.0:
        return xyz
    rot = build_y_rotation_matrix(theta, device=xyz.device, dtype=xyz.dtype)
    return xyz @ rot.T
