import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from utils import (
    GaussianModel,
    load_gaussians,
)
from .rotation import build_y_rotation_matrix

PreparedFrame = Tuple[torch.Tensor, torch.Tensor]


def _prepare_frames(gaussians: Sequence[GaussianModel]) -> List[PreparedFrame]:
    prepared: List[PreparedFrame] = []
    for gaussian in gaussians:
        xyz = gaussian.get_xyz.detach()
        xyz_centered = xyz - xyz.mean(dim=0, keepdim=True)
        prepared.append((xyz_centered, xyz_centered[:, [0, 2]]))
    return prepared


def _window_weights(window_size: int) -> List[float]:
    center = (window_size - 1) * 0.5
    sigma = max(window_size / 3.0, 1e-6)
    return [
        math.exp(-((offset - center) ** 2) / (2 * sigma ** 2))
        for offset in range(window_size)
    ]


def _estimate_window_rotation(
    prepared_a: Sequence[PreparedFrame],
    prepared_b: Sequence[PreparedFrame],
    start_a: int,
    start_b: int,
    weights: Sequence[float],
) -> float:
    a_sum = 0.0
    b_sum = 0.0

    for offset, weight in enumerate(weights):
        xz_a = prepared_a[start_a + offset][1]
        xz_b = prepared_b[start_b + offset][1]

        point_count = min(xz_a.shape[0], xz_b.shape[0])
        if point_count == 0:
            continue

        xz_a = xz_a[:point_count]
        xz_b = xz_b[:point_count]

        px = xz_a[:, 0]
        pz = xz_a[:, 1]
        qx = xz_b[:, 0]
        qz = xz_b[:, 1]

        a_sum += weight * torch.sum(px * qx + pz * qz).item()
        b_sum += weight * torch.sum(px * qz - pz * qx).item()

    return float(math.atan2(b_sum, a_sum))


def _compute_window_distance(
    prepared_a: Sequence[PreparedFrame],
    prepared_b: Sequence[PreparedFrame],
    start_a: int,
    start_b: int,
    window_size: int,
    theta: float,
) -> float:
    rotation = None
    if theta != 0.0:
        sample_xyz = prepared_b[start_b][0]
        rotation = build_y_rotation_matrix(
            theta,
            device=sample_xyz.device,
            dtype=sample_xyz.dtype,
        )

    total_distance = 0.0
    compared_frames = 0
    for offset in range(window_size):
        xyz_a = prepared_a[start_a + offset][0]
        xyz_b = prepared_b[start_b + offset][0]

        point_count = min(xyz_a.shape[0], xyz_b.shape[0])
        if point_count == 0:
            continue

        xyz_a = xyz_a[:point_count]
        xyz_b = xyz_b[:point_count]

        if rotation is not None:
            xyz_b = xyz_b @ rotation.T

        total_distance += torch.norm(xyz_a - xyz_b, dim=1).mean().item()
        compared_frames += 1

    if compared_frames == 0:
        return float("inf")
    return total_distance / compared_frames

def compute_similarity_matrix(
    frames_a: List[Path],
    frames_b: List[Path],
    sh_degree: int,
    window_size: int,
    min_gap: int,
    gaussians_a: Optional[List[GaussianModel]] = None,
    gaussians_b: Optional[List[GaussianModel]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if gaussians_a is None:
        gaussians_a = load_gaussians(frames_a, sh_degree=sh_degree)
    if gaussians_b is None:
        gaussians_b = load_gaussians(frames_b, sh_degree=sh_degree)

    n_a = len(gaussians_a)
    n_b = len(gaussians_b)

    if window_size <= 0:
        raise ValueError("window_size must be positive")

    matrix = torch.full((n_a, n_b), float("inf"), dtype=torch.float32)
    angle_matrix = torch.zeros((n_a, n_b), dtype=torch.float32)

    if window_size > n_a or window_size > n_b:
        return matrix, angle_matrix

    prepared_a = _prepare_frames(gaussians_a)
    prepared_b = prepared_a if gaussians_a is gaussians_b else _prepare_frames(gaussians_b)
    same_sequence = frames_a == frames_b or gaussians_a is gaussians_b
    weights = _window_weights(window_size)

    valid_start_a = 0
    valid_end_a = n_a - window_size + 1

    valid_start_b = 0
    valid_end_b = n_b - window_size + 1

    with torch.inference_mode():
        for s in tqdm(
            range(valid_start_a, valid_end_a),
            desc="Computing similarity matrix",
            unit="row",
        ):
            for t in range(valid_start_b, valid_end_b):
                if same_sequence and abs(s - t) < min_gap:
                    continue

                theta = _estimate_window_rotation(
                    prepared_a,
                    prepared_b,
                    start_a=s,
                    start_b=t,
                    weights=weights,
                )

                angle_matrix[s, t] = theta

                matrix[s, t] = _compute_window_distance(
                    prepared_a,
                    prepared_b,
                    start_a=s,
                    start_b=t,
                    window_size=window_size,
                    theta=theta,
                )

    return matrix, angle_matrix


def transpose_similarity_matrices(
    similarity_matrices: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    distance_matrix, angle_matrix = similarity_matrices
    return (
        distance_matrix.transpose(0, 1).contiguous(),
        angle_matrix.transpose(0, 1).neg().contiguous(),
    )

def save_similarity_matrix(
    similarity_matrices: Tuple[torch.Tensor, torch.Tensor],
    output_dir: Path,
    window_size: int,
    min_gap: int,
) -> None:
    distance_matrix, angle_matrix = similarity_matrices
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "distance_matrix": distance_matrix,
            "angle_matrix": angle_matrix,
            "window_size": window_size,
            "min_gap": min_gap,
            "source_index_mode": "window_start",
            "target_index_mode": "window_start",
        },
        output_dir / "similarity.pt",
    )
