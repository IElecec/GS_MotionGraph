from pathlib import Path
from typing import List, Tuple
import torch
from tqdm import tqdm

from utils import (
    load_gaussians,
)

from .rotation import estimate_sequence_rotation
from .distance import compute_sequence_distance

def compute_similarity_matrix(
    frames_a: List[Path],
    frames_b: List[Path],
    sh_degree: int,
    window_size: int,
    min_gap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gaussians_a = load_gaussians(frames_a, sh_degree=sh_degree)
    gaussians_b = load_gaussians(frames_b, sh_degree=sh_degree)

    n_a = len(gaussians_a)
    n_b = len(gaussians_b)

    matrix = torch.full((n_a, n_b), float("inf"), dtype=torch.float32)
    angle_matrix = torch.zeros((n_a, n_b), dtype=torch.float32)

    valid_start_a = 0
    valid_end_a = n_a - window_size + 1

    valid_start_b = window_size - 1
    valid_end_b = n_b

    with torch.no_grad():
        for s in tqdm(
            range(valid_start_a, valid_end_a),
            desc="Computing similarity matrix",
            unit="row",
        ):
            for t in range(valid_start_b, valid_end_b):
                if frames_a == frames_b and abs(s - t) < min_gap:
                    continue

                theta = estimate_sequence_rotation(
                    gaussians_a[s : s + window_size], 
                    gaussians_b[t - window_size + 1 : t + 1],
                )

                angle_matrix[s, t] = theta

                matrix[s, t] = compute_sequence_distance(
                    gaussians_a[s : s + window_size],
                    gaussians_b[t - window_size + 1 : t + 1],
                    theta=theta,
                )

    return matrix, angle_matrix

def save_similarity_matrix(
    similarity_matrices: Tuple[torch.Tensor, torch.Tensor],
    output_dir: Path,
) -> None:
    distance_matrix, angle_matrix = similarity_matrices
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "distance_matrix": distance_matrix,
            "angle_matrix": angle_matrix,
        },
        output_dir / "similarity.pt",
    )
