from .utils_gaussians import GaussianModel
from pathlib import Path
from typing import List


def load_gaussians(
    frame_paths: List[Path],
    sh_degree: int,
) -> List[GaussianModel]:
    gaussians = []

    for frame_path in frame_paths:
        frame_path = Path(frame_path)

        # if not frame_path.exists():
        #     raise FileNotFoundError(f"Gaussian ply not found: {frame_path}")

        gaussian = GaussianModel(sh_degree)
        gaussian.load_ply(str(frame_path))

        gaussians.append(gaussian)

    return gaussians