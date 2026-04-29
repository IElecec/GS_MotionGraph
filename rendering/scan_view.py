import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .camera import build_mini_cam, build_orbit_camera, parse_vector3
from .frame_cache import GaussianFrameCache
from .gaussian_renderer import render_official_gaussian_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a multi-view scan from a single Gaussian frame."
    )
    parser.add_argument("--frame-path", required=True, help="Direct path to a Gaussian PLY frame.")
    parser.add_argument("--output-dir", required=True, help="Directory for the rendered scan images.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fov-deg", type=float, default=50.0)
    parser.add_argument("--scan-start-azimuth", type=float, default=0.0)
    parser.add_argument("--scan-end-azimuth", type=float, default=330.0)
    parser.add_argument("--scan-step-azimuth", type=float, default=30.0)
    parser.add_argument("--scan-elevation-deg", type=float, default=10.0)
    parser.add_argument("--scan-distance-scale", type=float, default=3.0)
    parser.add_argument("--background", default="1,1,1", help="Background RGB in [0,1], e.g. 1,1,1")
    parser.add_argument("--sh-degree", type=int, default=3, help="Gaussian SH degree used when loading PLY")
    return parser


def parse_background_tensor(background_text: str, device: torch.device) -> torch.Tensor:
    background_np = parse_vector3(background_text)
    if background_np is None:
        background_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return torch.tensor(
        np.clip(background_np, 0.0, 1.0).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )


def build_azimuths(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("--scan-step-azimuth must be > 0")

    azimuths: List[float] = []
    current = float(start)
    end = float(end)
    while current <= end + 1e-6:
        azimuths.append(current)
        current += float(step)
    return azimuths


def main() -> None:
    args = build_parser().parse_args()

    source_frame = Path(args.frame_path)
    if not source_frame.exists():
        raise FileNotFoundError(f"Missing frame file: {source_frame}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("diff_gaussian_rasterization requires CUDA, but no CUDA device is available.")

    cache = GaussianFrameCache(max_size=1, sh_degree=args.sh_degree)
    gaussian = cache.get(source_frame)
    xyz = gaussian.get_xyz.detach().cpu().numpy()
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = ((mins + maxs) * 0.5).astype(np.float32)
    extent = float(max(np.max(maxs - mins), 1e-3))
    background_torch = parse_background_tensor(args.background, device=device)
    azimuths = build_azimuths(
        start=args.scan_start_azimuth,
        end=args.scan_end_azimuth,
        step=args.scan_step_azimuth,
    )

    for azimuth_deg in tqdm(azimuths, desc="Rendering scan view", unit="view"):
        camera = build_orbit_camera(
            center=center,
            extent=extent,
            width=args.width,
            height=args.height,
            fov_deg=args.fov_deg,
            azimuth_deg=azimuth_deg,
            elevation_deg=args.scan_elevation_deg,
            distance_scale=args.scan_distance_scale,
        )
        mini_cam = build_mini_cam(camera, device=device)
        rendered = render_official_gaussian_frame(
            gaussian=gaussian,
            mini_cam=mini_cam,
            background=background_torch,
        )
        frame_rgb = (np.clip(rendered, 0.0, 1.0) * 255.0).astype(np.uint8)
        label = f"az={azimuth_deg:.0f}"
        cv2.putText(
            frame_rgb,
            label,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        output_path = output_dir / f"view_az_{int(round(azimuth_deg)):03d}.png"
        if not cv2.imwrite(str(output_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to save scan image: {output_path}")

    print(f"Saved scan view images to: {output_dir}")
