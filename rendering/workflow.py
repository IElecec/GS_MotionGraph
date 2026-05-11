import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .camera import build_fixed_camera, build_mini_cam, normalize, parse_vector3
from .cli import build_parser
from .frame_cache import GaussianFrameCache
from .gaussian_renderer import render_official_gaussian_frame
from .models import FixedCamera
from utils import Database
from utils.data_loader import collect_frame_files


def build_camera(args, reference_xyz: np.ndarray) -> FixedCamera:
    fixed_position = parse_vector3(args.camera_position)
    fixed_target = parse_vector3(args.camera_target)
    fixed_up = parse_vector3(args.camera_up)
    if fixed_position is not None and fixed_target is not None:
        return FixedCamera(
            width=args.width,
            height=args.height,
            position=fixed_position,
            target=fixed_target,
            up=normalize(
                fixed_up
                if fixed_up is not None
                else np.array([0.0, 1.0, 0.0], dtype=np.float32)
            ),
            fov_deg=float(args.fov_deg),
        )

    return build_fixed_camera(
        reference_xyz=reference_xyz,
        width=args.width,
        height=args.height,
        fov_deg=args.fov_deg,
        azimuth_deg=args.azimuth_deg,
        elevation_deg=args.elevation_deg,
        distance_scale=args.distance_scale,
    )


def parse_background_tensor(background_text: str, device: torch.device) -> torch.Tensor:
    background_np = parse_vector3(background_text)
    if background_np is None:
        background_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    return torch.tensor(
        np.clip(background_np, 0.0, 1.0).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )


def save_rgb_image(image_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to save rendered image: {output_path}")


def render_frame_image(
    frame_path: Path,
    output_path: Path,
    cache: GaussianFrameCache,
    mini_cam,
    background_torch: torch.Tensor,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        return

    gaussian = cache.get(frame_path)
    rendered = render_official_gaussian_frame(
        gaussian=gaussian,
        mini_cam=mini_cam,
        background=background_torch,
    )
    frame_rgb = (np.clip(rendered, 0.0, 1.0) * 255.0).astype(np.uint8)
    save_rgb_image(frame_rgb, output_path)


def find_reference_frame(database: Database) -> Tuple[str, str, Path]:
    for action in database.get_actions():
        for animation in database.get_animations(action):
            frames = database.get_render_frames(action, animation)
            if frames:
                return action, animation, frames[0]
    raise RuntimeError("Database does not contain any Gaussian PLY frames.")


def render_normal_frames(
    database: Database,
    output_dir: Path,
    cache: GaussianFrameCache,
    mini_cam,
    background_torch: torch.Tensor,
    overwrite: bool,
) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for action in database.get_actions():
        for animation in database.get_animations(action):
            frames = database.get_render_frames(action, animation)
            desc = f"Rendering {action}/{animation}"
            for frame_idx, frame_path in enumerate(tqdm(frames, desc=desc, unit="frame")):
                relative_output = Path("normal") / action / animation / f"frame_{frame_idx:04d}.png"
                render_frame_image(
                    frame_path=frame_path,
                    output_path=output_dir / relative_output,
                    cache=cache,
                    mini_cam=mini_cam,
                    background_torch=background_torch,
                    overwrite=overwrite,
                )
                manifest[f"{action}|{animation}|{frame_idx}"] = relative_output.as_posix()
    return manifest


def render_transition_frames(
    transition_dir: Optional[Path],
    output_dir: Path,
    cache: GaussianFrameCache,
    mini_cam,
    background_torch: torch.Tensor,
    overwrite: bool,
) -> Dict[str, List[str]]:
    manifest: Dict[str, List[str]] = {}
    if transition_dir is None or not transition_dir.exists():
        return manifest

    for folder in sorted(path for path in transition_dir.iterdir() if path.is_dir()):
        frame_paths = collect_frame_files(folder)
        if not frame_paths:
            continue

        rendered_paths: List[str] = []
        desc = f"Rendering transition {folder.name}"
        for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc=desc, unit="frame")):
            relative_output = Path("transitions") / folder.name / f"frame_{frame_idx:04d}.png"
            render_frame_image(
                frame_path=frame_path,
                output_path=output_dir / relative_output,
                cache=cache,
                mini_cam=mini_cam,
                background_torch=background_torch,
                overwrite=overwrite,
            )
            rendered_paths.append(relative_output.as_posix())

        manifest[folder.name] = rendered_paths
    return manifest


def write_manifest(
    output_dir: Path,
    database: Database,
    transition_dir: Optional[Path],
    camera: FixedCamera,
    normal_frames: Dict[str, str],
    transition_frames: Dict[str, List[str]],
) -> Path:
    manifest_path = output_dir / "manifest.json"
    payload = {
        "database_dir": str(database.base_dir),
        "transition_dir": None if transition_dir is None else str(transition_dir),
        "camera": {
            "width": camera.width,
            "height": camera.height,
            "fov_deg": camera.fov_deg,
            "position": [float(x) for x in camera.position.tolist()],
            "target": [float(x) for x in camera.target.tolist()],
            "up": [float(x) for x in camera.up.tolist()],
        },
        "normal_frames": normal_frames,
        "transition_frames": transition_frames,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    args = build_parser().parse_args()

    database = Database(Path(args.database))
    transition_dir = Path(args.transition_dir) if args.transition_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("diff_gaussian_rasterization requires CUDA, but no CUDA device is available.")

    _, _, reference_frame = find_reference_frame(database)
    cache = GaussianFrameCache(max_size=args.cache_size, sh_degree=args.sh_degree)
    reference_gaussian = cache.get(reference_frame)
    reference_xyz = reference_gaussian.get_xyz.detach().cpu().numpy()
    camera = build_camera(args, reference_xyz)
    mini_cam = build_mini_cam(camera, device=device)
    background_torch = parse_background_tensor(args.background, device=device)

    normal_frames = render_normal_frames(
        database=database,
        output_dir=output_dir,
        cache=cache,
        mini_cam=mini_cam,
        background_torch=background_torch,
        overwrite=args.overwrite,
    )
    transition_frames = render_transition_frames(
        transition_dir=transition_dir,
        output_dir=output_dir,
        cache=cache,
        mini_cam=mini_cam,
        background_torch=background_torch,
        overwrite=args.overwrite,
    )
    manifest_path = write_manifest(
        output_dir=output_dir,
        database=database,
        transition_dir=transition_dir,
        camera=camera,
        normal_frames=normal_frames,
        transition_frames=transition_frames,
    )

    print(f"Rendered {len(normal_frames)} normal frames to {output_dir / 'normal'}")
    if transition_frames:
        print(f"Rendered {sum(len(paths) for paths in transition_frames.values())} transition frames to {output_dir / 'transitions'}")
    else:
        print("No transition frames were rendered.")
    print(f"Saved image manifest to {manifest_path}")
