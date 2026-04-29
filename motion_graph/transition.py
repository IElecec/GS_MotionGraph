import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from similarity.rotation import estimate_sequence_rotation, rotate_points
from utils import Database, GaussianModel, load_gaussians

FrameKey = Tuple[str, str, int]


@dataclass(frozen=True)
class FrameRef:
    action: str
    animation: str
    frame: int

    def key(self) -> FrameKey:
        return (self.action, self.animation, self.frame)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Transition:
    source: FrameRef
    target: FrameRef
    distance: float
    theta: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphEdge:
    source: FrameRef
    target: FrameRef
    kind: str
    distance: float = 0.0
    theta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionFrame:
    source_frame: int
    target_frame: int
    alpha: float
    gaussian: GaussianModel

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "alpha": self.alpha,
        }


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _is_local_minimum(matrix: Any, row: int, col: int) -> bool:
    center = _scalar(matrix[row, col])
    row_start = max(0, row - 1)
    row_end = min(matrix.shape[0], row + 2)
    col_start = max(0, col - 1)
    col_end = min(matrix.shape[1], col + 2)

    for neighbor_row in range(row_start, row_end):
        for neighbor_col in range(col_start, col_end):
            if neighbor_row == row and neighbor_col == col:
                continue

            neighbor = _scalar(matrix[neighbor_row, neighbor_col])
            if math.isfinite(neighbor) and neighbor < center:
                return False
    return True


def _matrix_index_to_frame(index: int, mode: str, window_size: int) -> int:
    if mode == "window_start":
        return index
    if mode == "window_end":
        return index - window_size + 1
    raise ValueError(f"Unsupported matrix index mode: {mode}")


def _shared_point_count(src: GaussianModel, dst: GaussianModel) -> int:
    return min(src.get_xyz.shape[0], dst.get_xyz.shape[0])


def _slice_points(tensor: torch.Tensor, count: int) -> torch.Tensor:
    return tensor[:count]


def transition_from_dict(data: Dict[str, Any]) -> Transition:
    return Transition(
        source=FrameRef(**data["source"]),
        target=FrameRef(**data["target"]),
        distance=float(data["distance"]),
        theta=float(data.get("theta", 0.0)),
    )


def slerp_quaternion(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    close = dot > 0.9995
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    s0 = torch.sin(theta_0 - theta_t) / torch.clamp(sin_theta_0, min=1e-8)
    s1 = sin_theta_t / torch.clamp(sin_theta_0, min=1e-8)

    out_slerp = s0 * q0 + s1 * q1
    out_lerp = F.normalize((1.0 - t) * q0 + t * q1, dim=-1)
    out = torch.where(close, out_lerp, out_slerp)
    return F.normalize(out, dim=-1)


def synthesize_transition_gaussian(
    src: GaussianModel,
    dst: GaussianModel,
    alpha: float,
    theta: float = 0.0,
) -> GaussianModel:
    point_count = _shared_point_count(src, dst)
    if point_count == 0:
        raise ValueError("Cannot synthesize transition from empty Gaussian models.")

    out = copy.deepcopy(src)

    src_xyz = _slice_points(src.get_xyz, point_count)
    dst_xyz = _slice_points(dst.get_xyz, point_count)

    src_center = src_xyz.mean(dim=0, keepdim=True)
    dst_center = dst_xyz.mean(dim=0, keepdim=True)

    src_xyz_centered = src_xyz - src_center
    dst_xyz_centered = dst_xyz - dst_center

    if theta != 0.0:
        src_xyz_centered = rotate_points(src_xyz_centered, -(1.0 - alpha) * theta)
        dst_xyz_centered = rotate_points(dst_xyz_centered, alpha * theta)

    mixed_xyz = alpha * src_xyz_centered + (1.0 - alpha) * dst_xyz_centered
    mixed_center = alpha * src_center + (1.0 - alpha) * dst_center

    src_opacity = _slice_points(src._opacity, point_count)
    dst_opacity = _slice_points(dst._opacity, point_count)
    src_scaling = _slice_points(src._scaling, point_count)
    dst_scaling = _slice_points(dst._scaling, point_count)
    src_features_dc = _slice_points(src._features_dc, point_count)
    dst_features_dc = _slice_points(dst._features_dc, point_count)
    src_features_rest = _slice_points(src._features_rest, point_count)
    dst_features_rest = _slice_points(dst._features_rest, point_count)

    q_src = _slice_points(src.get_rotation, point_count)
    q_dst = _slice_points(dst.get_rotation, point_count)
    q_mix = slerp_quaternion(q_src, q_dst, 1.0 - alpha)

    out._xyz = mixed_xyz + mixed_center
    out._opacity = alpha * src_opacity + (1.0 - alpha) * dst_opacity
    out._scaling = alpha * src_scaling + (1.0 - alpha) * dst_scaling
    out._features_dc = alpha * src_features_dc + (1.0 - alpha) * dst_features_dc
    out._features_rest = alpha * src_features_rest + (1.0 - alpha) * dst_features_rest
    out._rotation = q_mix

    return out


def build_transition_window(
    source_gaussians: List[GaussianModel],
    target_gaussians: List[GaussianModel],
    start_frame: int,
    target_frame: int,
    num_transition_frames: int,
    theta: Optional[float] = None,
) -> Dict[str, Any]:
    if num_transition_frames < 0:
        raise ValueError("num_transition_frames must be non-negative")

    transition_frames: List[TransitionFrame] = []
    sequence_length = num_transition_frames + 2
    target_start = target_frame

    if start_frame < 0 or start_frame + sequence_length > len(source_gaussians):
        raise ValueError("Source transition sequence exceeds frame range.")
    if target_start < 0 or target_start + sequence_length > len(target_gaussians):
        raise ValueError("Target transition sequence exceeds frame range.")

    if theta is None:
        theta = estimate_sequence_rotation(
            source_gaussians[start_frame : start_frame + sequence_length],
            target_gaussians[target_start : target_start + sequence_length],
        )

    for local_idx in range(num_transition_frames):
        sequence_offset = local_idx + 1
        src_idx = start_frame + sequence_offset
        dst_idx = target_start + sequence_offset
        u = (local_idx + 1) / (num_transition_frames + 1)
        alpha = 1.0 - (3.0 * u * u - 2.0 * u * u * u)
        synthesized = synthesize_transition_gaussian(
            source_gaussians[src_idx],
            target_gaussians[dst_idx],
            alpha,
            theta,
        )
        transition_frames.append(
            TransitionFrame(
                source_frame=src_idx,
                target_frame=dst_idx,
                alpha=float(alpha),
                gaussian=synthesized,
            )
        )

    return {
        "source_frame": start_frame,
        "target_frame": target_frame,
        "num_transition_frames": num_transition_frames,
        "theta": float(theta),
        "frames": transition_frames,
    }


def build_transition_window_from_database(
    database: Database,
    transition: Transition,
    sh_degree: int,
    num_transition_frames: int,
    gaussian_cache: Optional[Dict[Tuple[str, str, int], List[GaussianModel]]] = None,
) -> Dict[str, Any]:
    if gaussian_cache is None:
        gaussian_cache = {}

    source_key = (
        transition.source.action,
        transition.source.animation,
        sh_degree,
    )
    target_key = (
        transition.target.action,
        transition.target.animation,
        sh_degree,
    )

    if source_key not in gaussian_cache:
        gaussian_cache[source_key] = load_gaussians(
            database.get_frames(transition.source.action, transition.source.animation),
            sh_degree=sh_degree,
        )
    if target_key not in gaussian_cache:
        gaussian_cache[target_key] = load_gaussians(
            database.get_frames(transition.target.action, transition.target.animation),
            sh_degree=sh_degree,
        )

    window = build_transition_window(
        source_gaussians=gaussian_cache[source_key],
        target_gaussians=gaussian_cache[target_key],
        start_frame=transition.source.frame,
        target_frame=transition.target.frame,
        num_transition_frames=num_transition_frames,
        theta=transition.theta,
    )
    window["source"] = transition.source.to_dict()
    window["target"] = transition.target.to_dict()
    window["distance"] = transition.distance
    return window


def save_transition_window(window: Dict[str, Any], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": window.get("source"),
        "target": window.get("target"),
        "distance": window.get("distance"),
        "source_frame": window["source_frame"],
        "target_frame": window["target_frame"],
        "num_transition_frames": window["num_transition_frames"],
        "theta": window["theta"],
        "frames": [frame.to_dict() for frame in window["frames"]],
    }

    with (output_dir / "transition.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    saved_paths: List[Path] = []
    for local_idx, frame in enumerate(window["frames"]):
        frame_path = output_dir / f"frame_{local_idx}.ply"
        frame.gaussian.save_ply(str(frame_path))
        saved_paths.append(frame_path)

    return saved_paths


def build_transitions_from_matrices(
    source_action: str,
    source_animation: str,
    target_action: str,
    target_animation: str,
    distance_matrix: Any,
    angle_matrix: Any,
    distance_threshold: float,
    top_k: int,
    window_size: int,
    source_index_mode: str,
    target_index_mode: str,
    source_frame_limit: int,
    target_frame_limit: int,
) -> List[Transition]:
    transitions: List[Transition] = []
    for source_index in range(distance_matrix.shape[0]):
        for target_index in range(distance_matrix.shape[1]):
            distance = _scalar(distance_matrix[source_index, target_index])
            if not math.isfinite(distance) or distance > distance_threshold:
                continue
            if not _is_local_minimum(distance_matrix, source_index, target_index):
                continue

            source_frame = _matrix_index_to_frame(
                source_index,
                source_index_mode,
                window_size,
            )
            target_frame = _matrix_index_to_frame(
                target_index,
                target_index_mode,
                window_size,
            )
            if source_frame < 0 or source_frame >= source_frame_limit:
                continue
            if target_frame < 0 or target_frame >= target_frame_limit:
                continue

            theta = _scalar(angle_matrix[source_index, target_index])
            if not math.isfinite(theta):
                theta = 0.0

            transitions.append(
                Transition(
                    source=FrameRef(
                        action=source_action,
                        animation=source_animation,
                        frame=source_frame,
                    ),
                    target=FrameRef(
                        action=target_action,
                        animation=target_animation,
                        frame=target_frame,
                    ),
                    distance=distance,
                    theta=theta,
                )
            )

    transitions.sort(key=lambda transition: transition.distance)
    if top_k > 0:
        return transitions[:top_k]
    return transitions
