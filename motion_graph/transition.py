import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

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
