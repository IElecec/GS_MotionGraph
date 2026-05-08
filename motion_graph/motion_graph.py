import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from utils import Database

from .paths import all_shortest_paths, path_to_action, save_shortest_paths
from .transition import FrameRef, GraphEdge, Transition, build_transitions_from_matrices

FrameKey = Tuple[str, str, int]
TransitionKey = Tuple[FrameKey, FrameKey]


def _similarity_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Similarity directory not found: {path}")
    nested = path / "similarity_matrices"
    if nested.is_dir():
        return nested
    if not path.is_dir():
        raise NotADirectoryError(f"Similarity directory is not a directory: {path}")
    return path


def _load_similarity(path: Path) -> Dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu")


def _scalar(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def _window_size(distance_matrix: Any) -> int:
    last_valid_row = -1
    first_valid_col = None

    for row in range(distance_matrix.shape[0]):
        row_valid = False
        for col in range(distance_matrix.shape[1]):
            if not math.isfinite(_scalar(distance_matrix[row, col])):
                continue
            row_valid = True
            if first_valid_col is None or col < first_valid_col:
                first_valid_col = col
        if row_valid:
            last_valid_row = row

    if last_valid_row < 0:
        return 1

    rows = distance_matrix.shape[0] - last_valid_row
    cols = 1 if first_valid_col is None else first_valid_col + 1
    return max(1, rows, cols)


def _target_mode(distance_matrix: Any, window_size: int) -> str:
    for col in range(distance_matrix.shape[1]):
        for row in range(distance_matrix.shape[0]):
            if math.isfinite(_scalar(distance_matrix[row, col])):
                return "window_end" if col == window_size - 1 else "window_start"
    return "window_start"


def _frame_counts(database: Database) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for action in database.get_actions():
        for animation in database.get_animations(action):
            counts[(action, animation)] = len(database.get_frames(action, animation))
    return counts


def _transition_nodes(transitions: List[Transition]) -> List[FrameRef]:
    unique: Dict[FrameKey, FrameRef] = {}
    for transition in transitions:
        unique[transition.source.key()] = transition.source
        unique[transition.target.key()] = transition.target
    return [unique[key] for key in sorted(unique)]


def _sequence_edges(nodes: List[FrameRef]) -> List[GraphEdge]:
    lanes: Dict[Tuple[str, str], List[FrameRef]] = {}
    for node in nodes:
        lanes.setdefault((node.action, node.animation), []).append(node)

    edges: List[GraphEdge] = []
    for refs in lanes.values():
        refs.sort(key=lambda item: item.frame)
        for source, target in zip(refs, refs[1:]):
            if source.frame == target.frame:
                continue
            edges.append(
                GraphEdge(
                    source=source,
                    target=target,
                    kind="sequence",
                    length=target.frame - source.frame,
                )
            )
    return edges


def _transition_key(source: FrameRef, target: FrameRef) -> TransitionKey:
    return (source.key(), target.key())


def _transition_length_map(edges: List[GraphEdge]) -> Dict[TransitionKey, int]:
    lookup: Dict[TransitionKey, int] = {}
    for edge in edges:
        if edge.kind == "transition":
            lookup[_transition_key(edge.source, edge.target)] = int(edge.length)
    return lookup


def _transition_edges(
    transitions: List[Transition],
    default_length: int,
    length_map: Optional[Dict[TransitionKey, int]] = None,
) -> List[GraphEdge]:
    edges: List[GraphEdge] = []
    for transition in transitions:
        length = default_length
        if length_map is not None:
            length = int(length_map.get(_transition_key(transition.source, transition.target), default_length))
        edges.append(
            GraphEdge(
                source=transition.source,
                target=transition.target,
                kind="transition",
                length=length,
                distance=transition.distance,
                theta=transition.theta,
            )
        )
    return edges


def _rebuild(
    transitions: List[Transition],
    default_length: int,
    length_map: Optional[Dict[TransitionKey, int]] = None,
) -> Tuple[List[FrameRef], List[GraphEdge], List[Transition]]:
    keep = {
        transition.source.key()
        for transition in transitions
    } | {
        transition.target.key()
        for transition in transitions
    }
    nodes = [node for node in _transition_nodes(transitions) if node.key() in keep]
    edges = _sequence_edges(nodes) + _transition_edges(
        transitions,
        default_length=default_length,
        length_map=length_map,
    )
    edges = [
        edge
        for edge in edges
        if edge.source.key() in keep and edge.target.key() in keep
    ]
    return nodes, edges, transitions


def _finish_order(
    node: FrameKey,
    graph: Dict[FrameKey, List[FrameKey]],
    seen: Set[FrameKey],
    order: List[FrameKey],
) -> None:
    seen.add(node)
    for neighbor in graph.get(node, []):
        if neighbor not in seen:
            _finish_order(neighbor, graph, seen, order)
    order.append(node)


def _collect(
    node: FrameKey,
    graph: Dict[FrameKey, List[FrameKey]],
    seen: Set[FrameKey],
    component: List[FrameKey],
) -> None:
    seen.add(node)
    component.append(node)
    for neighbor in graph.get(node, []):
        if neighbor not in seen:
            _collect(neighbor, graph, seen, component)


@dataclass
class MotionGraph:
    database: Database
    nodes: List[FrameRef] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)
    window_size: Optional[int] = None

    @classmethod
    def build(
        cls,
        *,
        database: Database,
        similarity_dir: Path,
        distance_threshold: float,
        top_k: int,
        prune_dead_ends: bool = True,
    ) -> "MotionGraph":
        root = _similarity_dir(similarity_dir)
        frame_counts = _frame_counts(database)
        transitions: List[Transition] = []
        transition_edges: List[GraphEdge] = []
        window_sizes: Set[int] = set()
        found = False

        for file in sorted(root.rglob("similarity.pt")):
            found = True
            parts = file.relative_to(root).parts
            if len(parts) != 5:
                continue

            src_action, src_anim, dst_action, dst_anim, _ = parts
            payload = _load_similarity(file)
            distance_matrix = payload["distance_matrix"]
            angle_matrix = payload["angle_matrix"]
            window_size = int(payload.get("window_size", _window_size(distance_matrix)))
            window_sizes.add(window_size)

            source_mode = payload.get("source_index_mode", "window_start")
            target_mode = payload.get("target_index_mode", _target_mode(distance_matrix, window_size))

            pair = build_transitions_from_matrices(
                source_action=src_action,
                source_animation=src_anim,
                target_action=dst_action,
                target_animation=dst_anim,
                distance_matrix=distance_matrix,
                angle_matrix=angle_matrix,
                distance_threshold=distance_threshold,
                top_k=top_k,
                window_size=window_size,
                source_index_mode=source_mode,
                target_index_mode=target_mode,
                source_frame_limit=frame_counts.get((src_action, src_anim), distance_matrix.shape[0]),
                target_frame_limit=frame_counts.get((dst_action, dst_anim), angle_matrix.shape[1]),
            )
            transitions.extend(pair)

            for transition in pair:
                transition_edges.append(
                    GraphEdge(
                        source=transition.source,
                        target=transition.target,
                        kind="transition",
                        length=max(0, window_size - 1),
                        distance=transition.distance,
                        theta=transition.theta,
                    )
                )

        if not found:
            raise FileNotFoundError(f"No similarity.pt files found under {root}")

        default_length = max(0, min(window_sizes) - 1) if window_sizes else 0
        nodes, edges, transitions = _rebuild(
            transitions,
            default_length=default_length,
            length_map=_transition_length_map(transition_edges),
        )
        graph = cls(
            database=database,
            nodes=nodes,
            edges=edges,
            transitions=transitions,
            window_size=next(iter(window_sizes)) if len(window_sizes) == 1 else None,
        )
        return graph.largest_strongly_connected_component() if prune_dead_ends else graph

    def largest_strongly_connected_component(self) -> "MotionGraph":
        graph: Dict[FrameKey, List[FrameKey]] = {node.key(): [] for node in self.nodes}
        reverse: Dict[FrameKey, List[FrameKey]] = {node.key(): [] for node in self.nodes}

        for edge in self.edges:
            source = edge.source.key()
            target = edge.target.key()
            graph.setdefault(source, []).append(target)
            reverse.setdefault(target, []).append(source)

        seen: Set[FrameKey] = set()
        order: List[FrameKey] = []
        for key in graph:
            if key not in seen:
                _finish_order(key, graph, seen, order)

        seen.clear()
        largest: List[FrameKey] = []
        for key in reversed(order):
            if key in seen:
                continue
            component: List[FrameKey] = []
            _collect(key, reverse, seen, component)
            if len(component) > len(largest):
                largest = component

        keep = set(largest)
        transitions = [
            transition
            for transition in self.transitions
            if transition.source.key() in keep and transition.target.key() in keep
        ]
        default_length = max(0, int(self.window_size) - 1) if self.window_size is not None else 0
        nodes, edges, transitions = _rebuild(
            transitions,
            default_length=default_length,
            length_map=_transition_length_map(self.edges),
        )
        return MotionGraph(
            database=self.database,
            nodes=nodes,
            edges=edges,
            transitions=transitions,
            window_size=self.window_size,
        )

    def shortest_paths(self) -> Dict[str, Any]:
        return all_shortest_paths(
            database_dir=str(self.database.base_dir),
            nodes=self.nodes,
            edges=self.edges,
        )

    def save_shortest_paths(self, output_path: Path) -> Path:
        return save_shortest_paths(self.shortest_paths(), output_path)

    def shortest_paths_to_action(self, target_action: str) -> Dict[str, Any]:
        return path_to_action(
            database_dir=str(self.database.base_dir),
            nodes=self.nodes,
            edges=self.edges,
            target_action=target_action,
        )

    def save_shortest_paths_to_action(self, output_path: Path, target_action: str) -> Path:
        if output_path.suffix == "":
            output_path = output_path / f"shortest_paths_to_{target_action}.json"
        return save_shortest_paths(self.shortest_paths_to_action(target_action), output_path)

    def shortest_paths_to_all_other_actions(self) -> Dict[str, Any]:
        return self.shortest_paths()

    def save_shortest_paths_to_all_other_actions(self, output_path: Path) -> Path:
        return self.save_shortest_paths(output_path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database_dir": str(self.database.base_dir),
            "node_mode": "transition_only",
            "sequence_edge_mode": "between_consecutive_transition_nodes",
            "source_frame_semantics": "window_start",
            "target_frame_semantics": "window_end",
            "window_size": self.window_size,
            "transition_edge_length": None if self.window_size is None else max(0, self.window_size - 1),
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_transitions": len(self.transitions),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "transitions": [transition.to_dict() for transition in self.transitions],
        }

    def save(self, output_path: Path) -> Path:
        if output_path.suffix == "":
            output_path = output_path / "motion_graph.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path
