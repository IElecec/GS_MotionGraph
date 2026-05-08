import json
import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from utils import Database

from .transition import FrameRef, GraphEdge, Transition, build_transitions_from_matrices

FrameKey = Tuple[str, str, int]
TransitionKey = Tuple[FrameKey, FrameKey]


def _resolve_similarity_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Similarity directory not found: {path}")

    nested_dir = path / "similarity_matrices"
    if nested_dir.is_dir():
        return nested_dir
    if not path.is_dir():
        raise NotADirectoryError(f"Similarity directory is not a directory: {path}")
    return path


def _load_similarity_payload(path: Path) -> Dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu")


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _infer_window_size(distance_matrix: Any) -> int:
    last_valid_row = -1
    first_valid_col = None

    for row in range(distance_matrix.shape[0]):
        row_has_valid = False
        for col in range(distance_matrix.shape[1]):
            value = _scalar(distance_matrix[row, col])
            if not math.isfinite(value):
                continue

            row_has_valid = True
            if first_valid_col is None or col < first_valid_col:
                first_valid_col = col

        if row_has_valid:
            last_valid_row = row

    if last_valid_row < 0:
        return 1

    inferred_from_rows = distance_matrix.shape[0] - last_valid_row
    inferred_from_cols = 1 if first_valid_col is None else first_valid_col + 1
    return max(1, inferred_from_rows, inferred_from_cols)


def _infer_target_index_mode(distance_matrix: Any, window_size: int) -> str:
    first_valid_col = None
    for col in range(distance_matrix.shape[1]):
        has_valid = False
        for row in range(distance_matrix.shape[0]):
            if math.isfinite(_scalar(distance_matrix[row, col])):
                has_valid = True
                break
        if has_valid:
            first_valid_col = col
            break

    if first_valid_col == window_size - 1:
        return "window_end"
    return "window_start"


def _frame_counts(database: Database) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for action in database.get_actions():
        for animation in database.get_animations(action):
            counts[(action, animation)] = len(database.get_frames(action, animation))
    return counts


def _build_transition_nodes(transitions: List[Transition]) -> List[FrameRef]:
    unique_nodes: Dict[FrameKey, FrameRef] = {}
    for transition in transitions:
        unique_nodes[transition.source.key()] = transition.source
        unique_nodes[transition.target.key()] = transition.target
    return [unique_nodes[key] for key in sorted(unique_nodes)]


def _build_sequence_edges(nodes: List[FrameRef]) -> List[GraphEdge]:
    grouped_nodes: Dict[Tuple[str, str], List[FrameRef]] = {}
    for node in nodes:
        grouped_nodes.setdefault((node.action, node.animation), []).append(node)

    edges: List[GraphEdge] = []
    for refs in grouped_nodes.values():
        refs.sort(key=lambda frame_ref: frame_ref.frame)
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


def _node_id_from_key(key: FrameKey) -> str:
    return f"{key[0]}|{key[1]}|{int(key[2])}"


def _slugify_token(value: str) -> str:
    slug = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in value
    ).strip("_")
    return slug or "value"


def _transition_edge_length_lookup(edges: List[GraphEdge]) -> Dict[TransitionKey, int]:
    lookup: Dict[TransitionKey, int] = {}
    for edge in edges:
        if edge.kind != "transition":
            continue
        lookup[_transition_key(edge.source, edge.target)] = int(edge.length)
    return lookup


def _build_transition_edges(
    transitions: List[Transition],
    default_length: int,
    length_lookup: Optional[Dict[TransitionKey, int]] = None,
) -> List[GraphEdge]:
    edges: List[GraphEdge] = []
    for transition in transitions:
        length = default_length
        if length_lookup is not None:
            length = int(
                length_lookup.get(
                    _transition_key(transition.source, transition.target),
                    default_length,
                )
            )
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


def _transition_connected_keys(transitions: List[Transition]) -> Set[FrameKey]:
    keep: Set[FrameKey] = set()
    for transition in transitions:
        keep.add(transition.source.key())
        keep.add(transition.target.key())
    return keep


def _prune_nodes_without_transitions(
    nodes: List[FrameRef],
    edges: List[GraphEdge],
    transitions: List[Transition],
) -> Tuple[List[FrameRef], List[GraphEdge], List[Transition]]:
    keep = _transition_connected_keys(transitions)
    nodes = [node for node in nodes if node.key() in keep]
    edges = [
        edge
        for edge in edges
        if edge.source.key() in keep and edge.target.key() in keep
    ]
    return nodes, edges, transitions


def _rebuild_graph_from_transitions(
    transitions: List[Transition],
    default_transition_length: int,
    transition_length_lookup: Optional[Dict[TransitionKey, int]] = None,
) -> Tuple[List[FrameRef], List[GraphEdge], List[Transition]]:
    nodes = _build_transition_nodes(transitions)
    edges = _build_sequence_edges(nodes) + _build_transition_edges(
        transitions,
        default_length=default_transition_length,
        length_lookup=transition_length_lookup,
    )
    return _prune_nodes_without_transitions(nodes, edges, transitions)


def _dfs(node: FrameKey, graph: Dict[FrameKey, List[FrameKey]], visited: Set[FrameKey], order: List[FrameKey]) -> None:
    visited.add(node)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            _dfs(neighbor, graph, visited, order)
    order.append(node)


def _collect_component(node: FrameKey, graph: Dict[FrameKey, List[FrameKey]], visited: Set[FrameKey], component: List[FrameKey]) -> None:
    visited.add(node)
    component.append(node)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            _collect_component(neighbor, graph, visited, component)


def _weighted_edge_cost(
    edge: GraphEdge,
    frame_weight: float,
    transition_distance_weight: float,
) -> float:
    return (
        float(frame_weight) * max(0, int(edge.length))
        + float(transition_distance_weight) * float(edge.distance)
    )


def _is_better_path(
    candidate: Tuple[float, int, float, int],
    current: Optional[Tuple[float, int, float, int]],
    tolerance: float = 1e-12,
) -> bool:
    if current is None:
        return True

    candidate_cost, candidate_frames, candidate_transition_distance, candidate_transition_count = candidate
    current_cost, current_frames, current_transition_distance, current_transition_count = current

    if candidate_cost < current_cost - tolerance:
        return True
    if abs(candidate_cost - current_cost) > tolerance:
        return False

    if candidate_frames < current_frames:
        return True
    if candidate_frames != current_frames:
        return False

    if candidate_transition_distance < current_transition_distance - tolerance:
        return True
    if abs(candidate_transition_distance - current_transition_distance) > tolerance:
        return False

    return candidate_transition_count < current_transition_count


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
        database: Database,
        similarity_dir: Path,
        distance_threshold: float,
        top_k: int,
        prune_dead_ends: bool = True,
    ) -> "MotionGraph":
        similarity_root = _resolve_similarity_dir(similarity_dir)
        transitions: List[Transition] = []
        transition_edges: List[GraphEdge] = []
        frame_counts = _frame_counts(database)
        found_similarity_file = False
        observed_window_sizes: Set[int] = set()

        for similarity_file in sorted(similarity_root.rglob("similarity.pt")):
            found_similarity_file = True
            relative_parts = similarity_file.relative_to(similarity_root).parts
            if len(relative_parts) != 5:
                continue

            src_action, src_anim, dst_action, dst_anim, _ = relative_parts
            payload = _load_similarity_payload(similarity_file)
            distance_matrix = payload["distance_matrix"]
            angle_matrix = payload["angle_matrix"]
            window_size = int(payload.get("window_size", _infer_window_size(distance_matrix)))
            observed_window_sizes.add(window_size)
            source_index_mode = payload.get("source_index_mode", "window_start")
            target_index_mode = payload.get(
                "target_index_mode",
                _infer_target_index_mode(distance_matrix, window_size),
            )
            source_frame_limit = frame_counts.get((src_action, src_anim), distance_matrix.shape[0])
            target_frame_limit = frame_counts.get((dst_action, dst_anim), angle_matrix.shape[1])

            pair_transitions = build_transitions_from_matrices(
                source_action=src_action,
                source_animation=src_anim,
                target_action=dst_action,
                target_animation=dst_anim,
                distance_matrix=distance_matrix,
                angle_matrix=angle_matrix,
                distance_threshold=distance_threshold,
                top_k=top_k,
                window_size=window_size,
                source_index_mode=source_index_mode,
                target_index_mode=target_index_mode,
                source_frame_limit=source_frame_limit,
                target_frame_limit=target_frame_limit,
            )
            transitions.extend(pair_transitions)

            for transition in pair_transitions:
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

        if not found_similarity_file:
            raise FileNotFoundError(
                f"No similarity.pt files found under {similarity_root}"
            )

        transition_length_lookup = _transition_edge_length_lookup(transition_edges)
        default_transition_length = 0
        if observed_window_sizes:
            default_transition_length = max(0, min(observed_window_sizes) - 1)
        nodes, edges, transitions = _rebuild_graph_from_transitions(
            transitions,
            default_transition_length=default_transition_length,
            transition_length_lookup=transition_length_lookup,
        )

        graph = cls(
            database=database,
            nodes=nodes,
            edges=edges,
            transitions=transitions,
            window_size=next(iter(observed_window_sizes)) if len(observed_window_sizes) == 1 else None,
        )
        if prune_dead_ends:
            return graph.largest_strongly_connected_component()
        return graph

    def largest_strongly_connected_component(self) -> "MotionGraph":
        adjacency: Dict[FrameKey, List[FrameKey]] = {
            node.key(): [] for node in self.nodes
        }
        reverse_adjacency: Dict[FrameKey, List[FrameKey]] = {
            node.key(): [] for node in self.nodes
        }

        for edge in self.edges:
            source_key = edge.source.key()
            target_key = edge.target.key()
            adjacency.setdefault(source_key, []).append(target_key)
            reverse_adjacency.setdefault(target_key, []).append(source_key)

        visited: Set[FrameKey] = set()
        order: List[FrameKey] = []
        for node_key in adjacency:
            if node_key not in visited:
                _dfs(node_key, adjacency, visited, order)

        visited.clear()
        largest_component: List[FrameKey] = []
        for node_key in reversed(order):
            if node_key in visited:
                continue
            component: List[FrameKey] = []
            _collect_component(node_key, reverse_adjacency, visited, component)
            if len(component) > len(largest_component):
                largest_component = component

        keep = set(largest_component)
        nodes = [node for node in self.nodes if node.key() in keep]
        edges = [
            edge
            for edge in self.edges
            if edge.source.key() in keep and edge.target.key() in keep
        ]
        transitions = [
            transition
            for transition in self.transitions
            if transition.source.key() in keep and transition.target.key() in keep
        ]
        transition_length_lookup = _transition_edge_length_lookup(edges)
        default_transition_length = 0
        if self.window_size is not None:
            default_transition_length = max(0, int(self.window_size) - 1)
        nodes, edges, transitions = _rebuild_graph_from_transitions(
            transitions,
            default_transition_length=default_transition_length,
            transition_length_lookup=transition_length_lookup,
        )
        return MotionGraph(
            database=self.database,
            nodes=nodes,
            edges=edges,
            transitions=transitions,
            window_size=self.window_size,
        )

    def shortest_paths_to_action(
        self,
        target_action: str,
        frame_weight: float = 1.0,
        transition_distance_weight: float = 1.0,
    ) -> Dict[str, Any]:
        if frame_weight < 0.0:
            raise ValueError("frame_weight must be non-negative.")
        if transition_distance_weight < 0.0:
            raise ValueError("transition_distance_weight must be non-negative.")
        if frame_weight == 0.0 and transition_distance_weight == 0.0:
            raise ValueError("At least one path-cost weight must be positive.")

        node_lookup: Dict[FrameKey, FrameRef] = {
            node.key(): node for node in self.nodes
        }
        if not node_lookup:
            raise ValueError("Motion graph has no nodes.")

        available_actions = sorted({node.action for node in self.nodes})
        target_keys = sorted(
            key for key in node_lookup
            if key[0] == target_action
        )
        if not target_keys:
            raise ValueError(
                f"Target action {target_action!r} does not exist in the motion graph. "
                f"Available actions: {available_actions}"
            )

        reverse_adjacency: Dict[FrameKey, List[Tuple[FrameKey, GraphEdge]]] = {}
        for edge in self.edges:
            reverse_adjacency.setdefault(edge.target.key(), []).append(
                (edge.source.key(), edge)
            )

        best_state: Dict[FrameKey, Tuple[float, int, float, int]] = {}
        next_node: Dict[FrameKey, FrameKey] = {}
        next_edge: Dict[FrameKey, GraphEdge] = {}
        heap: List[Tuple[float, int, float, int, FrameKey]] = []

        for key in target_keys:
            best_state[key] = (0.0, 0, 0.0, 0)
            heapq.heappush(heap, (0.0, 0, 0.0, 0, key))

        while heap:
            cost, frames, transition_distance, transition_count, node_key = heapq.heappop(heap)
            current_state = best_state.get(node_key)
            if current_state is None:
                continue

            if (
                abs(cost - current_state[0]) > 1e-12
                or frames != current_state[1]
                or abs(transition_distance - current_state[2]) > 1e-12
                or transition_count != current_state[3]
            ):
                continue

            for predecessor_key, edge in reverse_adjacency.get(node_key, []):
                edge_cost = _weighted_edge_cost(
                    edge,
                    frame_weight=frame_weight,
                    transition_distance_weight=transition_distance_weight,
                )
                candidate_state = (
                    cost + edge_cost,
                    frames + max(0, int(edge.length)),
                    transition_distance + float(edge.distance),
                    transition_count + (1 if edge.kind == "transition" else 0),
                )
                if not _is_better_path(
                    candidate_state,
                    best_state.get(predecessor_key),
                ):
                    continue

                best_state[predecessor_key] = candidate_state
                next_node[predecessor_key] = node_key
                next_edge[predecessor_key] = edge
                heapq.heappush(
                    heap,
                    (
                        candidate_state[0],
                        candidate_state[1],
                        candidate_state[2],
                        candidate_state[3],
                        predecessor_key,
                    ),
                )

        target_key_set = set(target_keys)
        paths: List[Dict[str, Any]] = []
        reachable_count = 0

        for source_key in sorted(node_lookup):
            source_ref = node_lookup[source_key]
            path_record: Dict[str, Any] = {
                "source": source_ref.to_dict(),
                "source_id": _node_id_from_key(source_key),
                "target_action": target_action,
            }
            state = best_state.get(source_key)
            if state is None:
                path_record["reachable"] = False
                paths.append(path_record)
                continue

            reachable_count += 1
            total_cost, total_frames, total_transition_distance, total_transition_count = state
            cursor = source_key
            path_nodes = [source_ref.to_dict()]
            path_edges: List[Dict[str, Any]] = []

            for _ in range(len(self.nodes) + 1):
                if cursor in target_key_set:
                    break

                edge = next_edge.get(cursor)
                successor_key = next_node.get(cursor)
                if edge is None or successor_key is None:
                    raise RuntimeError(
                        f"Shortest-path reconstruction failed for node {source_key}."
                    )

                path_edges.append(
                    {
                        **edge.to_dict(),
                        "cost": _weighted_edge_cost(
                            edge,
                            frame_weight=frame_weight,
                            transition_distance_weight=transition_distance_weight,
                        ),
                    }
                )
                path_nodes.append(node_lookup[successor_key].to_dict())
                cursor = successor_key
            else:
                raise RuntimeError(
                    f"Shortest-path reconstruction exceeded graph size for node {source_key}."
                )

            path_record.update(
                {
                    "reachable": True,
                    "target": node_lookup[cursor].to_dict(),
                    "target_id": _node_id_from_key(cursor),
                    "total_cost": float(total_cost),
                    "total_frames": int(total_frames),
                    "total_transition_distance": float(total_transition_distance),
                    "num_transitions": int(total_transition_count),
                    "num_edges": len(path_edges),
                    "path": path_nodes,
                    "edges": path_edges,
                }
            )
            paths.append(path_record)

        return {
            "database_dir": str(self.database.base_dir),
            "target_action": target_action,
            "available_actions": available_actions,
            "cost_function": {
                "formula": (
                    "sum(frame_weight * edge.length + "
                    "transition_distance_weight * edge.distance)"
                ),
                "frame_weight": float(frame_weight),
                "transition_distance_weight": float(transition_distance_weight),
            },
            "num_nodes": len(self.nodes),
            "num_reachable_nodes": reachable_count,
            "num_unreachable_nodes": len(self.nodes) - reachable_count,
            "target_nodes": [node_lookup[key].to_dict() for key in target_keys],
            "paths": paths,
        }

    def save_shortest_paths_to_action(
        self,
        output_path: Path,
        target_action: str,
        frame_weight: float = 1.0,
        transition_distance_weight: float = 1.0,
    ) -> Path:
        if output_path.suffix == "":
            output_path = output_path / f"shortest_paths_to_{_slugify_token(target_action)}.json"

        payload = self.shortest_paths_to_action(
            target_action=target_action,
            frame_weight=frame_weight,
            transition_distance_weight=transition_distance_weight,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_path

    def shortest_paths_to_all_other_actions(
        self,
        frame_weight: float = 1.0,
        transition_distance_weight: float = 1.0,
    ) -> Dict[str, Any]:
        available_actions = sorted({node.action for node in self.nodes})
        node_lookup: Dict[FrameKey, FrameRef] = {
            node.key(): node for node in self.nodes
        }

        if not node_lookup:
            raise ValueError("Motion graph has no nodes.")

        payloads_by_target_action = {
            action: self.shortest_paths_to_action(
                target_action=action,
                frame_weight=frame_weight,
                transition_distance_weight=transition_distance_weight,
            )
            for action in available_actions
        }
        records_by_target_action = {
            action: {
                record["source_id"]: record
                for record in payload["paths"]
            }
            for action, payload in payloads_by_target_action.items()
        }

        source_nodes: List[Dict[str, Any]] = []
        for source_key in sorted(node_lookup):
            source_ref = node_lookup[source_key]
            source_id = _node_id_from_key(source_key)
            paths_to_other_actions: List[Dict[str, Any]] = []

            for target_action in available_actions:
                if target_action == source_ref.action:
                    continue

                target_record = dict(records_by_target_action[target_action][source_id])
                target_record.pop("source", None)
                target_record.pop("source_id", None)
                paths_to_other_actions.append(target_record)

            source_nodes.append(
                {
                    "source": source_ref.to_dict(),
                    "source_id": source_id,
                    "source_action": source_ref.action,
                    "paths_to_other_actions": paths_to_other_actions,
                }
            )

        target_action_summaries = [
            {
                "target_action": action,
                "num_target_nodes": len(payload["target_nodes"]),
                "num_reachable_nodes": payload["num_reachable_nodes"],
                "num_unreachable_nodes": payload["num_unreachable_nodes"],
            }
            for action, payload in payloads_by_target_action.items()
        ]

        return {
            "database_dir": str(self.database.base_dir),
            "available_actions": available_actions,
            "cost_function": {
                "formula": (
                    "sum(frame_weight * edge.length + "
                    "transition_distance_weight * edge.distance)"
                ),
                "frame_weight": float(frame_weight),
                "transition_distance_weight": float(transition_distance_weight),
            },
            "num_actions": len(available_actions),
            "num_nodes": len(self.nodes),
            "target_action_summaries": target_action_summaries,
            "source_nodes": source_nodes,
        }

    def save_shortest_paths_to_all_other_actions(
        self,
        output_path: Path,
        frame_weight: float = 1.0,
        transition_distance_weight: float = 1.0,
    ) -> Path:
        if output_path.suffix == "":
            output_path = output_path / "shortest_path.json"

        payload = self.shortest_paths_to_all_other_actions(
            frame_weight=frame_weight,
            transition_distance_weight=transition_distance_weight,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_path

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
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return output_path
