import heapq
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .transition import FrameRef, GraphEdge

FrameKey = Tuple[str, str, int]
LaneKey = Tuple[str, str]
PathState = Tuple[float, int, float, int]
EdgeFilter = Callable[[FrameRef, FrameRef, GraphEdge], bool]


def node_id(key: FrameKey) -> str:
    return f"{key[0]}|{key[1]}|{int(key[2])}"


def cost_function_payload(length_weight: float, distance_weight: float) -> Dict[str, Any]:
    return {
        "formula": "sum(length_weight * edge.length + distance_weight * edge.distance)",
        "length_weight": float(length_weight),
        "distance_weight": float(distance_weight),
    }


def edge_cost(
    edge: GraphEdge,
    *,
    length_weight: float = 1.0,
    distance_weight: float = 1.0,
) -> float:
    return (
        float(length_weight) * float(max(0, int(edge.length)))
        + float(distance_weight) * float(edge.distance)
    )


def better(candidate: PathState, current: Optional[PathState]) -> bool:
    if current is None:
        return True
    return candidate < current


def _reverse_edges(edges: List[GraphEdge]) -> Dict[FrameKey, List[Tuple[FrameKey, GraphEdge]]]:
    reverse: Dict[FrameKey, List[Tuple[FrameKey, GraphEdge]]] = {}
    for edge in edges:
        reverse.setdefault(edge.target.key(), []).append((edge.source.key(), edge))
    return reverse


def _best_paths_to_targets(
    *,
    node_map: Dict[FrameKey, FrameRef],
    reverse: Dict[FrameKey, List[Tuple[FrameKey, GraphEdge]]],
    target_keys: List[FrameKey],
    edge_filter: Optional[EdgeFilter] = None,
    length_weight: float = 1.0,
    distance_weight: float = 1.0,
) -> Tuple[Dict[FrameKey, PathState], Dict[FrameKey, FrameKey], Dict[FrameKey, GraphEdge]]:
    best: Dict[FrameKey, PathState] = {}
    next_node: Dict[FrameKey, FrameKey] = {}
    next_edge: Dict[FrameKey, GraphEdge] = {}
    heap: List[Tuple[float, int, float, int, FrameKey]] = []

    for key in target_keys:
        best[key] = (0.0, 0, 0.0, 0)
        heapq.heappush(heap, (0.0, 0, 0.0, 0, key))

    while heap:
        state = heapq.heappop(heap)
        cost, frames, distance, transitions, current = state
        if best.get(current) != (cost, frames, distance, transitions):
            continue

        for previous, edge in reverse.get(current, []):
            previous_node = node_map[previous]
            current_node = node_map[current]
            if edge_filter is not None and not edge_filter(previous_node, current_node, edge):
                continue

            candidate = (
                cost + edge_cost(edge, length_weight=length_weight, distance_weight=distance_weight),
                frames + max(0, int(edge.length)),
                distance + float(edge.distance),
                transitions + (1 if edge.kind == "transition" else 0),
            )
            if not better(candidate, best.get(previous)):
                continue
            best[previous] = candidate
            next_node[previous] = current
            next_edge[previous] = edge
            heapq.heappush(heap, (*candidate, previous))

    return best, next_node, next_edge


def _path_record(
    *,
    source: FrameRef,
    target_keys: List[FrameKey],
    node_map: Dict[FrameKey, FrameRef],
    best: Dict[FrameKey, PathState],
    next_node: Dict[FrameKey, FrameKey],
    next_edge: Dict[FrameKey, GraphEdge],
    extra: Dict[str, Any],
    length_weight: float,
    distance_weight: float,
) -> Dict[str, Any]:
    source_key = source.key()
    record: Dict[str, Any] = {
        "source": source.to_dict(),
        "source_id": node_id(source_key),
        **extra,
    }

    state = best.get(source_key)
    if state is None:
        record["reachable"] = False
        return record

    target_set = set(target_keys)
    cost, frames, distance, transitions = state
    cursor = source_key
    nodes_on_path = [source.to_dict()]
    edges_on_path: List[Dict[str, Any]] = []

    while cursor not in target_set:
        edge = next_edge[cursor]
        cursor = next_node[cursor]
        edges_on_path.append(
            {
                **edge.to_dict(),
                "cost": edge_cost(edge, length_weight=length_weight, distance_weight=distance_weight),
            }
        )
        nodes_on_path.append(node_map[cursor].to_dict())

    target = node_map[cursor]
    record.update(
        {
            "reachable": True,
            "target": target.to_dict(),
            "target_id": node_id(cursor),
            "target_action": target.action,
            "target_animation": target.animation,
            "target_frame": int(target.frame),
            "total_cost": float(cost),
            "total_frames": int(frames),
            "total_transition_distance": float(distance),
            "num_transitions": int(transitions),
            "num_edges": len(edges_on_path),
            "path": nodes_on_path,
            "edges": edges_on_path,
        }
    )
    return record


def path_to_action(
    *,
    database_dir: str,
    nodes: List[FrameRef],
    edges: List[GraphEdge],
    target_action: str,
    length_weight: float = 1.0,
    distance_weight: float = 1.0,
) -> Dict[str, Any]:
    node_map = {node.key(): node for node in nodes}
    reverse = _reverse_edges(edges)
    targets = sorted(key for key in node_map if key[0] == target_action)
    if not targets:
        raise ValueError(f"Unknown action: {target_action}")

    best, next_node, next_edge = _best_paths_to_targets(
        node_map=node_map,
        reverse=reverse,
        target_keys=targets,
        length_weight=length_weight,
        distance_weight=distance_weight,
    )

    paths: List[Dict[str, Any]] = []
    reachable = 0
    for source in sorted(nodes, key=lambda item: item.key()):
        record = _path_record(
            source=source,
            target_keys=targets,
            node_map=node_map,
            best=best,
            next_node=next_node,
            next_edge=next_edge,
            extra={"target_action": target_action},
            length_weight=length_weight,
            distance_weight=distance_weight,
        )
        reachable += int(record.get("reachable", False))
        paths.append(record)

    actions = sorted({node.action for node in nodes})
    return {
        "database_dir": database_dir,
        "target_action": target_action,
        "available_actions": actions,
        "cost_function": cost_function_payload(length_weight, distance_weight),
        "num_nodes": len(nodes),
        "num_reachable_nodes": reachable,
        "num_unreachable_nodes": len(nodes) - reachable,
        "target_nodes": [node_map[key].to_dict() for key in targets],
        "paths": paths,
    }


def path_to_sequence_start(
    *,
    database_dir: str,
    nodes: List[FrameRef],
    edges: List[GraphEdge],
    target: FrameRef,
    length_weight: float = 1.0,
    distance_weight: float = 1.0,
) -> Dict[str, Any]:
    node_map = {node.key(): node for node in nodes}
    reverse = _reverse_edges(edges)
    target_key = target.key()
    lane_nodes = sorted(
        [node for node in nodes if node.action == target.action and node.animation == target.animation],
        key=lambda item: item.key(),
    )

    best, next_node, next_edge = _best_paths_to_targets(
        node_map=node_map,
        reverse=reverse,
        target_keys=[target_key],
        length_weight=length_weight,
        distance_weight=distance_weight,
    )

    paths: List[Dict[str, Any]] = []
    reachable = 0
    for source in lane_nodes:
        record = _path_record(
            source=source,
            target_keys=[target_key],
            node_map=node_map,
            best=best,
            next_node=next_node,
            next_edge=next_edge,
            extra={
                "target_action": target.action,
                "target_animation": target.animation,
                "target_frame": int(target.frame),
            },
            length_weight=length_weight,
            distance_weight=distance_weight,
        )
        reachable += int(record.get("reachable", False))
        paths.append(record)

    return {
        "database_dir": database_dir,
        "target_action": target.action,
        "target_animation": target.animation,
        "target_frame": int(target.frame),
        "cost_function": cost_function_payload(length_weight, distance_weight),
        "num_nodes": len(lane_nodes),
        "num_reachable_nodes": reachable,
        "num_unreachable_nodes": len(lane_nodes) - reachable,
        "target_node": target.to_dict(),
        "paths": paths,
    }


def all_shortest_paths(
    *,
    database_dir: str,
    nodes: List[FrameRef],
    edges: List[GraphEdge],
    length_weight: float = 1.0,
    distance_weight: float = 1.0,
) -> Dict[str, Any]:
    actions = sorted({node.action for node in nodes})
    by_action = {
        action: path_to_action(
            database_dir=database_dir,
            nodes=nodes,
            edges=edges,
            target_action=action,
            length_weight=length_weight,
            distance_weight=distance_weight,
        )
        for action in actions
    }
    by_source = {
        action: {record["source_id"]: record for record in payload["paths"]}
        for action, payload in by_action.items()
    }

    lanes: Dict[LaneKey, FrameRef] = {}
    for node in sorted(nodes, key=lambda item: item.key()):
        lane = (node.action, node.animation)
        current = lanes.get(lane)
        if current is None or node.frame < current.frame:
            lanes[lane] = node

    by_lane = {
        lane: path_to_sequence_start(
            database_dir=database_dir,
            nodes=nodes,
            edges=edges,
            target=start_node,
            length_weight=length_weight,
            distance_weight=distance_weight,
        )
        for lane, start_node in sorted(lanes.items())
    }
    lane_by_source = {
        lane: {record["source_id"]: record for record in payload["paths"]}
        for lane, payload in by_lane.items()
    }

    source_nodes: List[Dict[str, Any]] = []
    for source in sorted(nodes, key=lambda item: item.key()):
        source_id = node_id(source.key())
        routes = []
        for action in actions:
            if action == source.action:
                continue
            route = dict(by_source[action][source_id])
            route.pop("source", None)
            route.pop("source_id", None)
            routes.append(route)

        lane_route = dict(lane_by_source[(source.action, source.animation)][source_id])
        lane_route.pop("source", None)
        lane_route.pop("source_id", None)
        source_nodes.append(
            {
                "source": source.to_dict(),
                "source_id": source_id,
                "source_action": source.action,
                "source_animation": source.animation,
                "paths_to_other_actions": routes,
                "path_to_sequence_start": lane_route,
            }
        )

    return {
        "database_dir": database_dir,
        "available_actions": actions,
        "cost_function": cost_function_payload(length_weight, distance_weight),
        "num_actions": len(actions),
        "num_nodes": len(nodes),
        "target_action_summaries": [
            {
                "target_action": action,
                "num_target_nodes": len(payload["target_nodes"]),
                "num_reachable_nodes": payload["num_reachable_nodes"],
                "num_unreachable_nodes": payload["num_unreachable_nodes"],
            }
            for action, payload in by_action.items()
        ],
        "sequence_start_summaries": [
            {
                "target_action": lane[0],
                "target_animation": lane[1],
                "target_frame": int(payload["target_frame"]),
                "num_reachable_nodes": payload["num_reachable_nodes"],
                "num_unreachable_nodes": payload["num_unreachable_nodes"],
            }
            for lane, payload in by_lane.items()
        ],
        "source_nodes": source_nodes,
    }


def save_shortest_paths(payload: Dict[str, Any], output_path: Path) -> Path:
    if output_path.suffix == "":
        output_path = output_path / "shortest_path.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
