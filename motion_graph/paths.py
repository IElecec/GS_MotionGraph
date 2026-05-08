import heapq
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .transition import FrameRef, GraphEdge

FrameKey = Tuple[str, str, int]
PathState = Tuple[float, int, float, int]


def node_id(key: FrameKey) -> str:
    return f"{key[0]}|{key[1]}|{int(key[2])}"


def edge_cost(edge: GraphEdge) -> float:
    return float(max(0, int(edge.length))) + float(edge.distance)


def better(candidate: PathState, current: Optional[PathState]) -> bool:
    if current is None:
        return True
    return candidate < current


def path_to_action(
    *,
    database_dir: str,
    nodes: List[FrameRef],
    edges: List[GraphEdge],
    target_action: str,
) -> Dict[str, Any]:
    node_map = {node.key(): node for node in nodes}
    reverse: Dict[FrameKey, List[Tuple[FrameKey, GraphEdge]]] = {}

    for edge in edges:
        reverse.setdefault(edge.target.key(), []).append((edge.source.key(), edge))

    targets = sorted(key for key in node_map if key[0] == target_action)
    if not targets:
        raise ValueError(f"Unknown action: {target_action}")

    best: Dict[FrameKey, PathState] = {}
    next_node: Dict[FrameKey, FrameKey] = {}
    next_edge: Dict[FrameKey, GraphEdge] = {}
    heap: List[Tuple[float, int, float, int, FrameKey]] = []

    for key in targets:
        best[key] = (0.0, 0, 0.0, 0)
        heapq.heappush(heap, (0.0, 0, 0.0, 0, key))

    while heap:
        state = heapq.heappop(heap)
        cost, frames, distance, transitions, current = state
        if best.get(current) != (cost, frames, distance, transitions):
            continue

        for previous, edge in reverse.get(current, []):
            candidate = (
                cost + edge_cost(edge),
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

    target_set = set(targets)
    paths: List[Dict[str, Any]] = []
    reachable = 0

    for source_key in sorted(node_map):
        source = node_map[source_key]
        record: Dict[str, Any] = {
            "source": source.to_dict(),
            "source_id": node_id(source_key),
            "target_action": target_action,
        }
        state = best.get(source_key)
        if state is None:
            record["reachable"] = False
            paths.append(record)
            continue

        reachable += 1
        cost, frames, distance, transitions = state
        cursor = source_key
        nodes_on_path = [source.to_dict()]
        edges_on_path: List[Dict[str, Any]] = []

        while cursor not in target_set:
            edge = next_edge[cursor]
            cursor = next_node[cursor]
            edges_on_path.append({**edge.to_dict(), "cost": edge_cost(edge)})
            nodes_on_path.append(node_map[cursor].to_dict())

        record.update(
            {
                "reachable": True,
                "target": node_map[cursor].to_dict(),
                "target_id": node_id(cursor),
                "total_cost": float(cost),
                "total_frames": int(frames),
                "total_transition_distance": float(distance),
                "num_transitions": int(transitions),
                "num_edges": len(edges_on_path),
                "path": nodes_on_path,
                "edges": edges_on_path,
            }
        )
        paths.append(record)

    actions = sorted({node.action for node in nodes})
    return {
        "database_dir": database_dir,
        "target_action": target_action,
        "available_actions": actions,
        "cost_function": {
            "formula": "sum(edge.length + edge.distance)",
        },
        "num_nodes": len(nodes),
        "num_reachable_nodes": reachable,
        "num_unreachable_nodes": len(nodes) - reachable,
        "target_nodes": [node_map[key].to_dict() for key in targets],
        "paths": paths,
    }


def all_shortest_paths(
    *,
    database_dir: str,
    nodes: List[FrameRef],
    edges: List[GraphEdge],
) -> Dict[str, Any]:
    actions = sorted({node.action for node in nodes})
    by_action = {
        action: path_to_action(
            database_dir=database_dir,
            nodes=nodes,
            edges=edges,
            target_action=action,
        )
        for action in actions
    }
    by_source = {
        action: {
            record["source_id"]: record
            for record in payload["paths"]
        }
        for action, payload in by_action.items()
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
        source_nodes.append(
            {
                "source": source.to_dict(),
                "source_id": source_id,
                "source_action": source.action,
                "paths_to_other_actions": routes,
            }
        )

    return {
        "database_dir": database_dir,
        "available_actions": actions,
        "cost_function": {
            "formula": "sum(edge.length + edge.distance)",
        },
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
        "source_nodes": source_nodes,
    }


def save_shortest_paths(payload: Dict[str, Any], output_path: Path) -> Path:
    if output_path.suffix == "":
        output_path = output_path / "shortest_path.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    return output_path
