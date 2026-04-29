import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NodeKey = Tuple[str, str, int]
LaneKey = Tuple[str, str]


class UnionFind:
    def __init__(self, items: List[NodeKey]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: NodeKey) -> NodeKey:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: NodeKey, right: NodeKey) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False

        rank_left = self.rank[root_left]
        rank_right = self.rank[root_right]
        if rank_left < rank_right:
            root_left, root_right = root_right, root_left

        self.parent[root_right] = root_left
        if rank_left == rank_right:
            self.rank[root_left] += 1
        return True

    def component_count(self) -> int:
        roots = {self.find(item) for item in self.parent}
        return len(roots)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render motion_graph.json as a standalone HTML visualization."
    )
    parser.add_argument(
        "-g",
        "--motion-graph",
        required=True,
        help="path to motion_graph.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output HTML path",
    )
    parser.add_argument(
        "--frame-spacing",
        type=int,
        default=58,
        help="horizontal spacing between displayed transition nodes",
    )
    parser.add_argument(
        "--lane-spacing",
        type=int,
        default=72,
        help="vertical spacing per animation lane",
    )
    parser.add_argument(
        "--max-transition-edges",
        type=int,
        default=0,
        help="limit rendered cross-action transition edges; <= 0 renders all",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="logical playback fps used by the random walker; edge traversal time is edge.length / fps",
    )
    parser.add_argument(
        "--image-manifest",
        default=None,
        help="optional manifest.json produced by render_image_library.py; enables image playback in the HTML view",
    )
    return parser


def load_motion_graph(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def lane_key(node: Dict[str, Any]) -> LaneKey:
    return (node["action"], node["animation"])


def node_key(node: Dict[str, Any]) -> NodeKey:
    return (node["action"], node["animation"], int(node["frame"]))


def node_id(node: Dict[str, Any]) -> str:
    return f"{node['action']}|{node['animation']}|{int(node['frame'])}"


def transition_edge_length(payload: Dict[str, Any]) -> int:
    value = payload.get("transition_edge_length")
    if value is not None:
        return max(0, int(value))

    window_size = payload.get("window_size")
    if window_size is not None:
        return max(0, int(window_size) - 1)
    return 0


def normalize_motion_graph_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    transitions = list(payload.get("transitions", []))
    default_transition_edge_length = transition_edge_length(payload)

    keep_keys = set()
    for transition in transitions:
        keep_keys.add(node_key(transition["source"]))
        keep_keys.add(node_key(transition["target"]))

    if keep_keys:
        payload["nodes"] = [
            node for node in payload.get("nodes", [])
            if node_key(node) in keep_keys
        ]

    normalized_edges: List[Dict[str, Any]] = []
    for edge in payload.get("edges", []):
        source_key = node_key(edge["source"])
        target_key = node_key(edge["target"])
        if keep_keys and (source_key not in keep_keys or target_key not in keep_keys):
            continue

        edge = dict(edge)
        if "length" not in edge:
            if edge.get("kind") == "sequence":
                edge["length"] = abs(int(edge["target"]["frame"]) - int(edge["source"]["frame"]))
            elif edge.get("kind") == "transition":
                edge["length"] = int(default_transition_edge_length)
            else:
                edge["length"] = 0
        normalized_edges.append(edge)

    payload["edges"] = normalized_edges
    payload["num_nodes"] = len(payload.get("nodes", []))
    payload["num_edges"] = len(payload.get("edges", []))
    payload["num_transitions"] = len(transitions)
    return payload


def transition_sort_key(item: Dict[str, Any]) -> float:
    distance = item.get("distance", 0.0)
    try:
        return float(distance)
    except (TypeError, ValueError):
        return 0.0


def trim_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    start_padding: float,
    end_padding: float,
) -> Tuple[float, float, float, float]:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-6 or length <= start_padding + end_padding:
        return x1, y1, x2, y2

    ux = dx / length
    uy = dy / length
    return (
        x1 + ux * start_padding,
        y1 + uy * start_padding,
        x2 - ux * end_padding,
        y2 - uy * end_padding,
    )


def trim_quadratic_path(
    x1: float,
    y1: float,
    cx: float,
    cy: float,
    x2: float,
    y2: float,
    start_padding: float,
    end_padding: float,
) -> Tuple[float, float, float, float]:
    start_dx = cx - x1
    start_dy = cy - y1
    start_length = math.hypot(start_dx, start_dy)
    if start_length > 1e-6:
        x1 += start_dx / start_length * start_padding
        y1 += start_dy / start_length * start_padding

    end_dx = x2 - cx
    end_dy = y2 - cy
    end_length = math.hypot(end_dx, end_dy)
    if end_length > 1e-6:
        x2 -= end_dx / end_length * end_padding
        y2 -= end_dy / end_length * end_padding

    return x1, y1, x2, y2


def build_marker_defs() -> str:
    return """
    <defs>
      <marker id="arrow-sequence" markerWidth="11" markerHeight="11" refX="8" refY="5.5" orient="auto" markerUnits="strokeWidth">
        <path d="M 0 0 L 11 5.5 L 0 11 z" fill="#7e9fb0" />
      </marker>
      <marker id="arrow-intra" markerWidth="11" markerHeight="11" refX="8" refY="5.5" orient="auto" markerUnits="strokeWidth">
        <path d="M 0 0 L 11 5.5 L 0 11 z" fill="#d1633f" />
      </marker>
      <marker id="arrow-cross" markerWidth="11" markerHeight="11" refX="8" refY="5.5" orient="auto" markerUnits="strokeWidth">
        <path d="M 0 0 L 11 5.5 L 0 11 z" fill="#9f3d56" />
      </marker>
    </defs>
    """


def group_nodes_by_action(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for node in payload.get("nodes", []):
        grouped.setdefault(node["action"], []).append(node)
    return grouped


def get_action_sequence_edges(payload: Dict[str, Any], action: str) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    for edge in payload.get("edges", []):
        if edge.get("kind") != "sequence":
            continue
        if edge["source"]["action"] != action or edge["target"]["action"] != action:
            continue
        edges.append(edge)
    return edges


def build_sequence_edges_from_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped_by_lane: Dict[LaneKey, List[Dict[str, Any]]] = {}
    for node in nodes:
        grouped_by_lane.setdefault(lane_key(node), []).append(node)

    edges: List[Dict[str, Any]] = []
    for lane_nodes in grouped_by_lane.values():
        lane_nodes = sorted(lane_nodes, key=lambda item: int(item["frame"]))
        for source, target in zip(lane_nodes, lane_nodes[1:]):
            source_frame = int(source["frame"])
            target_frame = int(target["frame"])
            if source_frame == target_frame:
                continue
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "kind": "sequence",
                    "length": abs(target_frame - source_frame),
                    "distance": 0.0,
                    "theta": 0.0,
                }
            )
    return edges


def transition_folder_name(transition_index: int, transition: Dict[str, Any]) -> str:
    source = transition["source"]
    target = transition["target"]
    return (
        f"{transition_index:04d}_"
        f"{source['action']}_{source['animation']}_{int(source['frame']):04d}"
        f"__"
        f"{target['action']}_{target['animation']}_{int(target['frame']):04d}"
    )


def transition_to_edge_item(
    payload: Dict[str, Any],
    transition: Dict[str, Any],
    transition_index: int,
) -> Dict[str, Any]:
    return {
        "source": transition["source"],
        "target": transition["target"],
        "kind": "transition",
        "length": transition_edge_length(payload),
        "transition_index": int(transition_index),
        "transition_folder": transition_folder_name(transition_index, transition),
        "distance": float(transition.get("distance", 0.0)),
        "theta": float(transition.get("theta", 0.0)),
    }


def get_action_internal_transitions(
    payload: Dict[str, Any],
    action: str,
) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    for transition_index, transition in enumerate(payload.get("transitions", [])):
        if transition["source"]["action"] != action:
            continue
        if transition["target"]["action"] != action:
            continue
        transitions.append(transition_to_edge_item(payload, transition, transition_index))
    transitions.sort(key=transition_sort_key)
    return transitions


def get_cross_action_transitions(
    payload: Dict[str, Any],
    max_transition_edges: int,
) -> List[Dict[str, Any]]:
    transitions = [
        transition_to_edge_item(payload, transition, transition_index)
        for transition_index, transition in enumerate(payload.get("transitions", []))
        if transition["source"]["action"] != transition["target"]["action"]
    ]
    transitions.sort(key=transition_sort_key)
    if max_transition_edges > 0:
        return transitions[:max_transition_edges]
    return transitions


def build_minimal_action_graph(
    nodes: List[Dict[str, Any]],
    sequence_edges: List[Dict[str, Any]],
    internal_transitions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    keys = [node_key(node) for node in nodes]
    union_find = UnionFind(keys)

    for edge in sequence_edges:
        union_find.union(node_key(edge["source"]), node_key(edge["target"]))

    selected: List[Dict[str, Any]] = []
    for transition in internal_transitions:
        source = node_key(transition["source"])
        target = node_key(transition["target"])
        if union_find.union(source, target):
            selected.append(transition)

    return selected, union_find.component_count()


def count_components(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> int:
    if not nodes:
        return 0

    union_find = UnionFind([node_key(node) for node in nodes])
    for edge in edges:
        union_find.union(node_key(edge["source"]), node_key(edge["target"]))
    return union_find.component_count()


def build_cluster_local_layout(
    nodes: List[Dict[str, Any]],
    frame_spacing: int,
    lane_spacing: int,
) -> Tuple[Dict[NodeKey, Tuple[float, float]], List[LaneKey], int]:
    grouped_by_lane: Dict[LaneKey, List[Dict[str, Any]]] = {}
    for node in nodes:
        grouped_by_lane.setdefault(lane_key(node), []).append(node)

    lanes = sorted(grouped_by_lane)
    x_margin = 126
    y_margin = 78
    positions: Dict[NodeKey, Tuple[float, float]] = {}
    max_nodes_per_lane = 0

    for lane_idx, lane in enumerate(lanes):
        lane_nodes = sorted(grouped_by_lane[lane], key=lambda item: int(item["frame"]))
        max_nodes_per_lane = max(max_nodes_per_lane, len(lane_nodes))
        for node_idx, node in enumerate(lane_nodes):
            x = x_margin + node_idx * frame_spacing
            y = y_margin + lane_idx * lane_spacing
            positions[node_key(node)] = (x, y)

    return positions, lanes, max_nodes_per_lane


def build_action_clusters(
    payload: Dict[str, Any],
    frame_spacing: int,
    lane_spacing: int,
    max_transition_edges: int,
) -> Tuple[List[Dict[str, Any]], Dict[NodeKey, Tuple[float, float]], int, int, List[Dict[str, Any]]]:
    grouped_nodes = group_nodes_by_action(payload)
    action_names = sorted(grouped_nodes)
    cross_action_transitions = get_cross_action_transitions(payload, max_transition_edges)
    visible_transition_keys = {
        node_key(edge["source"]) for edge in cross_action_transitions
    } | {
        node_key(edge["target"]) for edge in cross_action_transitions
    }
    bridge_transitions_by_action: Dict[str, List[Dict[str, Any]]] = {}

    for action in action_names:
        nodes = grouped_nodes[action]
        sequence_edges = build_sequence_edges_from_nodes(nodes)
        internal_transitions = get_action_internal_transitions(payload, action)
        bridge_transitions, component_count = build_minimal_action_graph(
            nodes,
            sequence_edges,
            internal_transitions,
        )
        bridge_transitions_by_action[action] = bridge_transitions
        del component_count
        for edge in bridge_transitions:
            visible_transition_keys.add(node_key(edge["source"]))
            visible_transition_keys.add(node_key(edge["target"]))

    clusters: List[Dict[str, Any]] = []
    cross_action_transitions = [
        edge
        for edge in cross_action_transitions
        if node_key(edge["source"]) in visible_transition_keys
        and node_key(edge["target"]) in visible_transition_keys
    ]
    if not visible_transition_keys:
        return [], {}, 56, 56, []

    for action in action_names:
        nodes = [
            node
            for node in grouped_nodes[action]
            if node_key(node) in visible_transition_keys
        ]
        if not nodes:
            continue

        sequence_edges = build_sequence_edges_from_nodes(nodes)
        bridge_transitions = [
            edge
            for edge in bridge_transitions_by_action[action]
            if node_key(edge["source"]) in visible_transition_keys
            and node_key(edge["target"]) in visible_transition_keys
        ]
        component_count = count_components(
            nodes,
            sequence_edges + bridge_transitions,
        )
        local_positions, lanes, max_nodes_per_lane = build_cluster_local_layout(
            nodes,
            frame_spacing,
            lane_spacing,
        )
        width = 126 + max(1, max_nodes_per_lane) * frame_spacing + 86
        height = 78 + max(1, len(lanes) - 1) * lane_spacing + 78
        clusters.append(
            {
                "action": action,
                "nodes": nodes,
                "sequence_edges": sequence_edges,
                "bridge_transitions": bridge_transitions,
                "component_count": component_count,
                "lanes": lanes,
                "lane_spacing": lane_spacing,
                "width": width,
                "height": height,
                "local_positions": local_positions,
            }
        )

    column_count = 1 if len(clusters) <= 1 else 2 if len(clusters) <= 4 else 3
    row_count = (len(clusters) + column_count - 1) // column_count
    column_widths = [0] * column_count
    row_heights = [0] * row_count

    for index, cluster in enumerate(clusters):
        row_idx = index // column_count
        col_idx = index % column_count
        column_widths[col_idx] = max(column_widths[col_idx], cluster["width"])
        row_heights[row_idx] = max(row_heights[row_idx], cluster["height"])

    outer_margin = 28
    column_gap = 30
    row_gap = 34
    x_offsets = [outer_margin]
    for width in column_widths[:-1]:
        x_offsets.append(x_offsets[-1] + width + column_gap)
    y_offsets = [outer_margin]
    for height in row_heights[:-1]:
        y_offsets.append(y_offsets[-1] + height + row_gap)

    global_positions: Dict[NodeKey, Tuple[float, float]] = {}
    for index, cluster in enumerate(clusters):
        row_idx = index // column_count
        col_idx = index % column_count
        cluster["x"] = x_offsets[col_idx]
        cluster["y"] = y_offsets[row_idx]
        cluster["positions"] = {}
        for key, (x, y) in cluster["local_positions"].items():
            global_position = (cluster["x"] + x, cluster["y"] + y)
            cluster["positions"][key] = global_position
            global_positions[key] = global_position

    total_width = outer_margin + sum(column_widths) + column_gap * max(0, column_count - 1) + outer_margin
    total_height = outer_margin + sum(row_heights) + row_gap * max(0, row_count - 1) + outer_margin
    return clusters, global_positions, total_width, total_height, cross_action_transitions


def render_cluster_box(cluster: Dict[str, Any]) -> str:
    title = html.escape(cluster["action"])
    summary = html.escape(
        (
            f"animations={len(cluster['lanes'])}, nodes={len(cluster['nodes'])}, "
            f"sequence={len(cluster['sequence_edges'])}, bridges={len(cluster['bridge_transitions'])}, "
            f"components={cluster['component_count']}"
        )
    )
    return (
        f'<rect x="{cluster["x"]:.1f}" y="{cluster["y"]:.1f}" '
        f'width="{cluster["width"]:.1f}" height="{cluster["height"]:.1f}" '
        f'rx="14" class="cluster-box" />'
        f'<text x="{cluster["x"] + 16:.1f}" y="{cluster["y"] + 24:.1f}" class="cluster-title">{title}</text>'
        f'<text x="{cluster["x"] + 16:.1f}" y="{cluster["y"] + 42:.1f}" class="cluster-summary">{summary}</text>'
    )


def render_cluster_guides(cluster: Dict[str, Any]) -> str:
    parts: List[str] = []
    lane_left = cluster["x"] + 112
    lane_right = cluster["x"] + cluster["width"] - 18
    for lane_idx, (action, animation) in enumerate(cluster["lanes"]):
        y = cluster["y"] + 78 + lane_idx * cluster["lane_spacing"]
        label = html.escape(f"{action} / {animation}")
        parts.append(
            f'<text x="{cluster["x"] + 16:.1f}" y="{y + 4:.1f}" class="lane-label">{label}</text>'
        )
        parts.append(
            f'<line x1="{lane_left:.1f}" y1="{y:.1f}" x2="{lane_right:.1f}" y2="{y:.1f}" class="lane-guide" />'
        )
    return "\n".join(parts)


def render_node_frame_labels(
    nodes: List[Dict[str, Any]],
    positions: Dict[NodeKey, Tuple[float, float]],
) -> str:
    parts: List[str] = []
    for node in nodes:
        key = node_key(node)
        if key not in positions:
            continue
        x, y = positions[key]
        parts.append(
            f'<text x="{x:.1f}" y="{y + 16:.1f}" class="node-frame-label">{int(node["frame"])}</text>'
        )
    return "\n".join(parts)


def render_nodes(
    nodes: List[Dict[str, Any]],
    positions: Dict[NodeKey, Tuple[float, float]],
) -> str:
    parts: List[str] = []
    for node in nodes:
        key = node_key(node)
        if key not in positions:
            continue
        x, y = positions[key]
        tooltip = html.escape(f"{node['action']}/{node['animation']}/{node['frame']}")
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.4" class="node"><title>{tooltip}</title></circle>'
        )
    return "\n".join(parts)


def build_line_edge_descriptor(
    edge_id: str,
    item: Dict[str, Any],
    positions: Dict[NodeKey, Tuple[float, float]],
    css_class: str,
    marker_id: str,
    kind_label: str,
) -> Dict[str, Any]:
    source = node_key(item["source"])
    target = node_key(item["target"])
    x1, y1 = positions[source]
    x2, y2 = positions[target]
    x1, y1, x2, y2 = trim_line(x1, y1, x2, y2, 6.0, 10.0)
    tooltip = html.escape(
        f"{kind_label}: {source[0]}/{source[1]}/{source[2]} -> {target[0]}/{target[1]}/{target[2]} | length={int(item.get('length', 0))}"
    )
    return {
        "id": edge_id,
        "source_id": node_id(item["source"]),
        "target_id": node_id(item["target"]),
        "css_class": css_class,
        "marker_id": marker_id,
        "path_d": f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}",
        "tooltip": tooltip,
        "length": int(item.get("length", 0)),
        "kind": item.get("kind", "sequence"),
        "source": item.get("source"),
        "target": item.get("target"),
    }


def build_internal_transition_descriptor(
    edge_id: str,
    item: Dict[str, Any],
    positions: Dict[NodeKey, Tuple[float, float]],
    css_class: str,
    marker_id: str,
) -> Dict[str, Any]:
    source = node_key(item["source"])
    target = node_key(item["target"])
    x1, y1 = positions[source]
    x2, y2 = positions[target]
    dx = x2 - x1
    dy = y2 - y1
    curve_height = max(16.0, min(72.0, abs(dx) * 0.18 + abs(dy) * 0.26))
    cx = (x1 + x2) / 2.0
    cy = min(y1, y2) - curve_height if y1 != y2 else y1 - curve_height
    x1, y1, x2, y2 = trim_quadratic_path(x1, y1, cx, cy, x2, y2, 6.0, 10.0)
    tooltip = html.escape(
        (
            f"transition: {source[0]}/{source[1]}/{source[2]} -> "
            f"{target[0]}/{target[1]}/{target[2]} | "
            f"length={int(item.get('length', 0))} | "
            f"distance={item.get('distance', 0.0):.4f} | "
            f"theta={item.get('theta', 0.0):.4f}"
        )
    )
    return {
        "id": edge_id,
        "source_id": node_id(item["source"]),
        "target_id": node_id(item["target"]),
        "css_class": css_class,
        "marker_id": marker_id,
        "path_d": f"M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}",
        "tooltip": tooltip,
        "length": int(item.get("length", 0)),
        "kind": item.get("kind", "transition"),
        "source": item.get("source"),
        "target": item.get("target"),
        "transition_index": item.get("transition_index"),
        "transition_folder": item.get("transition_folder"),
    }


def build_cross_transition_descriptor(
    edge_id: str,
    item: Dict[str, Any],
    positions: Dict[NodeKey, Tuple[float, float]],
    css_class: str,
    marker_id: str,
) -> Dict[str, Any]:
    source = node_key(item["source"])
    target = node_key(item["target"])
    x1, y1 = positions[source]
    x2, y2 = positions[target]
    dx = x2 - x1
    dy = y2 - y1
    distance = max(1.0, math.hypot(dx, dy))
    offset = min(110.0, 18.0 + distance * 0.14)
    nx = -dy / distance
    ny = dx / distance
    sign = 1.0 if node_id(item["source"]) < node_id(item["target"]) else -1.0
    cx = (x1 + x2) / 2.0 + nx * offset * sign
    cy = (y1 + y2) / 2.0 + ny * offset * sign
    x1, y1, x2, y2 = trim_quadratic_path(x1, y1, cx, cy, x2, y2, 7.0, 11.0)
    tooltip = html.escape(
        (
            f"cross-action: {source[0]}/{source[1]}/{source[2]} -> "
            f"{target[0]}/{target[1]}/{target[2]} | "
            f"length={int(item.get('length', 0))} | "
            f"distance={item.get('distance', 0.0):.4f} | "
            f"theta={item.get('theta', 0.0):.4f}"
        )
    )
    return {
        "id": edge_id,
        "source_id": node_id(item["source"]),
        "target_id": node_id(item["target"]),
        "css_class": css_class,
        "marker_id": marker_id,
        "path_d": f"M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}",
        "tooltip": tooltip,
        "length": int(item.get("length", 0)),
        "kind": item.get("kind", "transition"),
        "source": item.get("source"),
        "target": item.get("target"),
        "transition_index": item.get("transition_index"),
        "transition_folder": item.get("transition_folder"),
    }


def build_visible_edges(
    clusters: List[Dict[str, Any]],
    positions: Dict[NodeKey, Tuple[float, float]],
    cross_action_transitions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    edge_descriptors: List[Dict[str, Any]] = []
    next_edge_idx = 0

    for cluster in clusters:
        for item in cluster["sequence_edges"]:
            edge_descriptors.append(
                build_line_edge_descriptor(
                    edge_id=f"edge_{next_edge_idx}",
                    item=item,
                    positions=positions,
                    css_class="sequence-edge",
                    marker_id="arrow-sequence",
                    kind_label="sequence",
                )
            )
            next_edge_idx += 1

        for item in cluster["bridge_transitions"]:
            edge_descriptors.append(
                build_internal_transition_descriptor(
                    edge_id=f"edge_{next_edge_idx}",
                    item=item,
                    positions=positions,
                    css_class="intra-transition-edge",
                    marker_id="arrow-intra",
                )
            )
            next_edge_idx += 1

    for item in cross_action_transitions:
        edge_descriptors.append(
            build_cross_transition_descriptor(
                edge_id=f"edge_{next_edge_idx}",
                item=item,
                positions=positions,
                css_class="cross-transition-edge",
                marker_id="arrow-cross",
            )
        )
        next_edge_idx += 1

    return edge_descriptors


def render_visible_edges(edge_descriptors: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for edge in edge_descriptors:
        parts.append(
            (
                f'<path id="{edge["id"]}" d="{edge["path_d"]}" class="{edge["css_class"]}" '
                f'marker-end="url(#{edge["marker_id"]})"><title>{edge["tooltip"]}</title></path>'
            )
        )
    return "\n".join(parts)


def normal_frame_key(action: str, animation: str, frame: int) -> str:
    return f"{action}|{animation}|{int(frame)}"


def load_image_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_html_relative_path(target_path: Path, html_output_path: Path) -> str:
    relative = os.path.relpath(target_path, html_output_path.parent)
    return Path(relative).as_posix()


def resolve_image_manifest_assets(
    output_path: Path,
    image_manifest_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if image_manifest_path is None:
        default_manifest = output_path.parent / "rendered_images" / "manifest.json"
        if default_manifest.exists():
            image_manifest_path = default_manifest
        else:
            return None

    if not image_manifest_path.exists():
        raise FileNotFoundError(f"Image manifest not found: {image_manifest_path}")

    raw_manifest = load_image_manifest(image_manifest_path)
    normal_frames = {
        key: make_html_relative_path((image_manifest_path.parent / rel_path).resolve(), output_path)
        for key, rel_path in raw_manifest.get("normal_frames", {}).items()
    }
    transition_frames = {
        folder: [
            make_html_relative_path((image_manifest_path.parent / rel_path).resolve(), output_path)
            for rel_path in rel_paths
        ]
        for folder, rel_paths in raw_manifest.get("transition_frames", {}).items()
    }
    return {
        "manifest_path": str(image_manifest_path),
        "normal_frames": normal_frames,
        "transition_frames": transition_frames,
    }


def build_sequence_image_paths(
    edge: Dict[str, Any],
    image_assets: Optional[Dict[str, Any]],
) -> List[str]:
    if image_assets is None:
        return []

    source = edge.get("source") or {}
    target = edge.get("target") or {}
    if not source or not target:
        return []
    if source.get("action") != target.get("action") or source.get("animation") != target.get("animation"):
        return []

    source_frame = int(source.get("frame", 0))
    target_frame = int(target.get("frame", 0))
    step = 1 if target_frame >= source_frame else -1
    normal_frames = image_assets["normal_frames"]

    image_paths: List[str] = []
    for frame in range(source_frame + step, target_frame + step, step):
        path = normal_frames.get(normal_frame_key(source["action"], source["animation"], frame))
        if path:
            image_paths.append(path)
    return image_paths


def build_transition_image_paths(
    edge: Dict[str, Any],
    image_assets: Optional[Dict[str, Any]],
) -> List[str]:
    if image_assets is None:
        return []

    transition_folder = edge.get("transition_folder")
    if not transition_folder:
        return []

    image_paths = list(image_assets["transition_frames"].get(str(transition_folder), []))
    target = edge.get("target") or {}
    if target:
        target_path = image_assets["normal_frames"].get(
            normal_frame_key(
                target["action"],
                target["animation"],
                int(target["frame"]),
            )
        )
        if target_path:
            image_paths.append(target_path)
    return image_paths


def build_edge_payload(
    edge_descriptors: List[Dict[str, Any]],
    image_assets: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for edge in edge_descriptors:
        kind = edge.get("kind", "sequence")
        image_paths = (
            build_transition_image_paths(edge, image_assets)
            if kind == "transition"
            else build_sequence_image_paths(edge, image_assets)
        )
        payload[edge["id"]] = {
            "source": edge["source_id"],
            "target": edge["target_id"],
            "length": int(edge.get("length", 0)),
            "kind": kind,
            "transition_index": edge.get("transition_index"),
            "transition_folder": edge.get("transition_folder"),
            "image_paths": image_paths,
            "label": f"{edge['source_id']} -> {edge['target_id']}",
        }
    return payload


def build_walker_payload(
    nodes: List[Dict[str, Any]],
    positions: Dict[NodeKey, Tuple[float, float]],
    edge_descriptors: List[Dict[str, Any]],
    fps: float,
    image_assets: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    node_positions = {
        node_id(node): [round(positions[node_key(node)][0], 3), round(positions[node_key(node)][1], 3)]
        for node in nodes
        if node_key(node) in positions
    }
    node_info = {}
    for node in nodes:
        if node_key(node) not in positions:
            continue
        image_path = None
        if image_assets is not None:
            image_path = image_assets["normal_frames"].get(
                normal_frame_key(node["action"], node["animation"], int(node["frame"]))
            )
        node_info[node_id(node)] = {
            "action": node["action"],
            "animation": node["animation"],
            "frame": int(node["frame"]),
            "image_path": image_path,
        }

    outgoing: Dict[str, List[str]] = {}
    for edge in edge_descriptors:
        outgoing.setdefault(edge["source_id"], []).append(edge["id"])
    return {
        "fps": float(fps),
        "has_images": image_assets is not None,
        "nodes": node_positions,
        "node_info": node_info,
        "outgoing": outgoing,
        "edges": build_edge_payload(edge_descriptors, image_assets),
    }


def render_svg(
    payload: Dict[str, Any],
    frame_spacing: int,
    lane_spacing: int,
    max_transition_edges: int,
    fps: float,
    image_assets: Optional[Dict[str, Any]],
) -> Tuple[str, str, str]:
    clusters, positions, width, height, cross_action_transitions = build_action_clusters(
        payload,
        frame_spacing,
        lane_spacing,
        max_transition_edges,
    )
    edge_descriptors = build_visible_edges(
        clusters,
        positions,
        cross_action_transitions,
    )
    all_nodes = [node for cluster in clusters for node in cluster["nodes"]]

    cluster_boxes = "".join(render_cluster_box(cluster) for cluster in clusters)
    cluster_guides = "\n".join(render_cluster_guides(cluster) for cluster in clusters)
    edge_svg = render_visible_edges(edge_descriptors)
    node_labels = render_node_frame_labels(all_nodes, positions)
    node_svg = render_nodes(all_nodes, positions)
    walker_data = build_walker_payload(
        all_nodes,
        positions,
        edge_descriptors,
        fps=fps,
        image_assets=image_assets,
    )
    visible_transition_count = sum(
        len(cluster["bridge_transitions"]) for cluster in clusters
    ) + len(cross_action_transitions)

    svg = f"""
    <svg id="graph-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
      {build_marker_defs()}
      <g>{cluster_boxes}</g>
      <g>{cluster_guides}</g>
      <g>{edge_svg}</g>
      <g>{node_labels}</g>
      <g>{node_svg}</g>
      <circle id="walker-ball" cx="-100" cy="-100" r="6.5" class="walker-ball"></circle>
    </svg>
    """
    summary = html.escape(
        (
            f"actions={len(clusters)}, nodes={len(all_nodes)}, "
            f"edges={len(edge_descriptors)}, transitions={visible_transition_count}, "
            f"visible cross-action transitions={len(cross_action_transitions)}"
        )
    )
    script_data = json.dumps(walker_data, ensure_ascii=True)
    return svg, summary, script_data


def render_html(
    payload: Dict[str, Any],
    frame_spacing: int,
    lane_spacing: int,
    max_transition_edges: int,
    fps: float,
    image_assets: Optional[Dict[str, Any]],
) -> str:
    svg, summary, walker_data = render_svg(
        payload,
        frame_spacing=frame_spacing,
        lane_spacing=lane_spacing,
        max_transition_edges=max_transition_edges,
        fps=fps,
        image_assets=image_assets,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Motion Graph Visualization</title>
  <style>
    :root {{
      --bg: #f7f3ea;
      --panel: #fffdf8;
      --ink: #1f1f1f;
      --muted: #7b7468;
      --guide: #ddd2c2;
      --sequence: #7e9fb0;
      --intra: #d1633f;
      --cross: #9f3d56;
      --node: #1d5c63;
      --border: #cdbca8;
      --cluster: #f5ede2;
      --walker: #f2b134;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f3eadf 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px;
    }}
    h1 {{
      margin: 0 0 4px 0;
      font-size: 28px;
      font-weight: 600;
    }}
    .summary {{
      color: var(--muted);
      font-size: 13px;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 10px;
      margin-bottom: 10px;
      font-size: 13px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    .controls label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .controls input[type="range"] {{
      width: 220px;
    }}
    .controls input[type="number"] {{
      width: 76px;
      padding: 4px 6px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fffdfa;
      color: var(--ink);
      font: inherit;
    }}
    .controls .readout {{
      color: var(--ink);
      font-variant-numeric: tabular-nums;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }}
    .swatch {{
      width: 18px;
      height: 0;
      border-top: 3px solid currentColor;
    }}
    .dot {{
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: currentColor;
      display: inline-block;
    }}
    .content-grid {{
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }}
    .preview-panel {{
      border: 1px solid var(--border);
      border-radius: 16px;
      background: #fffdfa;
      box-shadow: 0 10px 26px rgba(65, 45, 22, 0.08);
      padding: 10px;
    }}
    .preview-title {{
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .preview-frame {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      border-radius: 12px;
      background: linear-gradient(180deg, #f3eadf 0%, #fffdfa 100%);
      border: 1px solid var(--border);
      display: block;
    }}
    .preview-empty {{
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 12px;
      border: 1px dashed var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 18px;
      color: var(--muted);
      background: linear-gradient(180deg, #f3eadf 0%, #fffdfa 100%);
      box-sizing: border-box;
    }}
    .preview-caption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      min-height: 18px;
    }}
    .canvas {{
      border: 1px solid var(--border);
      border-radius: 16px;
      background: #fffdfa;
      box-shadow: 0 10px 26px rgba(65, 45, 22, 0.08);
      padding: 6px;
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .cluster-box {{
      fill: var(--cluster);
      stroke: var(--border);
      stroke-width: 1.2;
    }}
    .cluster-title {{
      fill: var(--ink);
      font-size: 18px;
      font-weight: 600;
    }}
    .cluster-summary {{
      fill: var(--muted);
      font-size: 10px;
    }}
    .lane-label {{
      font-size: 11px;
      fill: var(--ink);
    }}
    .lane-guide {{
      stroke: var(--guide);
      stroke-width: 1;
    }}
    .sequence-edge {{
      fill: none;
      stroke: var(--sequence);
      stroke-width: 1.9;
      opacity: 0.8;
    }}
    .intra-transition-edge {{
      fill: none;
      stroke: var(--intra);
      stroke-width: 2.1;
      opacity: 0.72;
    }}
    .cross-transition-edge {{
      fill: none;
      stroke: var(--cross);
      stroke-width: 1.9;
      opacity: 0.48;
    }}
    .node {{
      fill: var(--node);
      stroke: #fff;
      stroke-width: 1.1;
    }}
    .node-frame-label {{
      fill: var(--muted);
      font-size: 9px;
      text-anchor: middle;
    }}
    .walker-ball {{
      fill: var(--walker);
      stroke: #fff8e1;
      stroke-width: 1.6;
      filter: drop-shadow(0 0 5px rgba(242, 177, 52, 0.6));
    }}
    @media (max-width: 1080px) {{
      .content-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Motion Graph</h1>
    <div class="summary">{summary}</div>
    <div class="legend">
      <span class="legend-item" style="color: var(--sequence);"><span class="swatch"></span>Sequence edge</span>
      <span class="legend-item" style="color: var(--intra);"><span class="swatch"></span>Intra-action bridge</span>
      <span class="legend-item" style="color: var(--cross);"><span class="swatch"></span>Cross-action transition</span>
      <span class="legend-item" style="color: var(--node);"><span class="dot"></span>Transition node</span>
      <span class="legend-item" style="color: var(--walker);"><span class="dot"></span>Random walker</span>
    </div>
    <div class="controls">
      <label for="walker-fps-slider">Walker FPS
        <input id="walker-fps-slider" type="range" min="1" max="120" step="1" value="{int(round(fps))}">
      </label>
      <label for="walker-fps-input">Exact FPS
        <input id="walker-fps-input" type="number" min="1" max="240" step="1" value="{int(round(fps))}">
      </label>
      <span id="walker-fps-readout" class="readout"></span>
    </div>
    <div class="content-grid">
      <div class="preview-panel">
        <div class="preview-title">Walker Preview</div>
        <img id="walker-preview-image" class="preview-frame" alt="Walker frame preview" hidden>
        <div id="walker-preview-empty" class="preview-empty">Render an image library and pass `--image-manifest` to show frame playback here.</div>
        <div id="walker-preview-caption" class="preview-caption"></div>
      </div>
      <div class="canvas">
        {svg}
      </div>
    </div>
  </div>
  <script>
    const walkerData = {walker_data};
    const walkerBall = document.getElementById("walker-ball");
    const edgeIds = Object.keys(walkerData.edges);
    const nodesWithOutgoing = Object.keys(walkerData.outgoing);
    const fpsSlider = document.getElementById("walker-fps-slider");
    const fpsInput = document.getElementById("walker-fps-input");
    const fpsReadout = document.getElementById("walker-fps-readout");
    const previewImage = document.getElementById("walker-preview-image");
    const previewEmpty = document.getElementById("walker-preview-empty");
    const previewCaption = document.getElementById("walker-preview-caption");

    if (walkerBall && edgeIds.length > 0 && nodesWithOutgoing.length > 0) {{
      let walkerFps = Math.max(Number(walkerData.fps) || 30, 1);
      let currentEdgeId = null;
      let currentPath = null;
      let edgeLength = 0;
      let edgeDuration = 1000 / walkerFps;
      let edgeStartTime = null;
      let lastPreviewPath = null;

      function frameDurationMs() {{
        return 1000 / walkerFps;
      }}

      function syncFpsControls() {{
        const rounded = Math.max(1, Math.round(walkerFps));
        if (fpsSlider) {{
          fpsSlider.value = String(rounded);
        }}
        if (fpsInput) {{
          fpsInput.value = String(rounded);
        }}
        if (fpsReadout) {{
          fpsReadout.textContent = `${{rounded}} fps`;
        }}
      }}

      function logicalEdgeDurationMs(edgeId) {{
        const edgeData = walkerData.edges[edgeId] || {{}};
        const logicalEdgeLength = Math.max(Number(edgeData.length) || 0, 1);
        return logicalEdgeLength * frameDurationMs();
      }}

      function setWalkerFps(value) {{
        const nextFps = Math.max(Number(value) || walkerFps || 30, 1);
        const now = performance.now();
        if (currentEdgeId && edgeStartTime !== null) {{
          const previousDuration = Math.max(edgeDuration, 1);
          const progress = Math.min((now - edgeStartTime) / previousDuration, 1);
          walkerFps = nextFps;
          edgeDuration = logicalEdgeDurationMs(currentEdgeId);
          edgeStartTime = now - progress * edgeDuration;
        }} else {{
          walkerFps = nextFps;
          edgeDuration = frameDurationMs();
        }}
        walkerData.fps = walkerFps;
        syncFpsControls();
      }}

      function imagePathAtCurrentPosition(progress) {{
        const edge = walkerData.edges[currentEdgeId];
        if (!edge) {{
          return null;
        }}

        const imagePaths = edge.image_paths || [];
        if (imagePaths.length > 0) {{
          const idx = Math.min(
            imagePaths.length - 1,
            Math.max(0, Math.floor(progress * imagePaths.length))
          );
          return imagePaths[idx];
        }}

        const source = (walkerData.node_info || {{}})[edge.source];
        const target = (walkerData.node_info || {{}})[edge.target];
        if (!source || !target) {{
          return null;
        }}
        return progress < 0.5 ? source.image_path : target.image_path;
      }}

      function updatePreview(progress) {{
        const edge = walkerData.edges[currentEdgeId];
        const imagePath = imagePathAtCurrentPosition(progress);
        if (!previewImage || !previewEmpty || !previewCaption) {{
          return;
        }}

        if (!imagePath) {{
          previewImage.hidden = true;
          previewEmpty.hidden = false;
          previewCaption.textContent = edge ? edge.label || "" : "";
          return;
        }}

        if (imagePath !== lastPreviewPath) {{
          previewImage.src = imagePath;
          lastPreviewPath = imagePath;
        }}
        previewImage.hidden = false;
        previewEmpty.hidden = true;
        previewCaption.textContent = edge ? edge.label || imagePath : imagePath;
      }}

      function pickRandom(list) {{
        return list[Math.floor(Math.random() * list.length)];
      }}

      function chooseNextEdge(fromNodeId) {{
        const outgoing = walkerData.outgoing[fromNodeId];
        if (outgoing && outgoing.length > 0) {{
          return pickRandom(outgoing);
        }}
        return pickRandom(walkerData.outgoing[pickRandom(nodesWithOutgoing)]);
      }}

      function activateEdge(edgeId, timestamp) {{
        currentEdgeId = edgeId;
        currentPath = document.getElementById(edgeId);
        if (!currentPath) {{
          currentEdgeId = null;
          return;
        }}
        edgeLength = Math.max(currentPath.getTotalLength(), 1);
        edgeDuration = logicalEdgeDurationMs(edgeId);
        edgeStartTime = timestamp;
      }}

      function step(timestamp) {{
        if (!currentEdgeId) {{
          activateEdge(pickRandom(edgeIds), timestamp);
        }}
        if (!currentPath) {{
          requestAnimationFrame(step);
          return;
        }}

        const progress = Math.min((timestamp - edgeStartTime) / edgeDuration, 1);
        const point = currentPath.getPointAtLength(progress * edgeLength);
        walkerBall.setAttribute("cx", point.x.toFixed(2));
        walkerBall.setAttribute("cy", point.y.toFixed(2));
        updatePreview(progress);

        if (progress >= 1) {{
          const edge = walkerData.edges[currentEdgeId];
          activateEdge(chooseNextEdge(edge.target), timestamp);
        }}

        requestAnimationFrame(step);
      }}

      if (fpsSlider) {{
        fpsSlider.addEventListener("input", (event) => {{
          setWalkerFps(event.target.value);
        }});
      }}
      if (fpsInput) {{
        fpsInput.addEventListener("input", (event) => {{
          setWalkerFps(event.target.value);
        }});
      }}
      syncFpsControls();
      requestAnimationFrame(step);
    }}
  </script>
</body>
</html>
"""


def save_motion_graph_visualization(
    payload: Dict[str, Any],
    output_path: Path,
    frame_spacing: int = 58,
    lane_spacing: int = 72,
    max_transition_edges: int = 0,
    fps: float = 30.0,
    image_manifest_path: Optional[Path] = None,
) -> Path:
    payload = normalize_motion_graph_payload(dict(payload))
    image_assets = resolve_image_manifest_assets(
        output_path=output_path,
        image_manifest_path=image_manifest_path,
    )
    html_text = render_html(
        payload=payload,
        frame_spacing=frame_spacing,
        lane_spacing=lane_spacing,
        max_transition_edges=max_transition_edges,
        fps=fps,
        image_assets=image_assets,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    payload = normalize_motion_graph_payload(
        load_motion_graph(Path(args.motion_graph))
    )
    output_path = save_motion_graph_visualization(
        payload=payload,
        output_path=Path(args.output),
        frame_spacing=args.frame_spacing,
        lane_spacing=args.lane_spacing,
        max_transition_edges=args.max_transition_edges,
        fps=args.fps,
        image_manifest_path=None if args.image_manifest is None else Path(args.image_manifest),
    )
    print(f"Saved motion graph visualization to {output_path}")


if __name__ == "__main__":
    main()
