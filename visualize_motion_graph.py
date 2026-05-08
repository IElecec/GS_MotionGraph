import argparse
import html
import json
import math
import socket
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
        description="Serve motion_graph.json through a Flask-based web visualization."
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
        default=None,
        help="deprecated compatibility flag; static HTML export was removed and this value is ignored",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="host bind address for the Flask visualization server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="port for the Flask visualization server",
    )
    parser.add_argument(
        "--mode",
        choices=("graph", "image"),
        default="graph",
        help="visualization mode: `graph` shows traversal only, `image` also shows rendered image preview",
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
        help="optional manifest.json produced by render_image_library.py; enables image playback in the web view",
    )
    return parser


def load_motion_graph(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_shortest_path_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def port_bind_error(host: str, port: int) -> Optional[OSError]:
    family = socket.AF_INET6 if ":" in host and host != "0.0.0.0" else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        return exc
    finally:
        sock.close()
    return None


def suggest_available_ports(host: str, requested_port: int, limit: int = 3) -> List[int]:
    suggestions: List[int] = []

    if requested_port > 0:
        for candidate in range(requested_port + 1, requested_port + 33):
            if port_bind_error(host, candidate) is None:
                suggestions.append(candidate)
                if len(suggestions) >= limit:
                    return suggestions

    for candidate in (5000, 5001, 7000, 8080, 8765, 9000):
        if candidate == requested_port or candidate in suggestions:
            continue
        if port_bind_error(host, candidate) is None:
            suggestions.append(candidate)
            if len(suggestions) >= limit:
                break
    return suggestions


def lane_key(node: Dict[str, Any]) -> LaneKey:
    return (node["action"], node["animation"])


def node_key(node: Dict[str, Any]) -> NodeKey:
    return (node["action"], node["animation"], int(node["frame"]))


def node_id(node: Dict[str, Any]) -> str:
    return f"{node['action']}|{node['animation']}|{int(node['frame'])}"


def edge_lookup_key(item: Dict[str, Any]) -> str:
    return "|".join(
        [
            node_id(item["source"]),
            node_id(item["target"]),
            str(item.get("kind", "sequence")),
            str(int(item.get("length", 0))),
            f"{float(item.get('distance', 0.0)):.12g}",
            f"{float(item.get('theta', 0.0)):.12g}",
        ]
    )


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
    include_all_nodes: bool = False,
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
    if not include_all_nodes and not visible_transition_keys:
        return [], {}, 56, 56, []

    for action in action_names:
        if include_all_nodes:
            nodes = list(grouped_nodes[action])
        else:
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


def build_helper_graph_edges(
    payload: Dict[str, Any],
    positions: Dict[NodeKey, Tuple[float, float]],
) -> List[Dict[str, Any]]:
    edge_descriptors: List[Dict[str, Any]] = []
    for edge_idx, item in enumerate(payload.get("edges", [])):
        source = node_key(item["source"])
        target = node_key(item["target"])
        if source not in positions or target not in positions:
            continue

        edge_id = f"walker_helper_edge_{edge_idx}"
        if item.get("kind") == "sequence":
            descriptor = build_line_edge_descriptor(
                edge_id=edge_id,
                item=item,
                positions=positions,
                css_class="walker-helper-edge",
                marker_id="",
                kind_label="sequence",
            )
        elif item["source"]["action"] == item["target"]["action"]:
            descriptor = build_internal_transition_descriptor(
                edge_id=edge_id,
                item=item,
                positions=positions,
                css_class="walker-helper-edge",
                marker_id="",
            )
        else:
            descriptor = build_cross_transition_descriptor(
                edge_id=edge_id,
                item=item,
                positions=positions,
                css_class="walker-helper-edge",
                marker_id="",
            )
        descriptor["lookup_key"] = edge_lookup_key(item)
        descriptor["helper"] = True
        edge_descriptors.append(descriptor)
    return edge_descriptors


def render_edge_paths(edge_descriptors: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for edge in edge_descriptors:
        marker_attr = (
            f' marker-end="url(#{edge["marker_id"]})"'
            if edge.get("marker_id")
            else ""
        )
        parts.append(
            (
                f'<path id="{edge["id"]}" d="{edge["path_d"]}" class="{edge["css_class"]}" '
                f'{marker_attr}><title>{edge["tooltip"]}</title></path>'
            )
        )
    return "\n".join(parts)


def normal_frame_key(action: str, animation: str, frame: int) -> str:
    return f"{action}|{animation}|{int(frame)}"


def load_image_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def register_asset_url(
    asset_files: Dict[str, Path],
    asset_ids_by_path: Dict[str, str],
    target_path: Path,
) -> Optional[str]:
    resolved_path = target_path.resolve()
    if not resolved_path.exists():
        return None

    asset_key = str(resolved_path)
    asset_id = asset_ids_by_path.get(asset_key)
    if asset_id is None:
        asset_id = f"asset_{len(asset_files):06d}"
        asset_ids_by_path[asset_key] = asset_id
        asset_files[asset_id] = resolved_path
    return f"/assets/{asset_id}"


def resolve_image_manifest_assets(
    motion_graph_path: Path,
    image_manifest_path: Optional[Path],
    required: bool = False,
) -> Optional[Dict[str, Any]]:
    if image_manifest_path is None:
        default_manifest = motion_graph_path.parent / "rendered_images" / "manifest.json"
        if default_manifest.exists():
            image_manifest_path = default_manifest
        else:
            if required:
                raise FileNotFoundError(
                    "Image mode requires a rendered image manifest. "
                    "Pass --image-manifest or place rendered_images/manifest.json next to motion_graph.json."
                )
            return None

    if not image_manifest_path.exists():
        raise FileNotFoundError(f"Image manifest not found: {image_manifest_path}")

    raw_manifest = load_image_manifest(image_manifest_path)
    asset_files: Dict[str, Path] = {}
    asset_ids_by_path: Dict[str, str] = {}
    normal_frames: Dict[str, str] = {}

    for key, rel_path in raw_manifest.get("normal_frames", {}).items():
        asset_url = register_asset_url(
            asset_files,
            asset_ids_by_path,
            image_manifest_path.parent / rel_path,
        )
        if asset_url is not None:
            normal_frames[key] = asset_url

    transition_frames: Dict[str, List[str]] = {}
    for folder, rel_paths in raw_manifest.get("transition_frames", {}).items():
        urls: List[str] = []
        for rel_path in rel_paths:
            asset_url = register_asset_url(
                asset_files,
                asset_ids_by_path,
                image_manifest_path.parent / rel_path,
            )
            if asset_url is not None:
                urls.append(asset_url)
        transition_frames[folder] = urls

    return {
        "manifest_path": str(image_manifest_path),
        "normal_frames": normal_frames,
        "transition_frames": transition_frames,
        "asset_files": asset_files,
    }


def resolve_shortest_path_assets(
    motion_graph_path: Path,
) -> Optional[Dict[str, Any]]:
    shortest_path_path = motion_graph_path.parent / "shortest_path.json"
    if not shortest_path_path.exists():
        return None
    return {
        "path": str(shortest_path_path),
        "payload": load_shortest_path_payload(shortest_path_path),
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
            "helper": bool(edge.get("helper", False)),
            "label": f"{edge['source_id']} -> {edge['target_id']}",
        }
    return payload


def build_outgoing_payload(edge_descriptors: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    outgoing: Dict[str, List[str]] = {}
    for edge in edge_descriptors:
        outgoing.setdefault(edge["source_id"], []).append(edge["id"])
    return outgoing


def build_shortest_path_navigation_payload(
    shortest_path_assets: Optional[Dict[str, Any]],
    helper_edge_descriptors: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if shortest_path_assets is None:
        return None

    helper_edge_lookup = {
        edge["lookup_key"]: edge["id"]
        for edge in helper_edge_descriptors
        if "lookup_key" in edge
    }
    payload = shortest_path_assets["payload"]
    source_routes: Dict[str, Dict[str, Any]] = {}

    for source_node in payload.get("source_nodes", []):
        source_id = str(source_node.get("source_id", ""))
        if not source_id:
            continue

        target_routes: Dict[str, Any] = {}
        for route in source_node.get("paths_to_other_actions", []):
            target_action = str(route.get("target_action", ""))
            if not target_action:
                continue

            edge_ids: List[str] = []
            route_available = bool(route.get("reachable", False))
            for edge in route.get("edges", []):
                edge_id = helper_edge_lookup.get(edge_lookup_key(edge))
                if edge_id is None:
                    route_available = False
                    edge_ids = []
                    break
                edge_ids.append(edge_id)

            target_routes[target_action] = {
                "reachable": route_available,
                "target_id": route.get("target_id"),
                "total_cost": float(route.get("total_cost", 0.0)),
                "total_frames": int(route.get("total_frames", 0)),
                "total_transition_distance": float(route.get("total_transition_distance", 0.0)),
                "num_transitions": int(route.get("num_transitions", 0)),
                "num_edges": int(route.get("num_edges", len(edge_ids))),
                "edge_ids": edge_ids,
            }
        source_routes[source_id] = target_routes

    return {
        "available_actions": list(payload.get("available_actions", [])),
        "source_routes": source_routes,
        "path": shortest_path_assets.get("path"),
    }


def build_walker_payload(
    nodes: List[Dict[str, Any]],
    positions: Dict[NodeKey, Tuple[float, float]],
    random_edge_descriptors: List[Dict[str, Any]],
    helper_edge_descriptors: List[Dict[str, Any]],
    fps: float,
    image_assets: Optional[Dict[str, Any]],
    shortest_path_assets: Optional[Dict[str, Any]],
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

    all_edge_descriptors = list(random_edge_descriptors) + list(helper_edge_descriptors)
    random_outgoing = build_outgoing_payload(random_edge_descriptors)
    return {
        "fps": float(fps),
        "has_images": image_assets is not None,
        "nodes": node_positions,
        "node_info": node_info,
        "outgoing": random_outgoing,
        "random_edge_ids": [edge["id"] for edge in random_edge_descriptors],
        "random_outgoing": random_outgoing,
        "edges": build_edge_payload(all_edge_descriptors, image_assets),
        "shortest_paths": build_shortest_path_navigation_payload(
            shortest_path_assets,
            helper_edge_descriptors,
        ),
    }


def render_svg(
    payload: Dict[str, Any],
    frame_spacing: int,
    lane_spacing: int,
    max_transition_edges: int,
    fps: float,
    image_assets: Optional[Dict[str, Any]],
    shortest_path_assets: Optional[Dict[str, Any]],
) -> Tuple[str, str, str]:
    clusters, positions, width, height, cross_action_transitions = build_action_clusters(
        payload,
        frame_spacing,
        lane_spacing,
        max_transition_edges,
        include_all_nodes=shortest_path_assets is not None,
    )
    visible_edge_descriptors = build_visible_edges(
        clusters,
        positions,
        cross_action_transitions,
    )
    helper_edge_descriptors = (
        build_helper_graph_edges(payload, positions)
        if shortest_path_assets is not None
        else []
    )
    all_nodes = [node for cluster in clusters for node in cluster["nodes"]]

    cluster_boxes = "".join(render_cluster_box(cluster) for cluster in clusters)
    cluster_guides = "\n".join(render_cluster_guides(cluster) for cluster in clusters)
    helper_edge_svg = render_edge_paths(helper_edge_descriptors)
    edge_svg = render_edge_paths(visible_edge_descriptors)
    node_labels = render_node_frame_labels(all_nodes, positions)
    node_svg = render_nodes(all_nodes, positions)
    walker_data = build_walker_payload(
        all_nodes,
        positions,
        visible_edge_descriptors,
        helper_edge_descriptors,
        fps=fps,
        image_assets=image_assets,
        shortest_path_assets=shortest_path_assets,
    )
    visible_transition_count = sum(
        len(cluster["bridge_transitions"]) for cluster in clusters
    ) + len(cross_action_transitions)

    svg = f"""
    <svg id="graph-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
      {build_marker_defs()}
      <g>{cluster_boxes}</g>
      <g>{cluster_guides}</g>
      <g>{helper_edge_svg}</g>
      <g>{edge_svg}</g>
      <g>{node_labels}</g>
      <g>{node_svg}</g>
      <circle id="walker-ball" cx="-100" cy="-100" r="6.5" class="walker-ball"></circle>
    </svg>
    """
    summary = html.escape(
        (
            f"actions={len(clusters)}, nodes={len(all_nodes)}, "
            f"edges={len(visible_edge_descriptors)}, transitions={visible_transition_count}, "
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
    shortest_path_assets: Optional[Dict[str, Any]],
    view_mode: str,
) -> str:
    svg, summary, walker_data = render_svg(
        payload,
        frame_spacing=frame_spacing,
        lane_spacing=lane_spacing,
        max_transition_edges=max_transition_edges,
        fps=fps,
        image_assets=image_assets,
        shortest_path_assets=shortest_path_assets,
    )
    show_image_preview = view_mode == "image"
    content_grid_class = "content-grid" if show_image_preview else "content-grid graph-mode"
    preview_panel_class = "preview-panel" if show_image_preview else "preview-panel preview-panel-compact"
    preview_block = (
        """
        <div class="preview-title">Walker Preview</div>
        <img id="walker-preview-image" class="preview-frame" alt="Walker frame preview" hidden>
        <div id="walker-preview-empty" class="preview-empty">Render an image library and pass `--image-manifest` to show frame playback here.</div>
        <div id="walker-preview-caption" class="preview-caption"></div>
        """
        if show_image_preview
        else """
        <div class="preview-title">Graph Traversal</div>
        <div class="preview-mode-note">Graph mode is active. The walker follows graph edges without rendered frame preview.</div>
        """
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
      max-width: 1680px;
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
      grid-template-columns: minmax(420px, 560px) minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }}
    .content-grid.graph-mode {{
      grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
    }}
    .preview-panel {{
      border: 1px solid var(--border);
      border-radius: 16px;
      background: #fffdfa;
      box-shadow: 0 10px 26px rgba(65, 45, 22, 0.08);
      padding: 14px;
      position: sticky;
      top: 18px;
    }}
    .preview-panel.preview-panel-compact {{
      padding-bottom: 12px;
    }}
    .preview-title {{
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .preview-mode-note {{
      border-radius: 12px;
      border: 1px dashed var(--border);
      padding: 14px;
      color: var(--muted);
      background: linear-gradient(180deg, #f3eadf 0%, #fffdfa 100%);
      font-size: 12px;
      line-height: 1.45;
    }}
    .preview-frame {{
      width: 100%;
      aspect-ratio: 1 / 1;
      min-height: clamp(420px, 52vh, 680px);
      object-fit: contain;
      border-radius: 12px;
      background: linear-gradient(180deg, #f3eadf 0%, #fffdfa 100%);
      border: 1px solid var(--border);
      display: block;
    }}
    .preview-empty {{
      width: 100%;
      aspect-ratio: 1 / 1;
      min-height: clamp(420px, 52vh, 680px);
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
    .route-panel {{
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
    }}
    .route-toggle-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}
    .route-title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 0;
    }}
    .route-buttons {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .route-button {{
      border: 1px solid var(--border);
      border-radius: 999px;
      background: #fff7ec;
      color: var(--ink);
      padding: 7px 12px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
    }}
    .route-button:hover:not(:disabled) {{
      background: #f6e6c9;
      border-color: #c79f5b;
    }}
    .route-button.active {{
      background: #f2b134;
      border-color: #cf8f08;
      color: #2d2108;
    }}
    .route-button:disabled {{
      cursor: default;
      opacity: 0.45;
    }}
    .route-status {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      min-height: 34px;
      line-height: 1.4;
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
    .walker-helper-edge {{
      fill: none;
      stroke: transparent;
      stroke-width: 6;
      opacity: 0;
      pointer-events: none;
    }}
    .walker-helper-edge.guided-edge,
    .sequence-edge.guided-edge,
    .intra-transition-edge.guided-edge,
    .cross-transition-edge.guided-edge {{
      stroke: var(--walker);
      stroke-width: 3.3;
      opacity: 0.9;
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
      .preview-panel {{
        position: static;
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
    <div class="{content_grid_class}">
      <div class="{preview_panel_class}">
        {preview_block}
        <div class="route-panel">
          <div class="route-toggle-row">
            <div class="route-title">Navigate To Action</div>
            <button id="walker-action-lock-button" type="button" class="route-button">Stay Within Current Action</button>
          </div>
          <div id="walker-route-buttons" class="route-buttons"></div>
          <div id="walker-route-status" class="route-status"></div>
        </div>
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
    const randomEdgeIds = walkerData.random_edge_ids || [];
    const randomOutgoing = walkerData.random_outgoing || walkerData.outgoing || {{}};
    const nodesWithOutgoing = Object.keys(randomOutgoing);
    const shortestPathData = walkerData.shortest_paths || null;
    const fpsSlider = document.getElementById("walker-fps-slider");
    const fpsInput = document.getElementById("walker-fps-input");
    const fpsReadout = document.getElementById("walker-fps-readout");
    const previewImage = document.getElementById("walker-preview-image");
    const previewEmpty = document.getElementById("walker-preview-empty");
    const previewCaption = document.getElementById("walker-preview-caption");
    const routeButtonsContainer = document.getElementById("walker-route-buttons");
    const routeStatus = document.getElementById("walker-route-status");
    const actionLockButton = document.getElementById("walker-action-lock-button");

    if (walkerBall && edgeIds.length > 0 && nodesWithOutgoing.length > 0) {{
      let walkerFps = Math.max(Number(walkerData.fps) || 30, 1);
      let currentEdgeId = null;
      let currentPath = null;
      let edgeLength = 0;
      let edgeDuration = 1000 / walkerFps;
      let edgeStartTime = null;
      let lastPreviewPath = null;
      let desiredPreviewPath = null;
      let seededInitialEdgeId = null;
      let activeHighlightedEdge = null;
      let walkerMode = "random";
      let queuedRouteEdgeIds = [];
      let queuedRouteTargetAction = null;
      let routeButtons = [];
      let actionLockEnabled = false;
      let lockedAction = null;
      const randomEdgeIdsByAction = {{}};
      const imageCache = new Map();
      const preloadedEdgeIds = new Set();

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

      function showPreviewImage(imagePath) {{
        if (!previewImage || !previewEmpty) {{
          return;
        }}
        if (imagePath !== lastPreviewPath) {{
          previewImage.src = imagePath;
          lastPreviewPath = imagePath;
        }}
        previewImage.hidden = false;
        previewEmpty.hidden = true;
      }}

      function preloadImage(imagePath) {{
        if (!imagePath) {{
          return null;
        }}

        let record = imageCache.get(imagePath);
        if (record) {{
          return record;
        }}

        const img = new Image();
        img.decoding = "async";
        record = {{
          image: img,
          loaded: false,
          error: false,
        }};
        img.onload = () => {{
          record.loaded = true;
          if (desiredPreviewPath === imagePath) {{
            showPreviewImage(imagePath);
          }}
        }};
        img.onerror = () => {{
          record.error = true;
        }};
        img.src = imagePath;
        imageCache.set(imagePath, record);
        return record;
      }}

      function imagePathsForEdge(edgeId) {{
        const edge = walkerData.edges[edgeId];
        if (!edge) {{
          return [];
        }}

        const paths = [];
        const edgePaths = Array.isArray(edge.image_paths) ? edge.image_paths : [];
        if (edgePaths.length > 0) {{
          paths.push(...edgePaths);
        }} else {{
          const source = nodeInfoById(edge.source);
          const target = nodeInfoById(edge.target);
          if (source && source.image_path) {{
            paths.push(source.image_path);
          }}
          if (target && target.image_path && target.image_path !== paths[paths.length - 1]) {{
            paths.push(target.image_path);
          }}
        }}
        return paths;
      }}

      function primeEdgeImages(edgeId) {{
        if (!edgeId || preloadedEdgeIds.has(edgeId)) {{
          return;
        }}
        preloadedEdgeIds.add(edgeId);
        imagePathsForEdge(edgeId).forEach((imagePath) => {{
          preloadImage(imagePath);
        }});
      }}

      function primeOutgoingEdgeImages(nodeId, limit = 6) {{
        const outgoing = randomOutgoing[nodeId] || [];
        outgoing.slice(0, limit).forEach((edgeId) => {{
          primeEdgeImages(edgeId);
        }});
      }}

      function primeGuidedRouteImages(edgeIds, limit = 8) {{
        (edgeIds || []).slice(0, limit).forEach((edgeId) => {{
          primeEdgeImages(edgeId);
        }});
      }}

      function updatePreview(progress) {{
        const edge = walkerData.edges[currentEdgeId];
        const imagePath = imagePathAtCurrentPosition(progress);
        if (!previewImage || !previewEmpty || !previewCaption) {{
          return;
        }}

        if (!imagePath) {{
          desiredPreviewPath = null;
          previewImage.hidden = true;
          previewEmpty.hidden = false;
          previewCaption.textContent = edge ? edge.label || "" : "";
          return;
        }}

        desiredPreviewPath = imagePath;
        const record = preloadImage(imagePath);
        if (record && record.loaded) {{
          showPreviewImage(imagePath);
        }} else if (!lastPreviewPath) {{
          previewImage.hidden = true;
          previewEmpty.hidden = false;
          previewEmpty.textContent = "Loading frame preview...";
        }}
        previewCaption.textContent = edge ? edge.label || imagePath : imagePath;
      }}

      function pickRandom(list) {{
        return list[Math.floor(Math.random() * list.length)];
      }}

      function nodeInfoById(nodeId) {{
        return nodeId ? (walkerData.node_info || {{}})[nodeId] || null : null;
      }}

      function actionAtNodeId(nodeId) {{
        const info = nodeInfoById(nodeId);
        return info ? info.action : null;
      }}

      function currentActionAnchorNodeId() {{
        return currentRouteStartNodeId();
      }}

      function ensureRandomEdgeIdsForAction(action) {{
        if (!action) {{
          return [];
        }}
        if (randomEdgeIdsByAction[action]) {{
          return randomEdgeIdsByAction[action];
        }}
        randomEdgeIdsByAction[action] = randomEdgeIds.filter((edgeId) => {{
          const edge = walkerData.edges[edgeId];
          return edge && actionAtNodeId(edge.source) === action && actionAtNodeId(edge.target) === action;
        }});
        return randomEdgeIdsByAction[action];
      }}

      function syncActionLockButton() {{
        if (!actionLockButton) {{
          return;
        }}
        actionLockButton.classList.toggle("active", actionLockEnabled);
        if (actionLockEnabled && lockedAction) {{
          actionLockButton.textContent = `Stay Within ${{lockedAction}}`;
        }} else if (actionLockEnabled) {{
          actionLockButton.textContent = "Stay Within Current Action";
        }} else {{
          actionLockButton.textContent = "Stay Within Current Action";
        }}
      }}

      function chooseRandomEdge(fromNodeId) {{
        if (actionLockEnabled && lockedAction) {{
          const outgoing = (randomOutgoing[fromNodeId] || []).filter((edgeId) => {{
            const edge = walkerData.edges[edgeId];
            return edge && actionAtNodeId(edge.source) === lockedAction && actionAtNodeId(edge.target) === lockedAction;
          }});
          if (outgoing.length > 0) {{
            return pickRandom(outgoing);
          }}

          const fallbackEdges = ensureRandomEdgeIdsForAction(lockedAction);
          if (fallbackEdges.length > 0) {{
            return pickRandom(fallbackEdges);
          }}
        }}

        const outgoing = randomOutgoing[fromNodeId];
        if (outgoing && outgoing.length > 0) {{
          return pickRandom(outgoing);
        }}
        return pickRandom(randomOutgoing[pickRandom(nodesWithOutgoing)]);
      }}

      function setRouteStatus(text) {{
        if (routeStatus) {{
          routeStatus.textContent = text;
        }}
      }}

      function currentRouteStartNodeId() {{
        const edge = walkerData.edges[currentEdgeId];
        return edge ? edge.target : null;
      }}

      function routeForCurrentPosition(targetAction) {{
        if (!shortestPathData) {{
          return null;
        }}
        const startNodeId = currentRouteStartNodeId();
        if (!startNodeId) {{
          return null;
        }}
        const startInfo = (walkerData.node_info || {{}})[startNodeId];
        if (startInfo && startInfo.action === targetAction) {{
          return {{
            reachable: true,
            target_id: startNodeId,
            edge_ids: [],
            total_cost: 0,
            total_frames: 0,
            total_transition_distance: 0,
            num_transitions: 0,
            num_edges: 0,
          }};
        }}
        const sourceRoutes = (shortestPathData.source_routes || {{}})[startNodeId] || {{}};
        return sourceRoutes[targetAction] || null;
      }}

      function refreshRouteButtons() {{
        if (!shortestPathData || !routeButtons.length) {{
          syncActionLockButton();
          return;
        }}
        routeButtons.forEach((button) => {{
          const targetAction = button.dataset.targetAction;
          const route = routeForCurrentPosition(targetAction);
          button.disabled = !route || !route.reachable;
          button.classList.toggle(
            "active",
            walkerMode === "guided" && queuedRouteTargetAction === targetAction,
          );
        }});
        syncActionLockButton();
      }}

      function buildRouteButtons() {{
        if (!routeButtonsContainer || !routeStatus) {{
          return;
        }}
        if (!shortestPathData || !(shortestPathData.available_actions || []).length) {{
          routeButtonsContainer.innerHTML = "";
          setRouteStatus("Generate `shortest_path.json` with `build_motion_graph.py --shortest-path` to enable action navigation.");
          return;
        }}

        routeButtonsContainer.innerHTML = "";
        routeButtons = [];
        (shortestPathData.available_actions || []).forEach((action) => {{
          const button = document.createElement("button");
          button.type = "button";
          button.className = "route-button";
          button.dataset.targetAction = action;
          button.textContent = `To ${{action}}`;
          button.addEventListener("click", () => {{
            const route = routeForCurrentPosition(action);
            if (!route || !route.reachable) {{
              setRouteStatus(`No shortest path is available from the current node to ${{action}}.`);
              refreshRouteButtons();
              return;
            }}

            walkerMode = "guided";
            queuedRouteTargetAction = action;
            queuedRouteEdgeIds = Array.isArray(route.edge_ids) ? [...route.edge_ids] : [];
            if (queuedRouteEdgeIds.length > 0) {{
              setRouteStatus(
                `Following shortest path to ${{action}}: ${{route.total_frames}} frames, ${{route.num_transitions}} transition(s).`
              );
            }} else {{
              setRouteStatus(`Current edge already reaches ${{action}}. Random walk will resume after arrival.`);
            }}
            primeGuidedRouteImages(queuedRouteEdgeIds);
            setHighlightedEdge(currentEdgeId);
            refreshRouteButtons();
          }});
          routeButtonsContainer.appendChild(button);
          routeButtons.push(button);
        }});
        refreshRouteButtons();
        if (!routeStatus.textContent) {{
          setRouteStatus("Choose an action button to interrupt the random walk and follow the saved shortest path.");
        }}
      }}

      function toggleActionLock() {{
        if (!actionLockEnabled) {{
          const nextAction = actionAtNodeId(currentActionAnchorNodeId());
          if (!nextAction) {{
            setRouteStatus("Action lock will take effect after the walker reaches a node.");
            return;
          }}
          actionLockEnabled = true;
          lockedAction = nextAction;
          setRouteStatus(`Single-action mode enabled for ${{lockedAction}}.`);
        }} else {{
          actionLockEnabled = false;
          lockedAction = null;
          setRouteStatus("Single-action mode disabled. Random walk can cross actions again.");
        }}
        refreshRouteButtons();
      }}

      function setHighlightedEdge(edgeId) {{
        if (activeHighlightedEdge) {{
          activeHighlightedEdge.classList.remove("guided-edge");
        }}
        activeHighlightedEdge = edgeId ? document.getElementById(edgeId) : null;
        if (activeHighlightedEdge && walkerMode === "guided") {{
          activeHighlightedEdge.classList.add("guided-edge");
        }}
      }}

      function activateEdge(edgeId, timestamp) {{
        setHighlightedEdge(null);
        currentEdgeId = edgeId;
        currentPath = document.getElementById(edgeId);
        if (!currentPath) {{
          currentEdgeId = null;
          return;
        }}
        edgeLength = Math.max(currentPath.getTotalLength(), 1);
        edgeDuration = logicalEdgeDurationMs(edgeId);
        edgeStartTime = timestamp;
        primeEdgeImages(edgeId);
        const edge = walkerData.edges[edgeId];
        if (edge) {{
          primeOutgoingEdgeImages(edge.target);
        }}
        setHighlightedEdge(edgeId);
        refreshRouteButtons();
        updatePreview(0);
      }}

      function step(timestamp) {{
        if (!currentEdgeId) {{
          const startEdgeId = seededInitialEdgeId || pickRandom(randomEdgeIds);
          seededInitialEdgeId = null;
          activateEdge(startEdgeId, timestamp);
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
          if (walkerMode === "guided") {{
            if (queuedRouteEdgeIds.length > 0) {{
              const nextRouteEdgeId = queuedRouteEdgeIds.shift();
              activateEdge(nextRouteEdgeId, timestamp);
              if (queuedRouteTargetAction && routeStatus) {{
                setRouteStatus(
                  `Following shortest path to ${{queuedRouteTargetAction}}. Remaining edges: ${{queuedRouteEdgeIds.length}}.`
                );
              }}
            }} else {{
              const reachedAction = queuedRouteTargetAction;
              walkerMode = "random";
              queuedRouteTargetAction = null;
              setHighlightedEdge(null);
              if (actionLockEnabled) {{
                lockedAction = reachedAction || actionAtNodeId(edge.target) || lockedAction;
              }}
              if (reachedAction) {{
                if (actionLockEnabled && lockedAction) {{
                  setRouteStatus(`Arrived at ${{reachedAction}}. Continuing within ${{lockedAction}} only.`);
                }} else {{
                  setRouteStatus(`Arrived at ${{reachedAction}}. Random walk resumed.`);
                }}
              }}
              activateEdge(chooseRandomEdge(edge.target), timestamp);
            }}
          }} else {{
            if (actionLockEnabled && !lockedAction) {{
              lockedAction = actionAtNodeId(edge.target);
            }}
            activateEdge(chooseRandomEdge(edge.target), timestamp);
          }}
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
      if (actionLockButton) {{
        actionLockButton.addEventListener("click", () => {{
          toggleActionLock();
        }});
      }}
      buildRouteButtons();
      syncActionLockButton();
      syncFpsControls();
      seededInitialEdgeId = pickRandom(randomEdgeIds);
      primeEdgeImages(seededInitialEdgeId);
      const seededEdge = walkerData.edges[seededInitialEdgeId];
      if (seededEdge) {{
        primeOutgoingEdgeImages(seededEdge.target);
      }}
      requestAnimationFrame(step);
    }}
  </script>
</body>
</html>
"""


def create_motion_graph_app(
    motion_graph_path: Path,
    payload: Dict[str, Any],
    frame_spacing: int = 58,
    lane_spacing: int = 72,
    max_transition_edges: int = 0,
    fps: float = 30.0,
    view_mode: str = "graph",
    image_manifest_path: Optional[Path] = None,
) -> Any:
    try:
        from flask import Flask, abort, send_file
    except ImportError as exc:
        raise RuntimeError(
            "Flask is required for the web visualizer. Install it with `pip install flask`."
        ) from exc

    payload = normalize_motion_graph_payload(dict(payload))
    image_assets = None
    if view_mode == "image":
        image_assets = resolve_image_manifest_assets(
            motion_graph_path=motion_graph_path,
            image_manifest_path=image_manifest_path,
            required=True,
        )
    shortest_path_assets = resolve_shortest_path_assets(
        motion_graph_path=motion_graph_path,
    )
    asset_files = {} if image_assets is None else image_assets["asset_files"]
    page_html = render_html(
        payload=payload,
        frame_spacing=frame_spacing,
        lane_spacing=lane_spacing,
        max_transition_edges=max_transition_edges,
        fps=fps,
        image_assets=image_assets,
        shortest_path_assets=shortest_path_assets,
        view_mode=view_mode,
    )

    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return page_html

    @app.route("/assets/<asset_id>")
    def serve_asset(asset_id: str) -> Any:
        asset_path = asset_files.get(asset_id)
        if asset_path is None or not asset_path.exists():
            abort(404)
        return send_file(asset_path, conditional=True, max_age=3600)

    return app


def serve_motion_graph_visualization(
    motion_graph_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    frame_spacing: int = 58,
    lane_spacing: int = 72,
    max_transition_edges: int = 0,
    fps: float = 30.0,
    view_mode: str = "graph",
    image_manifest_path: Optional[Path] = None,
) -> None:
    motion_graph_path = motion_graph_path.resolve()
    payload = normalize_motion_graph_payload(load_motion_graph(motion_graph_path))
    app = create_motion_graph_app(
        motion_graph_path=motion_graph_path,
        payload=payload,
        frame_spacing=frame_spacing,
        lane_spacing=lane_spacing,
        max_transition_edges=max_transition_edges,
        fps=fps,
        view_mode=view_mode,
        image_manifest_path=image_manifest_path,
    )
    bind_error = port_bind_error(host, port)
    if bind_error is not None:
        suggestions = suggest_available_ports(host, port)
        suggestion_text = ""
        if suggestions:
            suggestion_text = " Try one of: " + ", ".join(f"--port {item}" for item in suggestions) + "."
        raise SystemExit(
            f"Cannot start the Flask visualizer on {host}:{port}: {bind_error}.{suggestion_text}"
        )
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"Serving motion graph visualization at http://{display_host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)


def main() -> None:
    args = build_parser().parse_args()
    if args.output:
        print(
            f"Ignoring deprecated --output value {args.output!r}; "
            "static HTML export was removed in favor of the Flask visualizer."
        )

    serve_motion_graph_visualization(
        motion_graph_path=Path(args.motion_graph),
        host=args.host,
        port=args.port,
        frame_spacing=args.frame_spacing,
        lane_spacing=args.lane_spacing,
        max_transition_edges=args.max_transition_edges,
        fps=args.fps,
        view_mode=args.mode,
        image_manifest_path=None if args.image_manifest is None else Path(args.image_manifest),
    )


if __name__ == "__main__":
    main()
