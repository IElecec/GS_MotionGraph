import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

NodeKey = Tuple[str, str, int]
LaneKey = Tuple[str, str]
ActionPair = Tuple[str, str]


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
        default=36,
        help="horizontal spacing per frame",
    )
    parser.add_argument(
        "--lane-spacing",
        type=int,
        default=110,
        help="vertical spacing per animation lane",
    )
    parser.add_argument(
        "--max-transition-edges",
        type=int,
        default=0,
        help="limit rendered cross-action transition edges; <= 0 renders all",
    )
    return parser


def load_motion_graph(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def lane_key(node: Dict[str, Any]) -> LaneKey:
    return (node["action"], node["animation"])


def node_key(node: Dict[str, Any]) -> NodeKey:
    return (node["action"], node["animation"], int(node["frame"]))


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


def build_layout(
    nodes: List[Dict[str, Any]],
    frame_spacing: int,
    lane_spacing: int,
) -> Tuple[Dict[NodeKey, Tuple[int, int]], List[LaneKey], int, int, int]:
    lanes = sorted({lane_key(node) for node in nodes})
    lane_to_index = {lane: idx for idx, lane in enumerate(lanes)}

    max_frame = max((int(node["frame"]) for node in nodes), default=0)
    positions: Dict[NodeKey, Tuple[int, int]] = {}

    x_margin = 170
    y_margin = 70
    for node in nodes:
        key = node_key(node)
        lane = lane_key(node)
        x = x_margin + int(node["frame"]) * frame_spacing
        y = y_margin + lane_to_index[lane] * lane_spacing
        positions[key] = (x, y)

    width = x_margin + max(1, max_frame + 1) * frame_spacing + 120
    height = y_margin + max(1, len(lanes)) * lane_spacing + 80
    return positions, lanes, width, height, max_frame


def render_lane_labels(lanes: List[LaneKey], lane_spacing: int, width: int) -> str:
    y_margin = 70
    parts: List[str] = []
    for lane_idx, (action, animation) in enumerate(lanes):
        y = y_margin + lane_idx * lane_spacing
        label = html.escape(f"{action} / {animation}")
        parts.append(f'<text x="20" y="{y + 5}" class="lane-label">{label}</text>')
        parts.append(
            f'<line x1="150" y1="{y}" x2="{width - 30}" y2="{y}" class="lane-guide" />'
        )
    return "\n".join(parts)


def render_frame_ticks(max_frame: int, frame_spacing: int, height: int) -> str:
    x_margin = 170
    y = 36
    parts: List[str] = []
    step = max(1, int(math.ceil(max_frame / 12))) if max_frame > 0 else 1
    for frame_idx in range(0, max_frame + 1, step):
        x = x_margin + frame_idx * frame_spacing
        parts.append(
            f'<line x1="{x}" y1="40" x2="{x}" y2="{height - 20}" class="frame-guide" />'
        )
        parts.append(f'<text x="{x}" y="{y}" class="frame-label">{frame_idx}</text>')
    return "\n".join(parts)


def render_marker_defs() -> str:
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


def render_edges(
    items: List[Dict[str, Any]],
    positions: Dict[NodeKey, Tuple[int, int]],
    css_class: str,
    kind: str,
    marker_id: str,
) -> str:
    parts: List[str] = []
    for item in items:
        source = node_key(item["source"])
        target = node_key(item["target"])
        if source not in positions or target not in positions:
            continue

        x1, y1 = positions[source]
        x2, y2 = positions[target]
        x1, y1, x2, y2 = trim_line(
            x1,
            y1,
            x2,
            y2,
            start_padding=6.0,
            end_padding=10.0,
        )
        tooltip = html.escape(
            f"{kind}: {source[0]}/{source[1]}/{source[2]} -> {target[0]}/{target[1]}/{target[2]}"
        )
        parts.append(
            (
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{css_class}" marker-end="url(#{marker_id})">'
                f"<title>{tooltip}</title></line>"
            )
        )
    return "\n".join(parts)


def render_curved_transitions(
    transitions: List[Dict[str, Any]],
    positions: Dict[NodeKey, Tuple[int, int]],
    css_class: str,
    marker_id: str,
    start_padding: float = 6.0,
    end_padding: float = 10.0,
) -> str:
    parts: List[str] = []
    for transition in transitions:
        source = node_key(transition["source"])
        target = node_key(transition["target"])
        if source not in positions or target not in positions:
            continue

        x1, y1 = positions[source]
        x2, y2 = positions[target]
        dx = x2 - x1
        dy = y2 - y1
        curve_height = max(28.0, min(180.0, abs(dx) * 0.25 + abs(dy) * 0.3))
        cx = (x1 + x2) / 2.0
        cy = min(y1, y2) - curve_height if y1 != y2 else y1 - curve_height
        x1, y1, x2, y2 = trim_quadratic_path(
            x1,
            y1,
            cx,
            cy,
            x2,
            y2,
            start_padding=start_padding,
            end_padding=end_padding,
        )
        path = f"M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}"
        tooltip = html.escape(
            (
                f"transition: {source[0]}/{source[1]}/{source[2]} -> "
                f"{target[0]}/{target[1]}/{target[2]} | "
                f"distance={transition.get('distance', 0.0):.4f} | "
                f"theta={transition.get('theta', 0.0):.4f}"
            )
        )
        parts.append(
            f'<path d="{path}" class="{css_class}" marker-end="url(#{marker_id})"><title>{tooltip}</title></path>'
        )
    return "\n".join(parts)


def render_nodes(
    nodes: List[Dict[str, Any]],
    positions: Dict[NodeKey, Tuple[int, int]],
) -> str:
    parts: List[str] = []
    for node in nodes:
        key = node_key(node)
        if key not in positions:
            continue
        x, y = positions[key]
        tooltip = html.escape(f"{node['action']}/{node['animation']}/{node['frame']}")
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="4" class="node"><title>{tooltip}</title></circle>'
        )
    return "\n".join(parts)


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


def get_action_internal_transitions(
    payload: Dict[str, Any],
    action: str,
) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    for transition in payload.get("transitions", []):
        if transition["source"]["action"] != action:
            continue
        if transition["target"]["action"] != action:
            continue
        transitions.append(transition)
    transitions.sort(key=transition_sort_key)
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


def render_action_panel(
    action: str,
    nodes: List[Dict[str, Any]],
    payload: Dict[str, Any],
    frame_spacing: int,
    lane_spacing: int,
) -> str:
    sequence_edges = get_action_sequence_edges(payload, action)
    internal_transitions = get_action_internal_transitions(payload, action)
    bridge_transitions, component_count = build_minimal_action_graph(
        nodes,
        sequence_edges,
        internal_transitions,
    )
    positions, lanes, width, height, max_frame = build_layout(
        nodes,
        frame_spacing,
        lane_spacing,
    )

    lane_labels = render_lane_labels(lanes, lane_spacing, width)
    frame_ticks = render_frame_ticks(max_frame, frame_spacing, height)
    sequence_html = render_edges(
        sequence_edges,
        positions,
        css_class="sequence-edge",
        kind="sequence",
        marker_id="arrow-sequence",
    )
    transition_html = render_curved_transitions(
        bridge_transitions,
        positions,
        css_class="intra-transition-edge",
        marker_id="arrow-intra",
    )
    node_html = render_nodes(nodes, positions)
    marker_defs = render_marker_defs()

    summary = html.escape(
        (
            f"animations={len(lanes)}, nodes={len(nodes)}, "
            f"sequence edges={len(sequence_edges)}, bridge transitions={len(bridge_transitions)}, "
            f"remaining components={component_count}"
        )
    )

    return f"""
    <section class="panel">
      <h2>{html.escape(action)}</h2>
      <div class="summary">{summary}</div>
      <div class="canvas">
        <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
          {marker_defs}
          {frame_ticks}
          {lane_labels}
          <g>{sequence_html}</g>
          <g>{transition_html}</g>
          <g>{node_html}</g>
        </svg>
      </div>
    </section>
    """


def build_cross_action_groups(payload: Dict[str, Any]) -> Dict[ActionPair, List[Dict[str, Any]]]:
    groups: Dict[ActionPair, List[Dict[str, Any]]] = {}
    for transition in payload.get("transitions", []):
        source_action = transition["source"]["action"]
        target_action = transition["target"]["action"]
        if source_action == target_action:
            continue
        groups.setdefault((source_action, target_action), []).append(transition)

    for group in groups.values():
        group.sort(key=transition_sort_key)
    return groups


def render_cross_action_panel(
    payload: Dict[str, Any],
    max_transition_edges: int,
) -> str:
    groups = build_cross_action_groups(payload)
    group_items = sorted(
        groups.items(),
        key=lambda item: transition_sort_key(item[1][0]) if item[1] else float("inf"),
    )
    if max_transition_edges > 0:
        group_items = group_items[:max_transition_edges]

    actions = sorted({node["action"] for node in payload.get("nodes", [])})
    if not actions:
        return ""

    width = max(680, 180 * len(actions) + 120)
    height = 320
    y = 180
    positions: Dict[str, Tuple[float, float]] = {}
    step = (width - 140) / max(1, len(actions) - 1) if len(actions) > 1 else 0.0
    for index, action in enumerate(actions):
        x = 70 + step * index if len(actions) > 1 else width / 2.0
        positions[action] = (x, y)

    path_parts: List[str] = []
    label_parts: List[str] = []
    for (source_action, target_action), transitions in group_items:
        if source_action not in positions or target_action not in positions:
            continue

        x1, y1 = positions[source_action]
        x2, y2 = positions[target_action]
        if x1 == x2 and y1 == y2:
            continue

        curve_sign = -1.0 if x1 < x2 else 1.0
        curve_height = 60.0 + min(100.0, abs(x2 - x1) * 0.15)
        cx = (x1 + x2) / 2.0
        cy = y - curve_sign * curve_height
        path = f"M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}"

        sample_lines = []
        for transition in transitions[:4]:
            sample_lines.append(
                (
                    f"{transition['source']['animation']}/{transition['source']['frame']} -> "
                    f"{transition['target']['animation']}/{transition['target']['frame']} "
                    f"(d={transition.get('distance', 0.0):.4f})"
                )
            )
        tooltip = html.escape(
            (
                f"{source_action} -> {target_action} | count={len(transitions)} | "
                f"best_distance={transition_sort_key(transitions[0]):.4f} | "
                f"samples: {'; '.join(sample_lines)}"
            )
        )
        label_y = cy - 8 if cy < y else cy + 18
        label = html.escape(f"{source_action} -> {target_action} ({len(transitions)})")

        x1, y1, x2, y2 = trim_quadratic_path(
            x1,
            y1,
            cx,
            cy,
            x2,
            y2,
            start_padding=68.0,
            end_padding=72.0,
        )
        path = f"M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}"

        path_parts.append(
            f'<path d="{path}" class="cross-transition-edge" marker-end="url(#arrow-cross)"><title>{tooltip}</title></path>'
        )
        label_parts.append(
            f'<text x="{cx:.1f}" y="{label_y:.1f}" class="cross-label">{label}</text>'
        )

    node_parts: List[str] = []
    for action, (x, y_pos) in positions.items():
        label = html.escape(action)
        node_parts.append(
            f'<rect x="{x - 60:.1f}" y="{y_pos - 24:.1f}" width="120" height="48" rx="12" class="action-box" />'
        )
        node_parts.append(
            f'<text x="{x:.1f}" y="{y_pos + 5:.1f}" class="action-box-label">{label}</text>'
        )

    summary = html.escape(
        (
            f"actions={len(actions)}, directed action pairs={len(groups)}, "
            f"rendered cross-action edges={len(group_items)}, "
            f"individual transitions={sum(len(items) for items in groups.values())}"
        )
    )

    if not group_items:
        empty_html = '<div class="summary">No cross-action transitions found.</div>'
    else:
        empty_html = ""
    marker_defs = render_marker_defs()

    return f"""
    <section class="panel">
      <h2>Cross-Action Transitions</h2>
      <div class="summary">{summary}</div>
      {empty_html}
      <div class="canvas">
        <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
          {marker_defs}
          <g>{''.join(path_parts)}</g>
          <g>{''.join(label_parts)}</g>
          <g>{''.join(node_parts)}</g>
        </svg>
      </div>
    </section>
    """


def render_html(
    payload: Dict[str, Any],
    frame_spacing: int,
    lane_spacing: int,
    max_transition_edges: int,
) -> str:
    grouped_nodes = group_nodes_by_action(payload)
    action_panels = [
        render_action_panel(
            action=action,
            nodes=grouped_nodes[action],
            payload=payload,
            frame_spacing=frame_spacing,
            lane_spacing=lane_spacing,
        )
        for action in sorted(grouped_nodes)
    ]
    cross_action_panel = render_cross_action_panel(payload, max_transition_edges)

    summary = html.escape(
        (
            f"actions={len(grouped_nodes)}, nodes={payload.get('num_nodes', 0)}, "
            f"edges={payload.get('num_edges', 0)}, transitions={payload.get('num_transitions', 0)}"
        )
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
      --box: #efe2d1;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f3eadf 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0 0 6px 0;
      font-size: 30px;
      font-weight: 600;
    }}
    .summary {{
      color: var(--muted);
      font-size: 14px;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      margin-top: 14px;
      font-size: 14px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 22px;
      height: 0;
      border-top: 3px solid currentColor;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: currentColor;
      display: inline-block;
    }}
    .grid {{
      display: grid;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px 20px;
      box-shadow: 0 10px 26px rgba(65, 45, 22, 0.08);
    }}
    .panel h2 {{
      margin: 0 0 6px 0;
      font-size: 22px;
      font-weight: 600;
    }}
    .canvas {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fffdfa;
      margin-top: 14px;
    }}
    svg {{
      display: block;
      min-width: 100%;
    }}
    .lane-label {{
      font-size: 13px;
      fill: var(--ink);
    }}
    .frame-label {{
      font-size: 12px;
      text-anchor: middle;
      fill: var(--muted);
    }}
    .lane-guide {{
      stroke: var(--guide);
      stroke-width: 1;
    }}
    .frame-guide {{
      stroke: #efe7db;
      stroke-width: 1;
    }}
    .sequence-edge {{
      stroke: var(--sequence);
      stroke-width: 2;
      opacity: 0.78;
    }}
    .intra-transition-edge {{
      fill: none;
      stroke: var(--intra);
      stroke-width: 2.3;
      opacity: 0.7;
    }}
    .cross-transition-edge {{
      fill: none;
      stroke: var(--cross);
      stroke-width: 2.5;
      opacity: 0.72;
    }}
    .node {{
      fill: var(--node);
      stroke: #fff;
      stroke-width: 1.2;
    }}
    .action-box {{
      fill: var(--box);
      stroke: var(--border);
      stroke-width: 1.4;
    }}
    .action-box-label {{
      fill: var(--ink);
      font-size: 14px;
      text-anchor: middle;
      font-weight: 600;
    }}
    .cross-label {{
      fill: var(--cross);
      font-size: 12px;
      text-anchor: middle;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Motion Graph</h1>
      <div class="summary">{summary}</div>
      <div class="legend">
        <span class="legend-item" style="color: var(--sequence);"><span class="swatch"></span>Sequence path inside one animation</span>
        <span class="legend-item" style="color: var(--intra);"><span class="swatch"></span>Minimal intra-action bridge transition</span>
        <span class="legend-item" style="color: var(--cross);"><span class="swatch"></span>Cross-action transition summary</span>
        <span class="legend-item">Arrowheads indicate direction</span>
        <span class="legend-item" style="color: var(--node);"><span class="dot"></span>Frame node</span>
      </div>
    </section>
    <div class="grid">
      {''.join(action_panels)}
      {cross_action_panel}
    </div>
  </div>
</body>
</html>
"""


def save_motion_graph_visualization(
    payload: Dict[str, Any],
    output_path: Path,
    frame_spacing: int = 36,
    lane_spacing: int = 110,
    max_transition_edges: int = 0,
) -> Path:
    html_text = render_html(
        payload=payload,
        frame_spacing=frame_spacing,
        lane_spacing=lane_spacing,
        max_transition_edges=max_transition_edges,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    payload = load_motion_graph(Path(args.motion_graph))
    output_path = save_motion_graph_visualization(
        payload=payload,
        output_path=Path(args.output),
        frame_spacing=args.frame_spacing,
        lane_spacing=args.lane_spacing,
        max_transition_edges=args.max_transition_edges,
    )
    print(f"Saved motion graph visualization to {output_path}")


if __name__ == "__main__":
    main()
