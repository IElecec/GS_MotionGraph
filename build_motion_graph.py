import argparse
from pathlib import Path


def resolved_graph_output(path: Path) -> Path:
    return path / "motion_graph.json" if path.suffix == "" else path


def pre_prune_image_output(path: Path) -> Path:
    resolved = resolved_graph_output(path)
    return resolved.with_name(f"{resolved.stem}_before_prune.svg")


def post_prune_image_output(path: Path) -> Path:
    resolved = resolved_graph_output(path)
    return resolved.with_name(f"{resolved.stem}_after_prune.svg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a motion graph from a database and precomputed similarity matrices."
    )
    parser.add_argument(
        "-m",
        "--database",
        required=True,
        help="motion database directory, e.g. /data/motion_db",
    )
    parser.add_argument(
        "-s",
        "--similarity-dir",
        required=True,
        help="directory containing similarity matrices or its parent output directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output graph file path or directory",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=float("inf"),
        help="keep only transitions whose distance is <= this threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="default top-k for transition candidates; <= 0 keeps all",
    )
    parser.add_argument(
        "--top-k-intra-sequence",
        type=int,
        default=None,
        help="top-k for source/target from the same action and animation; defaults to --top-k",
    )
    parser.add_argument(
        "--top-k-inter-animation",
        type=int,
        default=None,
        help="top-k for source/target from the same action but different animations; defaults to --top-k-inter-sequence",
    )
    parser.add_argument(
        "--top-k-inter-sequence",
        type=int,
        default=None,
        help="default top-k for source/target from different sequences; defaults to --top-k",
    )
    parser.add_argument(
        "--keep-dead-ends",
        action="store_true",
        help="do not prune the graph to its largest strongly connected component",
    )
    parser.add_argument(
        "--shortest-path",
        action="store_true",
        help="also export shortest path data",
    )
    parser.add_argument(
        "--path-length-weight",
        type=float,
        default=1.0,
        help="weight for edge.length in shortest-path cost",
    )
    parser.add_argument(
        "--path-distance-weight",
        type=float,
        default=1.0,
        help="weight for edge.distance in shortest-path cost",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from motion_graph import MotionGraph
    from motion_graph.viewer import save_graph_svg
    from utils import Database

    db = Database(Path(args.database))
    top_k_inter_sequence = args.top_k if args.top_k_inter_sequence is None else args.top_k_inter_sequence
    top_k_inter_animation = top_k_inter_sequence if args.top_k_inter_animation is None else args.top_k_inter_animation
    raw_graph = MotionGraph.build(
        database=db,
        similarity_dir=Path(args.similarity_dir),
        distance_threshold=args.distance_threshold,
        top_k_intra_sequence=args.top_k if args.top_k_intra_sequence is None else args.top_k_intra_sequence,
        top_k_inter_animation=top_k_inter_animation,
        top_k_inter_sequence=top_k_inter_sequence,
        prune_dead_ends=False,
    )
    pre_prune_image_path = save_graph_svg(
        raw_graph.to_dict(),
        pre_prune_image_output(Path(args.output)),
    )
    graph = raw_graph if args.keep_dead_ends else raw_graph.largest_strongly_connected_component()
    post_prune_image_path = save_graph_svg(
        graph.to_dict(),
        post_prune_image_output(Path(args.output)),
    )
    output_path = graph.save(Path(args.output))
    shortest_path_path = None
    if args.shortest_path:
        shortest_path_path = graph.save_shortest_paths(
            output_path.parent,
            length_weight=args.path_length_weight,
            distance_weight=args.path_distance_weight,
        )

    print(
        f"Saved motion graph with {len(graph.nodes)} nodes and "
        f"{len(graph.edges)} edges to {output_path}"
    )
    print(f"Saved pre-prune motion graph image at {pre_prune_image_path}")
    print(f"Saved final motion graph image at {post_prune_image_path}")
    if shortest_path_path is not None:
        print(f"Saved shortest path data at {shortest_path_path}")


if __name__ == "__main__":
    main()
