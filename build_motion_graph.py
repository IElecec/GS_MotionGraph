import argparse
from pathlib import Path

from motion_graph import MotionGraph
from utils import Database
from visualize_motion_graph import save_motion_graph_visualization


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
        help="keep at most top-k local-minimum transitions for each animation pair; <= 0 keeps all",
    )
    parser.add_argument(
        "--keep-dead-ends",
        action="store_true",
        help="do not prune the graph to its largest strongly connected component",
    )
    parser.add_argument(
        "--visualization",
        help="optional output HTML path for a motion-graph visualization",
    )
    parser.add_argument(
        "--max-visualized-transitions",
        type=int,
        default=0,
        help="limit rendered cross-action transition edges in the optional HTML output; <= 0 renders all",
    )
    parser.add_argument(
        "--visualization-fps",
        type=float,
        default=30.0,
        help="logical playback fps used by the random walker in the optional HTML visualization",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    database = Database(Path(args.database))
    motion_graph = MotionGraph.build(
        database=database,
        similarity_dir=Path(args.similarity_dir),
        distance_threshold=args.distance_threshold,
        top_k=args.top_k,
        prune_dead_ends=not args.keep_dead_ends,
    )
    output_path = motion_graph.save(Path(args.output))
    visualization_path = None
    if args.visualization:
        visualization_path = save_motion_graph_visualization(
            payload=motion_graph.to_dict(),
            output_path=Path(args.visualization),
            max_transition_edges=args.max_visualized_transitions,
            fps=args.visualization_fps,
        )
    print(
        f"Saved motion graph with {len(motion_graph.nodes)} nodes and "
        f"{len(motion_graph.edges)} edges to {output_path}"
    )
    if visualization_path is not None:
        print(f"Saved motion graph visualization to {visualization_path}")


if __name__ == "__main__":
    main()
