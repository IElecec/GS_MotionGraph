import argparse
from pathlib import Path


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
        "--shortest-path",
        action="store_true",
        help="also export shortest path data",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from motion_graph import MotionGraph
    from utils import Database

    db = Database(Path(args.database))
    graph = MotionGraph.build(
        database=db,
        similarity_dir=Path(args.similarity_dir),
        distance_threshold=args.distance_threshold,
        top_k=args.top_k,
        prune_dead_ends=not args.keep_dead_ends,
    )
    output_path = graph.save(Path(args.output))
    shortest_path_path = None
    if args.shortest_path:
        shortest_path_path = graph.save_shortest_paths(output_path.parent)

    print(
        f"Saved motion graph with {len(graph.nodes)} nodes and "
        f"{len(graph.edges)} edges to {output_path}"
    )
    if shortest_path_path is not None:
        print(f"Saved shortest path data at {shortest_path_path}")


if __name__ == "__main__":
    main()
