import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from motion_graph.transition import (
    Transition,
    build_transition_window_from_database,
    save_transition_window,
    transition_from_dict,
)
from utils import Database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthesized Gaussian transition frames from a saved motion graph."
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
        help="output directory for synthesized transition windows",
    )
    parser.add_argument(
        "-m",
        "--database",
        help="optional database directory override; defaults to database_dir in motion_graph.json",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="spherical harmonics degree used when loading Gaussian frames",
    )
    parser.add_argument(
        "--num-transition-frames",
        type=int,
        default=4,
        help="number of synthesized in-between frames for each transition",
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=0,
        help="limit the number of exported transitions; <= 0 exports all",
    )
    return parser


def load_motion_graph_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_database_dir(args: argparse.Namespace, payload: Dict[str, Any]) -> Path:
    if args.database:
        return Path(args.database)

    database_dir = payload.get("database_dir")
    if not database_dir:
        raise ValueError("database_dir is missing in motion_graph.json; pass --database explicitly.")
    return Path(database_dir)


def load_transitions(payload: Dict[str, Any]) -> List[Transition]:
    return [transition_from_dict(item) for item in payload.get("transitions", [])]


def export_transition_windows(
    database: Database,
    transitions: List[Transition],
    output_dir: Path,
    sh_degree: int,
    num_transition_frames: int,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gaussian_cache = {}
    saved_dirs: List[Path] = []

    for transition_idx, transition in enumerate(transitions):
        window = build_transition_window_from_database(
            database=database,
            transition=transition,
            sh_degree=sh_degree,
            num_transition_frames=num_transition_frames,
            gaussian_cache=gaussian_cache,
        )
        transition_dir = output_dir / (
            f"{transition_idx:04d}_"
            f"{transition.source.action}_{transition.source.animation}_{transition.source.frame:04d}"
            f"__"
            f"{transition.target.action}_{transition.target.animation}_{transition.target.frame:04d}"
        )
        save_transition_window(window, transition_dir)
        saved_dirs.append(transition_dir)

    return saved_dirs


def main() -> None:
    args = build_parser().parse_args()
    if args.num_transition_frames < 0:
        raise ValueError("--num-transition-frames must be non-negative")

    motion_graph_path = Path(args.motion_graph)
    payload = load_motion_graph_payload(motion_graph_path)
    database_dir = resolve_database_dir(args, payload)
    database = Database(database_dir)

    transitions = load_transitions(payload)
    if args.max_transitions > 0:
        transitions = transitions[: args.max_transitions]

    saved_dirs = export_transition_windows(
        database=database,
        transitions=transitions,
        output_dir=Path(args.output),
        sh_degree=args.sh_degree,
        num_transition_frames=args.num_transition_frames,
    )
    print(f"Saved {len(saved_dirs)} transition windows to {args.output}")


if __name__ == "__main__":
    main()
