import argparse
from email import parser
from pathlib import Path

from utils import (
    Database,
)
from similarity import (
    compute_similarity_matrix,
    save_similarity_matrix,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute per-action Gaussian motion similarity matrices."
    )
    parser.add_argument(
        "-m", "--database",
        required=True,
        help="motion database directory, e.g. /data/motion_db",
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="similarity matrix output directory",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="similarity window size (number of frames to consider for each similarity computation)",
    )

    parser.add_argument(
        "--min-gap",
        type=int,
        default=60,
        help="minimum gap between frames for similarity computation",
    )

    parser.add_argument("--sh-degree", type=int, default=3)
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    database = Database(Path(args.database))

    action_names = database.get_actions()
    for action_a in action_names:
        for action_b in action_names:
            print(f"Computing similarity between {action_a} and {action_b}...")
            for anim_a in database.get_animations(action_a):
                for anim_b in database.get_animations(action_b):
                    frames_a = database.data[action_a][anim_a]
                    frames_b = database.data[action_b][anim_b]
                    similarity_matrix = compute_similarity_matrix(
                        frames_a, frames_b, sh_degree=args.sh_degree, window_size=args.window, min_gap=args.min_gap
                    )
                    save_similarity_matrix(
                        similarity_matrix,
                        output_dir=Path(args.output)/ "similarity_matrices" / action_a / anim_a / action_b / anim_b,
                    )


if __name__ == "__main__":
    main()
