import argparse
from pathlib import Path

from utils import (
    Database,
    load_gaussians,
)
from similarity import (
    compute_similarity_matrix,
    save_similarity_matrix,
    transpose_similarity_matrices,
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
    output_root = Path(args.output) / "similarity_matrices"

    animation_entries = []
    for action in database.get_actions():
        for animation in database.get_animations(action):
            animation_entries.append(
                (action, animation, database.get_frames(action, animation))
            )

    gaussian_cache = {}

    def get_gaussians(action: str, animation: str, frames):
        key = (action, animation)
        if key not in gaussian_cache:
            print(f"Loading Gaussians for {action}/{animation}...")
            gaussian_cache[key] = load_gaussians(frames, sh_degree=args.sh_degree)
        return gaussian_cache[key]

    pair_jobs = [
        (source_idx, target_idx)
        for source_idx in range(len(animation_entries))
        for target_idx in range(source_idx, len(animation_entries))
    ]

    for pair_idx, (source_idx, target_idx) in enumerate(pair_jobs, start=1):
        action_a, anim_a, frames_a = animation_entries[source_idx]
        action_b, anim_b, frames_b = animation_entries[target_idx]
        print(
            f"[{pair_idx}/{len(pair_jobs)}] Computing similarity between "
            f"{action_a}/{anim_a} and {action_b}/{anim_b}..."
        )

        similarity_matrix = compute_similarity_matrix(
            frames_a,
            frames_b,
            sh_degree=args.sh_degree,
            window_size=args.window,
            min_gap=args.min_gap,
            gaussians_a=get_gaussians(action_a, anim_a, frames_a),
            gaussians_b=get_gaussians(action_b, anim_b, frames_b),
        )

        save_similarity_matrix(
            similarity_matrix,
            output_dir=output_root / action_a / anim_a / action_b / anim_b,
            window_size=args.window,
            min_gap=args.min_gap,
        )

        if source_idx == target_idx:
            continue

        reverse_similarity_matrix = transpose_similarity_matrices(similarity_matrix)
        save_similarity_matrix(
            reverse_similarity_matrix,
            output_dir=output_root / action_b / anim_b / action_a / anim_a,
            window_size=args.window,
            min_gap=args.min_gap,
        )


if __name__ == "__main__":
    main()
