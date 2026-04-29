import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Point = Tuple[float, float, float]

BASE_POINTS: List[Point] = [
    (0.00, 0.00, 0.00),
    (0.00, 0.25, 0.00),
    (0.00, 0.50, 0.00),
    (-0.18, 0.32, 0.02),
    (0.18, 0.32, -0.02),
    (-0.10, -0.35, 0.04),
    (0.10, -0.35, -0.04),
    (0.00, 0.12, 0.10),
]

ANIMATION_SPECS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("walk", "anim_1"): {"frames": 16, "forward": 1.10, "arm_swing": 0.12, "foot_lift": 0.10},
    ("attack_1", "anim_1"): {"frames": 16, "forward": 0.35, "arm_swing": 0.28, "foot_lift": 0.05},
    ("attack_1", "anim2"): {"frames": 16, "forward": 0.30, "arm_swing": 0.22, "foot_lift": 0.07},
    ("attack_2", "anim_1"): {"frames": 16, "forward": 0.20, "arm_swing": 0.18, "foot_lift": 0.14},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate sample Gaussian PLY files under data/char1."
    )
    parser.add_argument(
        "--output-dir",
        default="data/char1",
        help="output character directory",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="spherical harmonics degree used by GaussianModel.load_ply",
    )
    return parser


def sh_rest_count(sh_degree: int) -> int:
    return 3 * ((sh_degree + 1) ** 2 - 1)


def smooth_pulse(t: float, center: float, width: float) -> float:
    scaled = (t - center) / max(width, 1e-6)
    return math.exp(-(scaled * scaled))


def transform_points(
    action: str,
    animation: str,
    frame_idx: int,
    frame_count: int,
) -> List[Point]:
    spec = ANIMATION_SPECS[(action, animation)]
    t = frame_idx / max(frame_count - 1, 1)
    cycle = 2.0 * math.pi * t
    forward = spec["forward"] * t
    bob = 0.03 * math.sin(cycle)
    arm_swing = spec["arm_swing"]
    foot_lift = spec["foot_lift"]

    attack_pulse = smooth_pulse(t, center=0.60, width=0.18)
    secondary_pulse = smooth_pulse(t, center=0.38, width=0.15)

    points: List[Point] = []
    for point_idx, (x, y, z) in enumerate(BASE_POINTS):
        px = x
        py = y
        pz = z

        if action == "walk":
            pz += forward
            py += bob
            if point_idx == 3:
                px -= arm_swing * math.sin(cycle)
            elif point_idx == 4:
                px += arm_swing * math.sin(cycle)
            elif point_idx == 5:
                py += foot_lift * max(0.0, math.sin(cycle))
                pz -= 0.05 * math.sin(cycle)
            elif point_idx == 6:
                py += foot_lift * max(0.0, -math.sin(cycle))
                pz += 0.05 * math.sin(cycle)

        elif action == "attack_1" and animation == "anim_1":
            pz += 0.35 * attack_pulse + forward
            px += 0.03 * math.sin(cycle)
            if point_idx == 4:
                px += arm_swing * attack_pulse
                py += 0.08 * attack_pulse
                pz += 0.32 * attack_pulse
            elif point_idx == 3:
                px -= 0.10 * attack_pulse
                py += 0.03 * attack_pulse
            elif point_idx in (5, 6):
                py += 0.04 * secondary_pulse

        elif action == "attack_1" and animation == "anim2":
            pz += 0.25 * attack_pulse + forward
            px -= 0.08 * secondary_pulse
            if point_idx == 4:
                px += 0.18 * attack_pulse
                py += 0.14 * attack_pulse
                pz += 0.18 * attack_pulse
            elif point_idx == 3:
                px -= 0.16 * attack_pulse
                py += 0.10 * attack_pulse
            elif point_idx == 2:
                py += 0.04 * attack_pulse

        elif action == "attack_2":
            pz += 0.18 * t
            py += 0.12 * math.sin(math.pi * t)
            if point_idx == 4:
                px += 0.10 * secondary_pulse
                py += 0.10 * attack_pulse
                pz += 0.20 * attack_pulse
            elif point_idx == 3:
                px -= 0.08 * attack_pulse
                py += 0.16 * attack_pulse
            elif point_idx in (5, 6):
                py += foot_lift * math.sin(math.pi * t)

        points.append((px, py, pz))

    return points


def vertex_row(point: Point, point_idx: int, sh_degree: int) -> List[float]:
    x, y, z = point
    dc = [
        0.35 + 0.05 * point_idx,
        0.20 + 0.03 * (point_idx % 3),
        0.15 + 0.02 * (point_idx % 5),
    ]
    rest = [0.0] * sh_rest_count(sh_degree)
    opacity = [0.1]
    scales = [-2.2, -2.2, -2.2]
    rotation = [1.0, 0.0, 0.0, 0.0]
    return [x, y, z, 0.0, 0.0, 1.0] + dc + rest + opacity + scales + rotation


def property_names(sh_degree: int) -> Iterable[str]:
    yield from ("x", "y", "z", "nx", "ny", "nz")
    yield from (f"f_dc_{idx}" for idx in range(3))
    yield from (f"f_rest_{idx}" for idx in range(sh_rest_count(sh_degree)))
    yield "opacity"
    yield from (f"scale_{idx}" for idx in range(3))
    yield from (f"rot_{idx}" for idx in range(4))


def write_ply(path: Path, points: List[Point], sh_degree: int) -> None:
    rows = [vertex_row(point, point_idx, sh_degree) for point_idx, point in enumerate(points)]
    names = list(property_names(sh_degree))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(rows)}\n")
        for name in names:
            handle.write(f"property float {name}\n")
        handle.write("end_header\n")
        for row in rows:
            handle.write(" ".join(f"{value:.6f}" for value in row))
            handle.write("\n")


def generate_samples(output_dir: Path, sh_degree: int) -> int:
    file_count = 0
    for (action, animation), spec in ANIMATION_SPECS.items():
        frame_count = int(spec["frames"])
        for frame_idx in range(frame_count):
            frame_path = output_dir / action / animation / f"frame_{frame_idx}.ply"
            points = transform_points(action, animation, frame_idx, frame_count)
            write_ply(frame_path, points, sh_degree)
            file_count += 1
    return file_count


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    file_count = generate_samples(output_dir, sh_degree=args.sh_degree)
    print(f"Generated {file_count} sample PLY files under {output_dir}")


if __name__ == "__main__":
    main()
