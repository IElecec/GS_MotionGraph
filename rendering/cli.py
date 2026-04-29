import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render all normal and transition Gaussian frames to an image library."
    )
    parser.add_argument(
        "-m",
        "--database",
        required=True,
        help="Motion database directory containing normal Gaussian PLY frames.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for rendered images and manifest.json.",
    )
    parser.add_argument(
        "-t",
        "--transition-dir",
        default=None,
        help="Optional directory containing synthesized transition PLY folders.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fov-deg", type=float, default=50.0)
    parser.add_argument("--camera-position", default=None, help="Fixed camera position as x,y,z")
    parser.add_argument("--camera-target", default=None, help="Fixed camera look-at target as x,y,z")
    parser.add_argument("--camera-up", default="0,1,0", help="Camera up vector as x,y,z")
    parser.add_argument("--azimuth-deg", type=float, default=140.0, help="Auto camera azimuth when no fixed camera is provided")
    parser.add_argument("--elevation-deg", type=float, default=12.0, help="Auto camera elevation when no fixed camera is provided")
    parser.add_argument("--distance-scale", type=float, default=2.6, help="Auto camera distance multiplier")
    parser.add_argument("--cache-size", type=int, default=4, help="Max number of decoded Gaussian frames kept in memory")
    parser.add_argument("--sh-degree", type=int, default=3, help="Gaussian SH degree used when loading PLY")
    parser.add_argument("--background", default="1,1,1", help="Background RGB in [0,1], e.g. 1,1,1")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rendered PNG files instead of skipping them.",
    )
    return parser
