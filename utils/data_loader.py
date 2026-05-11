from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterator, List, Optional


def natural_frame_key(path: Path) -> tuple[int, str]:
    # Supports names like:
    # - frame_0.ply
    # - point_cloud_1.ply
    # - point_cloud1.ply
    stem = path.stem
    match = re.search(r"(\d+)$", stem)
    if match is None:
        return (10**12, stem)
    return (int(match.group(1)), stem)


def collect_frame_files(directory: Path) -> List[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".ply"
        ),
        key=natural_frame_key,
    )


@dataclass
class AnimationAssets:
    joint_frames: List[Path]
    skin_frames: List[Path]


class Database:
    """
    Database structure:

    {
      "attack_1": {
        "anim_1": AnimationAssets(...),
        "anim_2": AnimationAssets(...),
      },
      "walk": {...}
    }
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data: Dict[str, Dict[str, AnimationAssets]] = {}
        self.uses_split_roots = False
        self.load()

    def _collect_action_animations(self, root_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
        collected: Dict[str, Dict[str, List[Path]]] = {}
        if not root_dir.is_dir():
            return collected

        for action_dir in sorted(root_dir.iterdir()):
            if not action_dir.is_dir():
                continue

            animations: Dict[str, List[Path]] = {}
            direct_frames = collect_frame_files(action_dir)
            if direct_frames:
                animations["default"] = direct_frames

            for anim_dir in sorted(action_dir.iterdir()):
                if not anim_dir.is_dir():
                    continue

                frames = collect_frame_files(anim_dir)
                if frames:
                    animations[anim_dir.name] = frames

            if animations:
                collected[action_dir.name] = animations

        return collected

    @staticmethod
    def _frame_name_signature(frames: List[Path]) -> List[str]:
        return [frame.name for frame in frames]

    def _load_legacy_structure(self) -> None:
        legacy_actions = self._collect_action_animations(self.base_dir)
        for action_name, animations in legacy_actions.items():
            self.data[action_name] = {
                anim_name: AnimationAssets(
                    joint_frames=frames,
                    skin_frames=[],
                )
                for anim_name, frames in animations.items()
            }

    def _load_split_roots(self, joint_root: Path, skin_root: Path) -> None:
        if not joint_root.is_dir():
            raise FileNotFoundError(
                f"Expected joint root at {joint_root}, but it does not exist."
            )
        if not skin_root.is_dir():
            raise FileNotFoundError(
                f"Expected skin root at {skin_root}, but it does not exist."
            )

        joint_actions = self._collect_action_animations(joint_root)
        skin_actions = self._collect_action_animations(skin_root)

        for action_name, animations in joint_actions.items():
            resolved_animations: Dict[str, AnimationAssets] = {}
            for anim_name, joint_frames in animations.items():
                skin_frames = skin_actions.get(action_name, {}).get(anim_name)
                if not skin_frames:
                    raise FileNotFoundError(
                        "Missing matching skin frames for "
                        f"{action_name}/{anim_name} under {skin_root}."
                    )

                if len(joint_frames) != len(skin_frames):
                    raise ValueError(
                        "joint/skin frame count mismatch for "
                        f"{action_name}/{anim_name}: "
                        f"{len(joint_frames)} vs {len(skin_frames)}."
                    )

                if self._frame_name_signature(joint_frames) != self._frame_name_signature(skin_frames):
                    raise ValueError(
                        "joint/skin frame filenames do not match for "
                        f"{action_name}/{anim_name}."
                    )

                resolved_animations[anim_name] = AnimationAssets(
                    joint_frames=joint_frames,
                    skin_frames=skin_frames,
                )

            if resolved_animations:
                self.data[action_name] = resolved_animations

    def load(self) -> None:
        """Load all actions, animations, and frame paths from base_dir."""
        self.data.clear()
        joint_root = self.base_dir / "joint"
        skin_root = self.base_dir / "skin"
        self.uses_split_roots = joint_root.is_dir() or skin_root.is_dir()

        if self.uses_split_roots:
            self._load_split_roots(joint_root, skin_root)
            return

        self._load_legacy_structure()

    def get_actions(self) -> List[str]:
        """Return all action names."""
        return list(self.data.keys())

    def get_animations(self, action_name: str) -> List[str]:
        """Return animation names under an action."""
        return list(self.data.get(action_name, {}).keys())

    def get_frames(
        self,
        action_name: str,
        anim_name: str,
        variant: str = "joint",
    ) -> List[Path]:
        """Return frame paths for a specific action and animation."""
        assets = self.data[action_name][anim_name]
        if variant == "joint":
            return assets.joint_frames
        if variant == "skin":
            return assets.skin_frames if assets.skin_frames else assets.joint_frames
        raise ValueError(f"Unsupported frame variant: {variant}")

    def get_render_frames(self, action_name: str, anim_name: str) -> List[Path]:
        """Return the frames that should be used for normal rendering."""
        return self.get_frames(action_name, anim_name, variant="skin")

    def get_animation(
        self,
        action_name: str,
        anim_name: str,
        variant: str = "joint",
    ) -> Optional[List[Path]]:
        """Return frames if found, otherwise None."""
        animation = self.data.get(action_name, {}).get(anim_name)
        if animation is None:
            return None
        return self.get_frames(action_name, anim_name, variant=variant)

    def __getitem__(self, action_name: str) -> Dict[str, List[Path]]:
        """Allow database['attack_1'] access."""
        return {
            anim_name: assets.joint_frames
            for anim_name, assets in self.data[action_name].items()
        }

    def __contains__(self, action_name: str) -> bool:
        return action_name in self.data

    def __iter__(self) -> Iterator[str]:
        """Iterate over action names."""
        return iter(self.data)

    def __len__(self) -> int:
        """Number of actions."""
        return len(self.data)
