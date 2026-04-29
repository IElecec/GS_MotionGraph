from pathlib import Path
from typing import Dict, List, Iterator, Optional

def natural_frame_key(path: Path) -> int:
    # frame_0.ply -> 0
    stem = path.stem
    return int(stem.split("_")[-1])


class Database:
    """
    Database structure:

    {
      "attack_1": {
        "anim_1": [frame_0.ply, frame_1.ply, ...],
        "anim_2": [...],
      },
      "walk": {...}
    }
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data: Dict[str, Dict[str, List[Path]]] = {}
        self.load()

    def load(self) -> None:
        """Load all actions, animations, and frame paths from base_dir."""
        self.data.clear()

        for action_dir in sorted(self.base_dir.iterdir()):
            if not action_dir.is_dir():
                continue

            action_name = action_dir.name
            animations: Dict[str, List[Path]] = {}

            for anim_dir in sorted(action_dir.iterdir()):
                if not anim_dir.is_dir():
                    continue

                frames = sorted(
                    anim_dir.glob("frame_*.ply"),
                    key=natural_frame_key,
                )

                if frames:
                    animations[anim_dir.name] = frames

            if animations:
                self.data[action_name] = animations

    def get_actions(self) -> List[str]:
        """Return all action names."""
        return list(self.data.keys())

    def get_animations(self, action_name: str) -> List[str]:
        """Return animation names under an action."""
        return list(self.data.get(action_name, {}).keys())

    def get_frames(self, action_name: str, anim_name: str) -> List[Path]:
        """Return frame paths for a specific action and animation."""
        return self.data[action_name][anim_name]

    def get_animation(
        self,
        action_name: str,
        anim_name: str,
    ) -> Optional[List[Path]]:
        """Return frames if found, otherwise None."""
        return self.data.get(action_name, {}).get(anim_name)

    def __getitem__(self, action_name: str) -> Dict[str, List[Path]]:
        """Allow database['attack_1'] access."""
        return self.data[action_name]

    def __contains__(self, action_name: str) -> bool:
        return action_name in self.data

    def __iter__(self) -> Iterator[str]:
        """Iterate over action names."""
        return iter(self.data)

    def __len__(self) -> int:
        """Number of actions."""
        return len(self.data)