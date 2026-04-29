from collections import OrderedDict
from pathlib import Path

import torch

from utils import GaussianModel


class GaussianFrameCache:
    def __init__(self, max_size: int, sh_degree: int):
        self.max_size = max(1, int(max_size))
        self.sh_degree = int(sh_degree)
        self._cache: OrderedDict[str, GaussianModel] = OrderedDict()

    def get(self, frame_path: Path) -> GaussianModel:
        key = str(frame_path.resolve())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame file: {frame_path}")

        gaussian = GaussianModel(sh_degree=self.sh_degree)
        gaussian.load_ply(str(frame_path))
        self._cache[key] = gaussian
        self._cache.move_to_end(key)

        while len(self._cache) > self.max_size:
            _, evicted = self._cache.popitem(last=False)
            del evicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return gaussian
