from .data_loader import Database
from .utils_gaussians import GaussianModel, getProjectionMatrix
from .gaussians_helper import load_gaussians

# __all__ = ["Database", "GaussianModel", "load_gaussians", "graphics_utils"]


# def __getattr__(name: str):
#     if name == "GaussianModel":
#         from .utils_gaussians import GaussianModel

#         return GaussianModel
#     if name == "load_gaussians":
#         from .gaussians_helper import load_gaussians

#         return load_gaussians
#     if name == "graphics_utils":
#         from . import graphics_utils

#         return graphics_utils
#     raise AttributeError(f"module 'utils' has no attribute {name!r}")
