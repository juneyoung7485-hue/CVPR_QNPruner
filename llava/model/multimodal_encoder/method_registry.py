"""External token-selection methods for the LLaVA-3D video tower.

``POOLING=voxelize`` is LLaVA-3D itself (all voxel tokens, no selection). Any other ``POOLING`` value names a
package at the repository root -- ``POOLING=qnpruner`` imports ``qnpruner/`` -- that exposes

    def compress(tower, ctx) -> (pooled_video_features, xyz_list, batch_offset)

where ``ctx`` is the dict built in ``RGBDVideoTower.forward`` (see there).
"""
import importlib
import sys
from pathlib import Path

BUILTIN_POOLINGS = ('voxelize',)

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
_LOADED = {}


def is_external(name):
    return bool(name) and name not in BUILTIN_POOLINGS


def load_method(name):
    """Import the method package ``name`` from the repo root (cached)."""
    if name not in _LOADED:
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
            raise NotImplementedError(
                f"Unknown pooling '{name}': not 'voxelize' and no package '{name}/' at {_REPO_ROOT}") from exc
        if not callable(getattr(module, 'compress', None)):
            raise TypeError(f"method package '{name}' has no compress(tower, ctx)")
        _LOADED[name] = module
    return _LOADED[name]
