"""External token-compression methods, one top-level package per version.

The built-in ``POOLING`` branches (voxelize, 3dzip, dtc, ...) live in
``video_encoder.py``. Any other ``POOLING`` value names a package at the repo
root -- e.g. ``POOLING=query_token_compression_v1`` imports
``query_token_compression_v1/`` -- so a new version is a new directory and the
shared LLaVA-3D code is never copied.

A method package exposes:

    def compress(tower, ctx) -> (pooled_video_features, xyz_list, batch_offset)
        ``ctx`` is the dict built in ``RGBDVideoTower.forward`` (see there).

and optionally these flags (all default False):

    NEEDS_QUERY_PATCH_FEATURES  keep CLIP's semantic patch tokens
                                (``CLIPVisionTower._query_patch_features``)
    QUERY_DEPENDENT             output depends on the question, so the
                                per-scene feature cache must never serve it
"""
import importlib
import sys
from pathlib import Path

BUILTIN_POOLINGS = ('voxelize', '3dzip', '3dzip_gt_oracle', 'dtc', 'vispruner',
                    'divprune', 'segpruner', 'visionzip', 'tosa')

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
                f"Unknown pooling '{name}': not a built-in branch and no package "
                f"'{name}/' at {_REPO_ROOT}") from exc
        if not callable(getattr(module, 'compress', None)):
            raise TypeError(f"method package '{name}' has no compress(tower, ctx)")
        _LOADED[name] = module
    return _LOADED[name]


def method_flag(name, flag):
    """A method's capability flag; False for built-ins and unknown names."""
    if not is_external(name):
        return False
    try:
        return bool(getattr(load_method(name), flag, False))
    except NotImplementedError:
        return False
