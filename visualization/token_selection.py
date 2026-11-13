"""The record a compression run leaves behind for the Fig.4 visualization.

One schema covers every method, because they all answer the same question --
*which 3D regions ended up behind the K tokens the LLM reads* -- even though
they reach the answer differently:

    3dzip     units are Stage-1 voxels. Stage 2 picks K anchors, Stage 3 folds
              nearby voxels into them, so a unit is an anchor, a merged source,
              or unused. A merged unit's colour genuinely reaches the LLM.
    dtc       units are patch tokens. Survivors are kept whole and the rest are
              dropped; nothing is merged, so no unit is ever role 'merged'.
    vispruner same, selected by 2D CLS attention + diversity instead.
    divprune  same, selected by greedy max-min feature diversity instead.
    visionzip units are patch tokens again, but nothing is dropped: dominant
              tokens are anchors and every leftover patch is merged into a
              contextual target, so a merged patch's colour reaches the LLM.
    fastv     units are the voxel-pooled tokens that entered the decoder, since
              this method never touches the vision path. Survivors are anchors,
              the rest are unused. Query-based, so the record answers one
              *question*, not the scene.
    sparsevlm same, plus recycling: the tokens clustered into a surviving
              centre are merged into it.

``merge_assignment[i] == j`` means unit ``i`` contributed to final token ``j``
(``j`` indexes ``anchor_indices``); ``-1`` means unit ``i`` reached no token.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

from .voxel_mapping import xyz_to_voxel


ROLE_ANCHOR, ROLE_MERGED, ROLE_UNUSED = 0, 1, 2
ROLE_NAMES = {ROLE_ANCHOR: 'anchor', ROLE_MERGED: 'merged', ROLE_UNUSED: 'unused'}
# 'full' is uncompressed LLaVA-3D: its voxelize pooling, every voxel a token.
METHODS = ('3dzip', 'dtc', 'vispruner', 'divprune', 'visionzip', 'fastv', 'sparsevlm', 'full')


def base_method(name) -> str:
    """``sparsevlm_layers3-6`` -> ``sparsevlm``: a per-stage record's method.

    SparseVLM prunes at several layers; each earlier stage is recorded under
    ``<method>_layers<first>-<last>``, the decoder layers that attend to it.
    """
    return str(name).split('_layers', 1)[0]


def is_method_package(name) -> bool:
    """A version package at the repo root (``query_token_compression_v4/`` ...).

    Those are ``POOLING`` values of their own (``method_registry``), so their
    records carry the package name as the method. Checked on disk so the
    renderer never has to import the model code.
    """
    name = str(name)
    if not name or '/' in name or name.startswith('.'):
        return False
    init = Path(__file__).resolve().parents[1] / name / '__init__.py'
    # method_registry requires compress(tower, ctx); an ordinary package has none.
    return init.is_file() and re.search(r'\bcompress\b', init.read_text()) is not None


def is_known_method(name) -> bool:
    return base_method(name) in METHODS or is_method_package(name)


# query_token_compression_v4 records which branch kept each token; each part is
# drawn as a record of its own: <method>_focus, <method>_global, <method>_focusregion.
BRANCH_PARTS = (('focus', 'v4_branch', 0), ('global', 'v4_branch', 1),
                ('focusregion', 'v4_focus_region', 1),
                # query_token_compression_v9_3 hybrid: the clustering branch's and the v5-1 branch's tokens
                ('cluster_branch', 'v9_branch', 0), ('v51_branch', 'v9_branch', 1))


def branch_parts(selection: 'TokenSelection') -> list:
    """Derived records of a v4 selection: the tokens the focus vote kept, the ones
    global FPS kept, and the whole focus region (the candidates, not only the kept).
    Empty for any record without the per-voxel ``v4_branch`` extra."""
    if not any(key in selection.extra for _, key, _ in BRANCH_PARTS):
        return []
    parts = []
    for suffix, key, value in BRANCH_PARTS:
        per_unit = np.asarray(selection.extra.get(key, ()), dtype=np.int64).reshape(-1)
        if per_unit.shape[0] != selection.num_units:
            continue
        anchors = np.flatnonzero(per_unit == value).astype(np.int64)
        extra = {k: v for k, v in selection.extra.items() if np.asarray(v).ndim == 0}
        extra['branch_part_of'] = np.array(selection.method)
        parts.append(build_selection(
            scene_id=selection.scene_id, method=f'{selection.method}_{suffix}',
            voxel_size=selection.voxel_size, target_tokens=int(anchors.shape[0]),
            unit_xyz=selection.unit_xyz, anchor_indices=anchors, unit_voxel=selection.unit_voxel,
            voxel_rule=selection.voxel_rule, num_patch_tokens=selection.num_patch_tokens,
            source=selection.source, extra=extra))
    return parts


# The two that prune inside the decoder: their record describes one question,
# so a per-question run is the only one that says anything new about them.
QUERY_BASED_METHODS = ('fastv', 'sparsevlm')


@dataclass
class TokenSelection:
    """What one (scene, method, K) run selected, in the tokens' own frame."""
    scene_id: str
    method: str
    voxel_size: float
    target_tokens: int
    unit_xyz: np.ndarray            # (n_units, 3) float64
    unit_voxel: np.ndarray          # (n_units, 3) int64, forward-pass grid
    anchor_indices: np.ndarray      # (K,) int64 into units
    merge_assignment: np.ndarray    # (n_units,) int64, -1 = unused
    unit_role: np.ndarray           # (n_units,) int8
    voxel_rule: str = 'round'
    merge_cutoff: float = float('nan')
    num_patch_tokens: int = 0
    source: str = ''
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        n = self.unit_xyz.shape[0]
        for name, array, length in (('unit_voxel', self.unit_voxel, n),
                                    ('merge_assignment', self.merge_assignment, n),
                                    ('unit_role', self.unit_role, n)):
            if array.shape[0] != length:
                raise ValueError(f'{name} has {array.shape[0]} rows, expected {length}')
        if self.anchor_indices.size and int(self.anchor_indices.max()) >= n:
            raise ValueError('anchor_indices points past the end of the unit list')

    # -- role views ------------------------------------------------------
    @property
    def num_units(self) -> int:
        return int(self.unit_xyz.shape[0])

    @property
    def num_final_tokens(self) -> int:
        return int(self.anchor_indices.shape[0])

    def units_with_role(self, role) -> np.ndarray:
        return np.flatnonzero(self.unit_role == role)

    @property
    def anchor_voxels(self) -> np.ndarray:
        return self.unit_voxel[self.units_with_role(ROLE_ANCHOR)]

    @property
    def merged_voxels(self) -> np.ndarray:
        return self.unit_voxel[self.units_with_role(ROLE_MERGED)]

    def unique_voxels(self, role) -> np.ndarray:
        voxels = self.unit_voxel[self.units_with_role(role)]
        return np.unique(voxels, axis=0) if voxels.size else voxels.reshape(0, 3)

    # -- persistence -----------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'scene_id': np.array(self.scene_id),
            'method': np.array(self.method),
            'voxel_rule': np.array(self.voxel_rule),
            'voxel_size': np.array(self.voxel_size, dtype=np.float64),
            'target_tokens': np.array(self.target_tokens, dtype=np.int64),
            'merge_cutoff': np.array(self.merge_cutoff, dtype=np.float64),
            'num_patch_tokens': np.array(self.num_patch_tokens, dtype=np.int64),
            'unit_xyz': self.unit_xyz.astype(np.float32),
            'unit_voxel': self.unit_voxel.astype(np.int32),
            'anchor_indices': self.anchor_indices.astype(np.int64),
            'merge_assignment': self.merge_assignment.astype(np.int64),
            'unit_role': self.unit_role.astype(np.int8),
        }
        for key, value in self.extra.items():
            payload[f'extra__{key}'] = np.asarray(value)
        np.savez_compressed(path, **payload)
        return str(path)

    def stats(self) -> dict:
        roles = {name: int((self.unit_role == role).sum()) for role, name in ROLE_NAMES.items()}
        return {
            'scene_id': self.scene_id,
            'method': self.method,
            'voxel_size': self.voxel_size,
            'voxel_rule': self.voxel_rule,
            'target_tokens': int(self.target_tokens),
            'num_final_tokens': self.num_final_tokens,
            'num_compression_units': self.num_units,
            'num_anchor_voxels': roles['anchor'],
            'num_merged_source_voxels': roles['merged'],
            'num_unused_voxels': roles['unused'],
            'num_visualized_voxels': roles['anchor'] + roles['merged'],
            'num_patch_tokens': int(self.num_patch_tokens),
            'merge_cutoff': self.merge_cutoff,
            'selection_source': self.source,
        }


def _roles_from_assignment(n_units, anchor_indices, merge_assignment):
    role = np.full(n_units, ROLE_UNUSED, dtype=np.int8)
    role[merge_assignment >= 0] = ROLE_MERGED
    role[anchor_indices] = ROLE_ANCHOR
    return role


def build_selection(scene_id, method, voxel_size, target_tokens, unit_xyz,
                    anchor_indices, merge_assignment=None, unit_voxel=None,
                    voxel_rule='round', merge_cutoff=float('nan'),
                    num_patch_tokens=0, source='', extra=None) -> TokenSelection:
    """Assemble a record, filling in the parts that follow from the others."""
    unit_xyz = np.asarray(unit_xyz, dtype=np.float64).reshape(-1, 3)
    anchor_indices = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
    n_units = unit_xyz.shape[0]
    if unit_voxel is None:
        unit_voxel = xyz_to_voxel(unit_xyz, voxel_size, voxel_rule)
    unit_voxel = np.asarray(unit_voxel, dtype=np.int64).reshape(-1, 3)
    if merge_assignment is None:
        merge_assignment = np.full(n_units, -1, dtype=np.int64)
        merge_assignment[anchor_indices] = np.arange(anchor_indices.shape[0])
    merge_assignment = np.asarray(merge_assignment, dtype=np.int64).reshape(-1)
    return TokenSelection(
        scene_id=scene_id, method=method, voxel_size=float(voxel_size),
        target_tokens=int(target_tokens), unit_xyz=unit_xyz, unit_voxel=unit_voxel,
        anchor_indices=anchor_indices, merge_assignment=merge_assignment,
        unit_role=_roles_from_assignment(n_units, anchor_indices, merge_assignment),
        voxel_rule=voxel_rule, merge_cutoff=float(merge_cutoff),
        num_patch_tokens=int(num_patch_tokens), source=source, extra=dict(extra or {}))


def load_npz(path) -> TokenSelection:
    data = np.load(Path(path), allow_pickle=False)
    extra = {key[len('extra__'):]: data[key] for key in data.files if key.startswith('extra__')}
    return TokenSelection(
        scene_id=str(data['scene_id']),
        method=str(data['method']),
        voxel_size=float(data['voxel_size']),
        target_tokens=int(data['target_tokens']),
        unit_xyz=data['unit_xyz'].astype(np.float64),
        unit_voxel=data['unit_voxel'].astype(np.int64),
        anchor_indices=data['anchor_indices'].astype(np.int64),
        merge_assignment=data['merge_assignment'].astype(np.int64),
        unit_role=data['unit_role'].astype(np.int8),
        voxel_rule=str(data['voxel_rule']) if 'voxel_rule' in data.files else 'round',
        merge_cutoff=float(data['merge_cutoff']) if 'merge_cutoff' in data.files else float('nan'),
        num_patch_tokens=int(data['num_patch_tokens']) if 'num_patch_tokens' in data.files else 0,
        source=str(path), extra=extra)


def load_provenance_pt(path) -> TokenSelection:
    """Read a legacy ``*_provenance.pt`` bundle into the same record.

    The retired back-trace exporter stored exactly the relations this
    visualization needs, so runs made before this pipeline existed stay usable.
    """
    import torch

    bundle = torch.load(Path(path), map_location='cpu')
    method = str(bundle.get('method', '3dzip'))
    unit_xyz = bundle['voxel_xyz'].numpy().astype(np.float64)
    anchors = bundle['selected_voxel_indices'].numpy().astype(np.int64)
    assignment = bundle['voxel_to_anchor'].numpy().astype(np.int64)
    voxel_size = bundle.get('voxel_size')
    if voxel_size is None:
        raise ValueError(f'{path}: bundle has no voxel_size; cannot place it on the grid')
    extra = {}
    for key, value in (bundle.get('extra') or {}).items():
        extra[key] = value.numpy() if hasattr(value, 'numpy') else np.asarray(value)
    selection = build_selection(
        scene_id=str(bundle.get('scene_id', 'unknown_scene')), method=method,
        voxel_size=float(voxel_size), target_tokens=int(anchors.shape[0]),
        unit_xyz=unit_xyz, anchor_indices=anchors, merge_assignment=assignment,
        merge_cutoff=float(bundle.get('merge_cutoff') or float('nan')),
        num_patch_tokens=int(bundle['patch_to_voxel'].shape[0]) if 'patch_to_voxel' in bundle else 0,
        source=str(path), extra=extra)
    if 'voxel_role' in bundle:
        recorded = bundle['voxel_role'].numpy().astype(np.int8)
        if not np.array_equal(recorded, selection.unit_role):
            raise ValueError(f'{path}: recorded voxel_role disagrees with voxel_to_anchor')
    return selection


def load_selection(path) -> TokenSelection:
    """Load either the current ``.npz`` record or a legacy ``.pt`` bundle."""
    path = Path(path)
    if path.suffix == '.npz':
        return load_npz(path)
    if path.suffix in ('.pt', '.pth'):
        return load_provenance_pt(path)
    raise ValueError(f'{path}: expected a .npz selection record or a .pt provenance bundle')
