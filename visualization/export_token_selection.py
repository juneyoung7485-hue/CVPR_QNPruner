"""Forward-time capture of what each compression method selected.

Imported lazily by ``llava/model/multimodal_encoder/video_encoder.py`` and
gated behind ``TOKEN_SEL``. With the switch off nothing here is constructed and
the forward pass is the stock path; with it on, every tensor is detached and
copied to CPU before anything is written, so a recorded run answers exactly as
an unrecorded one.

What is written is the *index* relation the forward pass produced -- which
unit is an anchor, which unit merged into which anchor -- never a
nearest-neighbour guess reconstructed afterwards.

Environment
    TOKEN_SEL                 1 turns the exporter on
    TOKEN_SEL_DIR             output root (default pred/token_selection)
    TOKEN_SEL_TAG             free-form run label stored in the record
    TOKEN_SEL_SCENE_SUBDIR    1 writes to <dir>/<scene_id>/ (default 1)
    TOKEN_SEL_ONCE_PER_SCENE  1 records only the first question of a scene
    TOKEN_SEL_MAX_SCENES      stop after N scenes; 0 = no limit
    TOKEN_SEL_QID             question label appended to the filename
    TOKEN_SEL_VOXEL_SIZE      display grid for methods that do not voxelize
    TOKEN_SEL_DEBUG           1 prints one line per export (default 1)
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Set

import numpy as np

from .token_selection import build_selection, is_known_method


_EXPORTED: Set[str] = set()
_WARNED_ONCE_PER_SCENE: Set[str] = set()


def _flag(name, default='0'):
    return os.environ.get(name, default) not in ('0', 'false', 'False', '')


@dataclass
class SelectionExportConfig:
    enabled: bool = False
    output_dir: str = os.path.join('pred', 'token_selection')
    tag: str = ''
    scene_subdir: bool = True
    once_per_scene: bool = True
    max_scenes: int = 0
    question_id: str = ''
    debug: bool = True


def config() -> SelectionExportConfig:
    """Re-read the environment on every call so a run can toggle it per question."""
    if not _flag('TOKEN_SEL', '0'):
        return SelectionExportConfig(enabled=False)
    return SelectionExportConfig(
        enabled=True,
        output_dir=os.environ.get('TOKEN_SEL_DIR', os.path.join('pred', 'token_selection')),
        tag=os.environ.get('TOKEN_SEL_TAG', ''),
        scene_subdir=_flag('TOKEN_SEL_SCENE_SUBDIR', '1'),
        once_per_scene=_flag('TOKEN_SEL_ONCE_PER_SCENE', '1'),
        max_scenes=int(os.environ.get('TOKEN_SEL_MAX_SCENES', '0')),
        question_id=os.environ.get('TOKEN_SEL_QID', ''),
        debug=_flag('TOKEN_SEL_DEBUG', '1'),
    )


def enabled() -> bool:
    return config().enabled


def per_question_export_active() -> bool:
    """True when the exporter is recording every question, not every scene.

    ``llava/eval/scene_cache.py`` asks this before serving a cached vision
    tower output: the exporter runs *inside* the tower, so a cache hit would
    collapse a per-question run into one record per scene.
    """
    cfg = config()
    return cfg.enabled and not cfg.once_per_scene


def display_voxel_size(default=0.2) -> float:
    """Grid used to *draw* a method that never voxelized (DTC, VisPruner).

    Those two select individual patch tokens, so they have no voxel size of
    their own; the figure still has to place them on a grid, and comparing
    methods only means something when it is the same grid 3DZip ran on.
    """
    override = os.environ.get('TOKEN_SEL_VOXEL_SIZE')
    size = float(override) if override else float(default)
    return size if size > 0 else 0.2


def scene_id_from_key(scene_key) -> str:
    if scene_key is None:
        return 'unknown_scene'
    return os.path.basename(str(scene_key).rstrip('/'))


def should_record(cfg: SelectionExportConfig, scene_key, method: str) -> bool:
    if not cfg.enabled or not is_known_method(method):
        return False
    key = f'{method}:{scene_id_from_key(scene_key)}'
    if cfg.once_per_scene and key in _EXPORTED:
        return False
    if cfg.max_scenes > 0 and len(_EXPORTED) >= cfg.max_scenes and key not in _EXPORTED:
        return False
    return True


def reset_exported_scenes() -> None:
    _EXPORTED.clear()
    _WARNED_ONCE_PER_SCENE.clear()


def _numpy(tensor, dtype=None):
    array = tensor.detach()
    if array.is_floating_point():
        # bf16 has no NumPy counterpart, and the model runs in it.
        array = array.float()
    array = array.cpu().numpy()
    return array if dtype is None else array.astype(dtype)


def voxel_grid_from_xyz(token_xyz, voxel_size):
    """The forward pass' own cell for each coordinate.

    ``unproject.voxelization`` computes ``round(xyz / voxel_size)`` *in the
    tensor's dtype*, which during inference is bfloat16. Redoing it in float32
    would disagree near cell boundaries, and bf16 boundaries are wide enough
    (~2 cm at 5 m) for that to matter on a 20 cm grid -- so the division and the
    rounding stay in the incoming dtype, exactly as the model did them.
    """
    import torch

    return torch.round(token_xyz.detach() / float(voxel_size)).float().cpu().numpy().astype(np.int64)


def voxel_grid_per_unit(token_xyz, point_to_voxel, voxel_size, num_units):
    """Integer grid coordinate of each voxel, taken from its own points.

    Every point in a voxel shares one ``round(xyz / voxel_size)`` by
    construction, so copying any member's coordinate is exact -- unlike
    re-rounding the voxel centroid, which is only almost always the same cell.
    """
    import torch

    grid = torch.round(token_xyz.detach() / float(voxel_size)).float()
    out = torch.zeros(num_units, 3, dtype=grid.dtype, device=grid.device)
    out.index_copy_(0, point_to_voxel.detach().long(), grid)
    return out.cpu().numpy().astype(np.int64)


def _write(selection, cfg: SelectionExportConfig, scene_key):
    scene_id = selection.scene_id
    root = Path(cfg.output_dir).expanduser()
    if cfg.scene_subdir:
        root = root / scene_id
    suffix = ''
    if cfg.question_id:
        suffix = '_' + re.sub(r'[^A-Za-z0-9_.-]', '_', str(cfg.question_id))[:80]
    path = root / f'{scene_id}_{selection.method}_K{selection.target_tokens}{suffix}.npz'
    selection.extra['run_tag'] = np.array(cfg.tag)
    selection.extra['scene_key'] = np.array(str(scene_key))
    selection.save(path)
    _EXPORTED.add(f'{selection.method}:{scene_id}')
    if cfg.debug:
        stats = selection.stats()
        print(f"[TokenSel] {path} | K={stats['num_final_tokens']} "
              f"units={stats['num_compression_units']} anchors={stats['num_anchor_voxels']} "
              f"merged={stats['num_merged_source_voxels']} unused={stats['num_unused_voxels']}",
              flush=True)
    return str(path)


def record_3dzip(scene_key, voxel_size, target_tokens, unit_xyz, anchor_indices,
                 remain_idx=None, nearest_anchor=None, close_mask=None,
                 unit_voxel=None, merge_cutoff=float('nan'), num_patch_tokens=0,
                 method='3dzip', extra=None):
    """Stage 1 voxels + Stage 2 anchors + the Stage 3 merges that actually ran.

    ``nearest_anchor`` alone does not make a voxel used: only the ones that also
    pass the spatial cutoff (``close_mask``) are summed into a final token, and
    only those are recorded as merged. ``method``: a version package that runs
    the same voxel pipeline (query_token_compression_v1/v2) records under its
    own name, so it never overwrites the 3dzip record.
    """
    cfg = config()
    if not should_record(cfg, scene_key, method):
        return None
    try:
        anchors = _numpy(anchor_indices, np.int64).reshape(-1)
        xyz = _numpy(unit_xyz, np.float64)
        n_units = xyz.shape[0]
        assignment = np.full(n_units, -1, dtype=np.int64)
        assignment[anchors] = np.arange(anchors.shape[0], dtype=np.int64)
        if remain_idx is not None and close_mask is not None and int(close_mask.sum()) > 0:
            remain = _numpy(remain_idx, np.int64).reshape(-1)
            nearest = _numpy(nearest_anchor, np.int64).reshape(-1)
            close = _numpy(close_mask, bool).reshape(-1)
            assignment[remain[close]] = nearest[close]
        selection = build_selection(
            scene_id=scene_id_from_key(scene_key), method=method, voxel_size=voxel_size,
            target_tokens=int(anchors.shape[0]), unit_xyz=xyz, anchor_indices=anchors,
            merge_assignment=assignment, unit_voxel=unit_voxel, merge_cutoff=merge_cutoff,
            num_patch_tokens=num_patch_tokens, source='forward', extra=_extra(extra))
        if int(target_tokens) != selection.num_final_tokens:
            selection.extra['requested_target_tokens'] = np.array(int(target_tokens))
        return _write(selection, cfg, scene_key)
    except Exception as exc:                 # visualization must never break inference
        print(f'[TokenSel][WARN] {method} export failed: {exc}', flush=True)
        return None


def _extra(extra):
    return {key: _numpy(value) if hasattr(value, 'detach') else np.asarray(value)
            for key, value in (extra or {}).items()}


def record_voxel_selection(scene_key, method, voxel_size, target_tokens, unit_xyz,
                           anchor_indices, unit_voxel, num_patch_tokens=0, extra=None):
    """A version package that keeps K of LLaVA-3D's voxel tokens unchanged
    (query_token_compression_v3*, v4): units are the voxels, the kept ones are
    anchors, nothing is merged. ``unit_voxel`` must come from the voxels' own
    patches (``voxel_grid_per_unit``) so the cells are exactly the forward's."""
    return record_3dzip(scene_key, voxel_size, target_tokens, unit_xyz, anchor_indices,
                        unit_voxel=unit_voxel, num_patch_tokens=num_patch_tokens,
                        method=method, extra=extra)


def record_merge_selection(scene_key, method, voxel_size, target_tokens, token_xyz,
                           anchor_indices, merge_assignment, num_patch_tokens=0, extra=None):
    """VisionZip: patch-token units, some kept whole and the rest folded in.

    Same schema as ``record_3dzip`` but over patch tokens instead of Stage-1
    voxels: ``anchor_indices[j]`` is the patch the j-th output token was built
    on, and ``merge_assignment[i]`` is the output token patch ``i`` reached.
    A merged patch's colour genuinely arrives at the LLM here -- VisionZip adds
    the group mean into its target -- so it is recorded as merged, not unused.
    """
    cfg = config()
    if not should_record(cfg, scene_key, method):
        return None
    try:
        xyz = _numpy(token_xyz, np.float64)
        anchors = _numpy(anchor_indices, np.int64).reshape(-1)
        assignment = _numpy(merge_assignment, np.int64).reshape(-1)
        selection = build_selection(
            scene_id=scene_id_from_key(scene_key), method=method, voxel_size=voxel_size,
            target_tokens=int(anchors.shape[0]), unit_xyz=xyz, anchor_indices=anchors,
            merge_assignment=assignment,
            unit_voxel=voxel_grid_from_xyz(token_xyz, voxel_size),
            num_patch_tokens=num_patch_tokens, source='forward',
            extra={key: _numpy(value) if hasattr(value, 'detach') else np.asarray(value)
                   for key, value in (extra or {}).items()})
        if int(target_tokens) != selection.num_final_tokens:
            selection.extra['requested_target_tokens'] = np.array(int(target_tokens))
        return _write(selection, cfg, scene_key)
    except Exception as exc:
        print(f'[TokenSel][WARN] {method} export failed: {exc}', flush=True)
        return None


def record_selection_only(scene_key, method, voxel_size, target_tokens, token_xyz,
                          kept_indices, num_patch_tokens=0, extra=None):
    """DTC / VisPruner / DivPrune: survivors are whole patch tokens, the rest are dropped.

    No feature is folded into a survivor, so no unit is ever recorded as
    merged; the visualization of these methods is exactly their K kept tokens.
    """
    cfg = config()
    if not should_record(cfg, scene_key, method):
        return None
    try:
        xyz = _numpy(token_xyz, np.float64)
        kept = _numpy(kept_indices, np.int64).reshape(-1)
        selection = build_selection(
            scene_id=scene_id_from_key(scene_key), method=method, voxel_size=voxel_size,
            target_tokens=int(kept.shape[0]), unit_xyz=xyz, anchor_indices=kept,
            unit_voxel=voxel_grid_from_xyz(token_xyz, voxel_size),
            num_patch_tokens=num_patch_tokens, source='forward',
            extra={key: _numpy(value) if hasattr(value, 'detach') else np.asarray(value)
                   for key, value in (extra or {}).items()})
        if int(target_tokens) != selection.num_final_tokens:
            selection.extra['requested_target_tokens'] = np.array(int(target_tokens))
        return _write(selection, cfg, scene_key)
    except Exception as exc:
        print(f'[TokenSel][WARN] {method} export failed: {exc}', flush=True)
        return None


def record_full(scene_key, voxel_size, token_xyz, point_to_voxel, voxel_xyz,
                num_patch_tokens=0):
    """Uncompressed LLaVA-3D: every Stage-1 voxel reaches the LLM as a token.

    The reference the compression methods are drawn against, so every unit is
    an anchor and K is simply the scene's voxel count.
    """
    cfg = config()
    if not should_record(cfg, scene_key, 'full'):
        return None
    try:
        xyz = _numpy(voxel_xyz, np.float64)
        n_units = xyz.shape[0]
        selection = build_selection(
            scene_id=scene_id_from_key(scene_key), method='full', voxel_size=voxel_size,
            target_tokens=n_units, unit_xyz=xyz,
            anchor_indices=np.arange(n_units, dtype=np.int64),
            unit_voxel=voxel_grid_per_unit(token_xyz, point_to_voxel, voxel_size, n_units),
            num_patch_tokens=num_patch_tokens, source='forward')
        return _write(selection, cfg, scene_key)
    except Exception as exc:
        print(f'[TokenSel][WARN] full export failed: {exc}', flush=True)
        return None


def record_llm_prune(scene_key, pruner, video_tower, voxel_size=None):
    """FastV / SparseVLM: pruning that happened inside the decoder.

    These two do not touch the vision path, so there is nothing to record while
    the tower runs -- the decision is made later, from the text's attention, and
    it is made again for every question. The evaluators therefore call this
    right after ``generate``, and the units are the tokens that *entered* the
    LLM: LLaVA-3D's voxel-pooled baseline tokens, whose XYZ the tower left in
    ``_voxel_xyz``. That also means a per-question run genuinely differs from
    question to question here, which is not true of any vision-path method.

    FastV only drops, so its record has anchors and unused units. SparseVLM
    also recycles: the tokens it clustered into a surviving centre are recorded
    as merged into it, because their content does reach the LLM.

    ``pruner`` and ``video_tower`` are duck-typed on purpose, to keep this
    package free of model imports.
    """
    cfg = config()
    if not cfg.enabled or pruner is None or video_tower is None:
        return None
    try:
        snapshot = pruner.selection_snapshot()
        if snapshot is None:
            return None
        method = snapshot['method']
        if cfg.once_per_scene and method not in _WARNED_ONCE_PER_SCENE:
            _WARNED_ONCE_PER_SCENE.add(method)
            print(f'[TokenSel] {method} prunes per question, but the export is in '
                  'once-per-scene mode: only the first question of each scene will '
                  'be recorded. Pass --token-selection-every-question to record '
                  'each one.', flush=True)
        if not should_record(cfg, scene_key, method):
            return None
        xyz_list = getattr(video_tower, '_voxel_xyz', None)
        if not xyz_list:
            print('[TokenSel][WARN] no _voxel_xyz on the video tower; '
                  f'cannot place {method}\'s tokens', flush=True)
            return None
        token_xyz = xyz_list[0]
        subsample = snapshot.get('visual_subsample')
        if subsample is not None and int(subsample.shape[0]) == snapshot['num_visual']:
            # A scene over LLaVA-3D's 2560 cap: the decoder saw a random subset
            # of the tower's tokens, in that order.
            token_xyz = token_xyz[subsample.to(token_xyz.device).long()]
        n_units = int(token_xyz.shape[0])
        if n_units != snapshot['num_visual']:
            # The decoder saw a different number of visual tokens than the
            # tower emitted -- batch > 1, or a truncated prompt. Refuse rather
            # than draw a mismatched record.
            print(f'[TokenSel][WARN] {method}: decoder saw '
                  f"{snapshot['num_visual']} visual tokens but the tower emitted "
                  f'{n_units}; skipping the record', flush=True)
            return None

        pre = getattr(video_tower, '_pre_compress_xyz', None)
        num_patch_tokens = int(pre[0].shape[0]) if pre else 0
        # Not display_voxel_size(): these units really are voxels -- the
        # LLaVA-3D baseline pooled them on this grid before the LLM saw them --
        # so the record states the grid the forward pass used. Re-rounding a
        # voxel's centroid lands back in that voxel: every member shares one
        # round(xyz / size), so the mean stays inside the same cell.
        size = float(voxel_size) if voxel_size else display_voxel_size()
        unit_xyz = _numpy(token_xyz, np.float64)
        unit_voxel = voxel_grid_from_xyz(token_xyz, size)
        common = {'stage_counts': np.asarray(snapshot['stage_counts'], dtype=np.int64),
                  'prune_layers': np.asarray(snapshot['prune_layers'], dtype=np.int64),
                  'num_visual_entering': np.array(snapshot['num_visual'], dtype=np.int64)}

        def build(name, kept_local, pairs, first_layer, last_layer):
            kept = _numpy(kept_local, np.int64).reshape(-1)
            assignment = np.full(n_units, -1, dtype=np.int64)
            assignment[kept] = np.arange(kept.shape[0], dtype=np.int64)
            slot_of = {int(unit): int(slot) for slot, unit in enumerate(kept)}
            for member, centre in pairs:
                if 0 <= member < n_units and centre in slot_of:
                    assignment[member] = slot_of[centre]
            extra = dict(common)
            if first_layer is not None:
                extra['decoder_layers'] = np.array([first_layer, last_layer], dtype=np.int64)
            return build_selection(
                scene_id=scene_id_from_key(scene_key), method=name, voxel_size=size,
                target_tokens=int(kept.shape[0]), unit_xyz=unit_xyz,
                anchor_indices=kept, merge_assignment=assignment, unit_voxel=unit_voxel,
                num_patch_tokens=num_patch_tokens, source='decoder', extra=extra)

        # SparseVLM prunes three times, so the final set is only what layers
        # 16-31 see. Each earlier stage gets a record of its own, named for the
        # decoder layers that attend to it: sparsevlm_layers3-6, ..._layers7-15.
        if _flag('TOKEN_SEL_LLM_STAGES', '1'):
            for stage in snapshot.get('stages', ()):
                name = f"{method}_layers{stage['first_layer']}-{stage['last_layer']}"
                if should_record(cfg, scene_key, name):
                    _write(build(name, stage['kept_local'], stage['recycled_pairs'],
                                 stage['first_layer'], stage['last_layer']), cfg, scene_key)
        first = snapshot.get('final_first_layer')
        num_layers = getattr(pruner, 'num_layers', None)
        selection = build(method, snapshot['kept_local'], snapshot['recycled_pairs'],
                          first, None if first is None else (num_layers or 0) - 1)
        return _write(selection, cfg, scene_key)
    except Exception as exc:                 # visualization must never break inference
        print(f'[TokenSel][WARN] llm-prune export failed: {exc}', flush=True)
        return None


# -- eval-script plumbing ------------------------------------------------
def add_selection_export_args(parser):
    """CLI flags for the benchmark evaluators. Off unless --save-token-selection."""
    group = parser.add_argument_group('Fig.4 token-selection export')
    group.add_argument('--save-token-selection', action='store_true',
                       help='Record which 3D regions the compression kept, for '
                            'visualization/visualize_fig4.py. Does not change model output.')
    group.add_argument('--token-selection-dir', type=str, default=None,
                       help='Output root (default pred/token_selection).')
    group.add_argument('--token-selection-tag', type=str, default=None,
                       help='Run label stored inside each record.')
    group.add_argument('--token-selection-every-question', dest='token_selection_once_per_scene',
                       action='store_false', default=True,
                       help='Record every question instead of the first per scene.')
    group.add_argument('--token-selection-max-scenes', type=int, default=None,
                       help='Stop recording after this many scenes. 0 = no limit.')
    return parser


def apply_selection_export_args(args) -> bool:
    """Translate the flags into TOKEN_SEL_*; an explicit env var still wins."""
    if not getattr(args, 'save_token_selection', False):
        return False
    os.environ['TOKEN_SEL'] = '1'
    if getattr(args, 'token_selection_dir', None):
        os.environ['TOKEN_SEL_DIR'] = args.token_selection_dir
    if getattr(args, 'token_selection_tag', None):
        os.environ['TOKEN_SEL_TAG'] = args.token_selection_tag
    if getattr(args, 'token_selection_max_scenes', None) is not None:
        os.environ['TOKEN_SEL_MAX_SCENES'] = str(args.token_selection_max_scenes)
    os.environ['TOKEN_SEL_ONCE_PER_SCENE'] = (
        '1' if getattr(args, 'token_selection_once_per_scene', True) else '0')
    return True
