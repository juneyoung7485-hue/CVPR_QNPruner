"""GT Query-Relevant Object Oracle ("3DZip-GT-Oracle" / GT-Guided Anchor Oracle).

DIAGNOSTIC / UPPER-BOUND EXPERIMENT -- NOT a proposed method.

3DZip selects tokens from scene-level feature diversity alone, with no access to
the question. This module answers one question: if we *knew* which objects the
question is about, and forced their 3D evidence into the DPP anchor set at the
same token budget K, would QA accuracy improve?

To that end it uses ScanQA's annotated ``object_ids`` (the question's target
objects) -- never the answer text, and never the ground-truth answer. The GT is
used only to pick which voxels are protected during anchor selection.

Everything oracle-specific lives in this file. ``video_encoder.py`` only calls
into it when a context has been set, so deleting this file plus the small
``3dzip_gt_oracle`` hooks restores the stock 3DZip path. The same hooks can be
reused for DTC.

Coordinate frame
----------------
3DZip multiplies ScanNet camera poses by ``axis_align_matrix`` before
back-projecting depth, so ``video_xyz`` lives in the AXIS-ALIGNED ScanNet frame.
GT geometry must therefore also be axis-aligned (``*_aligned_vert.npy``, or the
raw mesh with ``axisAlignment`` from ``<scene>.txt`` applied here). Comparing
against the unaligned ``*_vert.npy`` / raw mesh would be wrong.

Instance-id convention (IMPORTANT)
----------------------------------
* Raw ``<scene>.aggregation.json``: ``segGroups[i]["objectId"]`` IS the ScanQA
  ``object_id``. Use it as-is, NO +1.
* ScanRefer/ScanQA preprocessed ``<scene>_ins_label.npy``: the preprocessing
  writes ``instance_id = objectId + 1`` (0 is reserved for unannotated), so the
  target ids must be ``object_id + 1``.
``SceneInstanceGT`` reports which convention a scene was loaded with and shifts
the query ids accordingly.

Thread/process safety
---------------------
The oracle context is per-tower mutable state (like the existing ``scene_key``),
so a tower must not be shared across threads or concurrently batched questions.
The eval loop sets the context immediately before ``model.generate`` and clears
it right after, and ``B > 1`` batches are rejected (only batch element 0 has a
defined question).
"""
from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean

ORACLE_POOLING = '3dzip_gt_oracle'
BASE_POOLING = '3dzip'
# v0_gt: keep only the voxel tokens in/around the question's GT object boxes.
# No DPP, no merge -- the GT box alone decides which voxels reach the LLM.
GT_BOX_POOLING = 'v0_gt'
GT_POOLINGS = (ORACLE_POOLING, GT_BOX_POOLING)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# ScanQA annotation (question -> target object ids)
# --------------------------------------------------------------------------
def load_scanqa_gt(path) -> Dict[str, dict]:
    """Index the original ScanQA annotation by ``question_id``.

    The 3DZip question file and the original ScanQA annotation share
    ``question_id``, so the join is exact. Only ``scene_id`` / ``object_ids`` /
    ``object_names`` are kept -- ``answers`` is deliberately dropped here so no
    answer text can reach the compression path.
    """
    with open(path, 'r') as f:
        raw = json.load(f)
    index: Dict[str, dict] = {}
    for item in raw:
        qid = item.get('question_id')
        if qid is None:
            continue
        # Field name differs across ScanQA/ScanRefer releases.
        object_ids = item.get('object_ids', item.get('object_id', []))
        if object_ids is None:
            object_ids = []
        if isinstance(object_ids, (int, np.integer)):
            object_ids = [int(object_ids)]
        object_names = item.get('object_names', item.get('object_name', []))
        if isinstance(object_names, str):
            object_names = [object_names]
        index[qid] = {
            'scene_id': item.get('scene_id'),
            'object_ids': [int(o) for o in object_ids],
            'object_names': list(object_names or []),
        }
    return index


# --------------------------------------------------------------------------
# ScanNet instance ground truth (axis-aligned points + instance ids)
# --------------------------------------------------------------------------
def _read_ply_vertices(path) -> np.ndarray:
    """Return (N, 3) float32 XYZ from a binary/ascii ScanNet ``.ply``.

    Small standalone reader: the container has neither plyfile nor open3d.
    """
    numpy_type = {
        'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
        'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
        'ushort': 'u2', 'uint16': 'u2', 'short': 'i2', 'int16': 'i2',
        'uint': 'u4', 'uint32': 'u4', 'int': 'i4', 'int32': 'i4',
    }
    with open(path, 'rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError(f'Not a PLY file: {path}')
        fmt, n_vertex, props, in_vertex = None, None, [], False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'Unterminated PLY header: {path}')
            tokens = line.decode('ascii', 'replace').strip().split()
            if not tokens:
                continue
            if tokens[0] == 'format':
                fmt = tokens[1]
            elif tokens[0] == 'element':
                in_vertex = tokens[1] == 'vertex'
                if in_vertex:
                    n_vertex = int(tokens[2])
            elif tokens[0] == 'property' and in_vertex:
                if tokens[1] == 'list':
                    raise ValueError('list property in vertex element')
                props.append((tokens[2], numpy_type[tokens[1]]))
            elif tokens[0] == 'end_header':
                break
        if fmt == 'binary_little_endian':
            dtype = np.dtype([(name, '<' + t) for name, t in props])
        elif fmt == 'binary_big_endian':
            dtype = np.dtype([(name, '>' + t) for name, t in props])
        elif fmt == 'ascii':
            data = np.loadtxt(f, max_rows=n_vertex)
            names = [p[0] for p in props]
            cols = [names.index(a) for a in ('x', 'y', 'z')]
            return np.ascontiguousarray(data[:, cols], dtype=np.float32)
        else:
            raise ValueError(f'Unsupported PLY format: {fmt}')
        verts = np.frombuffer(f.read(dtype.itemsize * n_vertex), dtype=dtype, count=n_vertex)
    return np.stack([verts['x'], verts['y'], verts['z']], axis=-1).astype(np.float32)


def _read_axis_align_matrix(meta_txt) -> np.ndarray:
    with open(meta_txt, 'r') as f:
        for line in f:
            if line.startswith('axisAlignment'):
                values = [float(v) for v in line.split('=')[1].strip().split()]
                return np.asarray(values, dtype=np.float64).reshape(4, 4)
    return np.eye(4, dtype=np.float64)


@dataclass
class SceneGT:
    """Axis-aligned GT points with per-point instance ids for one scene."""
    scene_id: str
    xyz: np.ndarray            # (M, 3) float32, axis-aligned ScanNet frame
    instance: np.ndarray       # (M,) int32, -1 = unannotated
    instance_id_offset: int    # 0 for raw aggregation, 1 for preprocessed npy
    source: str
    _tree: object = field(default=None, repr=False)

    @property
    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.xyz)
        return self._tree

    def target_instance_ids(self, object_ids: Sequence[int]) -> np.ndarray:
        """ScanQA ``object_id`` -> instance id under this scene's convention."""
        return np.asarray([int(o) + self.instance_id_offset for o in object_ids], dtype=np.int64)


class SceneInstanceGT:
    """Loads and caches per-scene GT geometry, preprocessed npy first.

    ``preproc_root``: ScanRefer/ScanQA ``scannet_data/`` with
    ``<scene>_aligned_vert.npy`` + ``<scene>_ins_label.npy`` (offset 1).
    ``raw_root``: ScanNet scene dirs with ``<scene>_vh_clean_2.ply``,
    ``<scene>_vh_clean_2.*.segs.json``, ``<scene>.aggregation.json`` and
    ``<scene>.txt`` (offset 0). The raw path applies ``axisAlignment`` here and
    caches the result as an ``.npz`` so the mesh is parsed once per scene.
    """

    def __init__(self, preproc_root=None, raw_root=None, cache_dir=None, capacity: int = 4):
        self.preproc_root = Path(preproc_root) if preproc_root else None
        self.raw_root = Path(raw_root) if raw_root else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.capacity = capacity
        self._store: "OrderedDict[str, SceneGT]" = OrderedDict()

    def get(self, scene_id: str) -> Optional[SceneGT]:
        if scene_id in self._store:
            self._store.move_to_end(scene_id)
            return self._store[scene_id]
        scene = self._load(scene_id)
        if scene is None:
            return None
        self._store[scene_id] = scene
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return scene

    def _load(self, scene_id: str) -> Optional[SceneGT]:
        if self.preproc_root is not None:
            vert = self.preproc_root / f'{scene_id}_aligned_vert.npy'
            ins = self.preproc_root / f'{scene_id}_ins_label.npy'
            if vert.exists() and ins.exists():
                xyz = np.load(vert)[:, :3].astype(np.float32)
                instance = np.load(ins).astype(np.int32)
                # ScanRefer/ScanQA preprocessing stores objectId + 1.
                return SceneGT(scene_id, xyz, instance, 1, f'preprocessed:{vert}')
        if self.raw_root is None:
            return None
        cached = None
        if self.cache_dir is not None:
            cached = self.cache_dir / f'{scene_id}_gt_aligned.npz'
            if cached.exists():
                data = np.load(cached)
                return SceneGT(scene_id, data['xyz'], data['instance'], 0, f'raw-cache:{cached}')
        scene_dir = self.raw_root / scene_id
        ply = scene_dir / f'{scene_id}_vh_clean_2.ply'
        agg = scene_dir / f'{scene_id}.aggregation.json'
        meta = scene_dir / f'{scene_id}.txt'
        segs = sorted(scene_dir.glob(f'{scene_id}_vh_clean_2*.segs.json'))
        if not (ply.exists() and agg.exists() and segs):
            return None

        xyz = _read_ply_vertices(ply)
        align = _read_axis_align_matrix(meta) if meta.exists() else np.eye(4)
        xyz = (xyz.astype(np.float64) @ align[:3, :3].T + align[:3, 3]).astype(np.float32)

        with open(segs[0], 'r') as f:
            seg_indices = np.asarray(json.load(f)['segIndices'], dtype=np.int64)
        with open(agg, 'r') as f:
            seg_groups = json.load(f)['segGroups']
        lut = np.full(int(seg_indices.max()) + 1, -1, dtype=np.int32)
        for group in seg_groups:
            # Raw aggregation objectId == ScanQA object_id: NO +1 here.
            lut[np.asarray(group['segments'], dtype=np.int64)] = int(group['objectId'])
        instance = lut[seg_indices]

        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached, xyz=xyz, instance=instance)
        return SceneGT(scene_id, xyz, instance, 0, f'raw:{ply}')


# --------------------------------------------------------------------------
# Oracle runtime attached to the video tower
# --------------------------------------------------------------------------
@dataclass
class VoxelTargets:
    voxel_mask: torch.Tensor        # (n_vox,) bool
    n_voxels: int
    n_gt_voxels: int
    n_gt_raw_tokens: int
    nn_dist_median: float
    nn_dist_p90: float
    target_instance_ids: List[int]


class GTOracleRuntime:
    """Per-question GT context + target-voxel computation + anchor selection.

    A single instance is attached to the video tower as ``_gt_oracle``. With no
    context set (or no GT for the current question) every entry point returns
    ``None`` and 3DZip runs exactly as before.
    """

    def __init__(self, scanqa_gt: Dict[str, dict], scene_gt: SceneInstanceGT,
                 nn_max_dist: Optional[float] = None, voxel_min_frac: Optional[float] = None,
                 max_anchor_frac: Optional[float] = None,
                 oracle_mode: bool = False, debug: bool = False):
        self.scanqa_gt = scanqa_gt
        self.scene_gt = scene_gt
        self.nn_max_dist = _env_float('GT_NN_MAX_DIST', 0.10) if nn_max_dist is None else float(nn_max_dist)
        self.voxel_min_frac = (_env_float('GT_VOXEL_MIN_FRAC', 0.25)
                               if voxel_min_frac is None else float(voxel_min_frac))
        # 1.0 = the plain oracle: GT voxels may take the whole budget (Case B).
        # Lower values reserve part of K for the stock diversity selection, which
        # keeps some scene context around the target object.
        self.max_anchor_frac = (_env_float('GT_MAX_ANCHOR_FRAC', 1.0)
                                if max_anchor_frac is None else float(max_anchor_frac))
        self.oracle_mode = oracle_mode
        self.debug = debug or os.environ.get('GT_ORACLE_DEBUG', '0') not in ('0', 'false', 'False')
        self.context: Optional[dict] = None
        self.last_stats: dict = {}
        # v0_gt visualization: when enabled, the forward leaves the arrays of
        # its selection here and the eval loop writes them out per question.
        self.dump_enabled = False
        self.last_dump: Optional[dict] = None
        self.align_warned: set = set()

    # -- context ----------------------------------------------------------
    def set_context(self, question_id, scene_id=None, object_ids=None, object_names=None):
        """Bind the question whose GT objects the next forward may use.

        Always overwrites, so a previous question's GT can never leak. When
        ``object_ids`` is omitted the ScanQA annotation is looked up by
        ``question_id``.
        """
        entry = self.scanqa_gt.get(question_id, {}) if self.scanqa_gt else {}
        ids = object_ids if object_ids is not None else entry.get('object_ids', [])
        names = object_names if object_names is not None else entry.get('object_names', [])
        scene = scene_id or entry.get('scene_id')
        self.context = {
            'question_id': question_id,
            'scene_id': scene,
            'object_ids': [int(o) for o in (ids or [])],
            'object_names': list(names or []),
            'gt_available': bool(entry) and bool(ids),
        }
        self.last_stats = {}
        self.last_dump = None
        return self.context

    def clear_context(self):
        self.context = None
        self.last_stats = {}
        self.last_dump = None

    # -- target voxels ----------------------------------------------------
    def compute_voxel_targets(self, video_xyz_b: torch.Tensor, p2v_b: torch.Tensor,
                              n_vox: int) -> Optional[VoxelTargets]:
        """Label each voxel as GT-relevant or not.

        ``video_xyz`` (the raw projected token points) is matched to the nearest
        axis-aligned GT vertex first; the per-point target flags are then pooled
        per voxel with the same ``p2v_fix`` 3DZip uses, so a voxel counts as GT
        only when at least ``voxel_min_frac`` of its points hit a target object.
        """
        ctx = self.context
        if not ctx or not ctx.get('gt_available'):
            return None
        scene = self.scene_gt.get(ctx['scene_id']) if ctx.get('scene_id') else None
        if scene is None:
            self.last_stats['gt_error'] = f"no scene GT for {ctx.get('scene_id')}"
            return None

        raw_xyz = video_xyz_b.detach().float().cpu().numpy()
        dist, nn_idx = scene.tree.query(raw_xyz, k=1, workers=-1)
        target_ids = scene.target_instance_ids(ctx['object_ids'])
        raw_instance = scene.instance[nn_idx]
        valid = dist <= self.nn_max_dist
        raw_is_target = valid & np.isin(raw_instance, target_ids)

        nn_median = float(np.median(dist))
        nn_p90 = float(np.percentile(dist, 90))
        # Sanity check: aligned GT and 3DZip token XYZ must overlap.
        if self.debug or nn_median > 0.5:
            print(f"[GT ALIGN CHECK] scene={scene.scene_id} median_nn_dist={nn_median:.4f} "
                  f"p90_nn_dist={nn_p90:.4f} source={scene.source}", flush=True)
        if nn_median > 0.5 and scene.scene_id not in self.align_warned:
            self.align_warned.add(scene.scene_id)
            print(f"[GT ORACLE][WARN] {scene.scene_id}: median NN distance {nn_median:.3f} m is large. "
                  f"Check that the GT point cloud is axis-ALIGNED (not *_vert.npy).", flush=True)

        is_target = torch.from_numpy(raw_is_target.astype(np.float32)).to(p2v_b.device)
        frac = scatter_mean(is_target, p2v_b, dim=0, dim_size=n_vox)
        voxel_mask = frac >= self.voxel_min_frac

        return VoxelTargets(
            voxel_mask=voxel_mask,
            n_voxels=int(n_vox),
            n_gt_voxels=int(voxel_mask.sum().item()),
            n_gt_raw_tokens=int(raw_is_target.sum()),
            nn_dist_median=nn_median,
            nn_dist_p90=nn_p90,
            target_instance_ids=[int(t) for t in target_ids],
        )

    # -- v0_gt: GT box token selection ------------------------------------
    def compute_gt_boxes(self, device=None) -> Optional[torch.Tensor]:
        """Axis-aligned boxes ``(n_obj, 2, 3)`` (min, max) of the question's GT objects.

        Built from the axis-aligned ScanNet vertices of each target instance, so
        the boxes live in the same frame as ``video_xyz``. Objects without any
        vertex are skipped; ``None`` when nothing is left.
        """
        ctx = self.context
        if not ctx or not ctx.get('gt_available'):
            return None
        scene = self.scene_gt.get(ctx['scene_id']) if ctx.get('scene_id') else None
        if scene is None:
            self.last_stats['gt_error'] = f"no scene GT for {ctx.get('scene_id')}"
            return None
        boxes = []
        for inst in scene.target_instance_ids(ctx['object_ids']):
            pts = scene.xyz[scene.instance == inst]
            if len(pts) > 0:
                boxes.append(np.stack([pts.min(0), pts.max(0)]))
        if not boxes:
            self.last_stats['gt_error'] = f"no GT vertices for {ctx['object_ids']}"
            return None
        return torch.from_numpy(np.stack(boxes).astype(np.float32)).to(device)

    @staticmethod
    def box_distance(xyz: torch.Tensor, boxes: torch.Tensor):
        """Per-point distance to the nearest box (0 inside) and to that box's centre."""
        xyz = xyz.float()[:, None, :]                              # (N, 1, 3)
        lo, hi = boxes[None, :, 0, :], boxes[None, :, 1, :]        # (1, n_obj, 3)
        outside = torch.clamp(lo - xyz, min=0) + torch.clamp(xyz - hi, min=0)
        dist = outside.norm(dim=-1)                                # (N, n_obj)
        box_dist, nearest = dist.min(dim=1)
        centres = boxes.mean(dim=1)                                # (n_obj, 3)
        centre_dist = (xyz[:, 0, :] - centres[nearest]).norm(dim=-1)
        return box_dist, centre_dist

    @classmethod
    def select_box_tokens(cls, vox_xyz: torch.Tensor, boxes: torch.Tensor, k: int,
                          margin: float) -> torch.Tensor:
        """Voxel indices in/around the GT boxes, nearest first.

        ``k > 0``: exactly ``min(k, n_vox)`` voxels, ranked by distance to the
        nearest box (inside = 0), ties inside a box broken by distance to that
        box's centre. A small object therefore pulls in its surroundings until
        the budget is spent; a large one keeps its core.
        ``k == 0``: every voxel within ``margin`` metres of a box (variable
        count), at least one.
        """
        box_dist, centre_dist = cls.box_distance(vox_xyz, boxes)
        score = box_dist + 1e-3 * centre_dist
        order = torch.argsort(score)
        if k > 0:
            return order[:min(k, order.numel())]
        n_near = max(1, int((box_dist <= margin).sum().item()))
        return order[:n_near]

    def record_dump(self, vox_xyz, vox_cell, sel_idx, targets: Optional[VoxelTargets],
                    boxes: Optional[torch.Tensor], voxel_size: float, k: int, margin: float):
        """Keep one question's v0_gt selection for ``save_dump`` (no-op unless enabled)."""
        if not self.dump_enabled:
            return
        ctx = self.context or {}
        n_vox = int(vox_xyz.shape[0])
        self.last_dump = {
            'question_id': str(ctx.get('question_id')),
            'scene_id': str(ctx.get('scene_id')),
            'object_ids': np.asarray(ctx.get('object_ids', []), dtype=np.int64),
            'object_names': np.asarray(ctx.get('object_names', []), dtype=str),
            'voxel_size': float(voxel_size),
            'target_tokens': int(k),
            'box_margin': float(margin),
            'fallback': boxes is None,
            'vox_xyz': vox_xyz.detach().float().cpu().numpy(),
            'vox_cell': vox_cell.astype(np.int64),
            'sel_idx': sel_idx.detach().cpu().numpy().astype(np.int64),
            'gt_voxel_mask': (targets.voxel_mask.detach().cpu().numpy() if targets is not None
                              else np.zeros(n_vox, dtype=bool)),
            'boxes': (boxes.detach().float().cpu().numpy() if boxes is not None
                      else np.zeros((0, 2, 3), dtype=np.float32)),
        }

    def save_dump(self, directory, **extra) -> Optional[str]:
        """Write the last dump as ``<directory>/<scene_id>/<question_id>.npz``."""
        if not self.last_dump:
            return None
        dump = dict(self.last_dump)
        dump.update({key: np.asarray(value, dtype=str) for key, value in extra.items()})
        path = Path(directory) / dump['scene_id'] / f"{dump['question_id']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **dump)
        return str(path)

    # -- anchor selection -------------------------------------------------
    @staticmethod
    def dpp_greedy_map(vox_feat: torch.Tensor, k: int) -> List[int]:
        """DPP greedy MAP (Cholesky), identical to the stock 3DZip selection.

        Kept as a separate copy so the baseline branch in ``video_encoder.py``
        stays untouched.
        """
        n_vox = vox_feat.shape[0]
        k = int(min(k, n_vox))
        feat_norm = F.normalize(vox_feat.float(), dim=-1)
        L = feat_norm @ feat_norm.T
        d = L.diag().clone()
        C = torch.zeros(k, n_vox, device=vox_feat.device, dtype=torch.float32)
        selected: List[int] = []
        for t in range(k):
            j = d.argmax().item()
            selected.append(j)
            if t < k - 1:
                c_j = L[j].clone()
                if t > 0:
                    c_j -= C[:t, j] @ C[:t]
                C[t] = c_j / torch.sqrt(d[j].clamp(min=1e-8))
                d -= C[t] ** 2
                d.clamp_(min=0)
            d[j] = -float('inf')
        return selected

    def select_anchors(self, vox_feat: torch.Tensor, targets: VoxelTargets,
                       k: int) -> Optional[torch.Tensor]:
        """GT-guided anchor selection returning EXACTLY ``k`` voxel indices.

        Case A (n_target < k): keep every GT voxel (ordered by DPP over the GT
        subset), then let the stock DPP fill the rest from non-GT voxels.
        Case B (n_target >= k): run the stock DPP inside the GT subset only.

        ``max_anchor_frac`` < 1.0 caps how much of K the GT may take, so the
        remainder still comes from the stock scene-diversity selection.
        """
        if targets is None or targets.n_gt_voxels == 0:
            return None
        device = vox_feat.device

        def take(pool: torch.Tensor, count: int) -> torch.Tensor:
            order = self.dpp_greedy_map(vox_feat[pool], count)
            return pool[torch.tensor(order, device=device)]

        target_idx = torch.where(targets.voxel_mask)[0]
        other_idx = torch.where(~targets.voxel_mask)[0]
        n_target = int(target_idx.numel())
        cap = k if self.max_anchor_frac >= 1.0 else max(1, int(round(self.max_anchor_frac * k)))
        n_take = min(n_target, cap)

        chosen = [take(target_idx, n_take)]
        remaining = k - n_take
        if remaining > 0:
            n_other = min(int(other_idx.numel()), remaining)
            if n_other > 0:
                chosen.append(take(other_idx, n_other))
                remaining -= n_other
            if remaining > 0:
                # Fewer non-GT voxels than the leftover budget: top up from the
                # GT voxels that the cap left out, so the budget stays exact.
                taken = torch.zeros(vox_feat.shape[0], dtype=torch.bool, device=device)
                taken[torch.cat(chosen)] = True
                leftover = target_idx[~taken[target_idx]]
                chosen.append(take(leftover, min(int(leftover.numel()), remaining)))
        return torch.cat(chosen)

    # -- stats ------------------------------------------------------------
    def record_stats(self, sel_idx: torch.Tensor, targets: Optional[VoxelTargets],
                     k: int, n_vox: int, oracle_applied: bool, extra: Optional[dict] = None):
        """Store per-question GT retention stats (computed in BOTH modes)."""
        ctx = self.context or {}
        stats = {
            'question_id': ctx.get('question_id'),
            'scene_id': ctx.get('scene_id'),
            'object_ids': ctx.get('object_ids', []),
            'object_names': ctx.get('object_names', []),
            'num_gt_objects': len(ctx.get('object_ids', []) or []),
            'token_budget': int(k),
            'n_voxels_before_dpp': int(n_vox),
            'oracle_mode': bool(self.oracle_mode),
            'oracle_applied': bool(oracle_applied),
            'gt_available': bool(ctx.get('gt_available', False)),
            'nn_max_dist': self.nn_max_dist,
            'voxel_min_frac': self.voxel_min_frac,
            'max_anchor_frac': self.max_anchor_frac,
        }
        if targets is None:
            stats.update(n_gt_voxels=0, n_gt_raw_tokens=0, n_gt_anchors=0,
                         gt_anchor_recall=0.0, gt_anchor_budget_ratio=0.0,
                         nn_dist_median=None, nn_dist_p90=None, target_instance_ids=[])
        else:
            n_gt_anchors = int(targets.voxel_mask[sel_idx].sum().item())
            stats.update(
                n_gt_voxels=targets.n_gt_voxels,
                n_gt_raw_tokens=targets.n_gt_raw_tokens,
                n_gt_anchors=n_gt_anchors,
                gt_anchor_recall=n_gt_anchors / max(targets.n_gt_voxels, 1),
                gt_anchor_budget_ratio=n_gt_anchors / max(k, 1),
                nn_dist_median=targets.nn_dist_median,
                nn_dist_p90=targets.nn_dist_p90,
                target_instance_ids=targets.target_instance_ids,
            )
        if extra:
            stats.update(extra)
        self.last_stats = stats
        if self.debug:
            print(f"[GT ORACLE] qid={stats['question_id']} scene={stats['scene_id']} "
                  f"object_ids={stats['object_ids']} object_names={stats['object_names']} "
                  f"target_instance_ids={stats['target_instance_ids']} "
                  f"n_voxels={stats['n_voxels_before_dpp']} n_gt_voxels={stats['n_gt_voxels']} "
                  f"gt_anchors={stats['n_gt_anchors']} recall={stats['gt_anchor_recall']:.3f} "
                  f"budget_ratio={stats['gt_anchor_budget_ratio']:.3f} K={stats['token_budget']} "
                  f"oracle_applied={stats['oracle_applied']}", flush=True)
        return stats


# --------------------------------------------------------------------------
# Wiring helpers used by the eval scripts
# --------------------------------------------------------------------------
def oracle_pooling_enabled(pooling: str) -> bool:
    return pooling in GT_POOLINGS


def attach_gt_oracle(video_tower, scanqa_gt_file, scannet_gt_root=None, scannet_raw_root=None,
                     cache_dir=None, nn_max_dist=None, voxel_min_frac=None,
                     max_anchor_frac=None, debug=False):
    """Attach a ``GTOracleRuntime`` to ``video_tower`` and add the setters.

    Works for both ``POOLING=3dzip`` (stats only, selection untouched) and
    ``POOLING=3dzip_gt_oracle`` (GT-guided anchors).
    """
    runtime = GTOracleRuntime(
        scanqa_gt=load_scanqa_gt(scanqa_gt_file),
        scene_gt=SceneInstanceGT(preproc_root=scannet_gt_root, raw_root=scannet_raw_root,
                                 cache_dir=cache_dir),
        nn_max_dist=nn_max_dist,
        voxel_min_frac=voxel_min_frac,
        max_anchor_frac=max_anchor_frac,
        oracle_mode=oracle_pooling_enabled(getattr(video_tower, 'pooling', '')),
        debug=debug,
    )
    video_tower._gt_oracle = runtime
    video_tower.set_gt_oracle_context = runtime.set_context
    video_tower.clear_gt_oracle_context = runtime.clear_context
    print(f"[GT ORACLE] attached: mode={'oracle' if runtime.oracle_mode else 'baseline-stats'}, "
          f"questions={len(runtime.scanqa_gt)}, nn_max_dist={runtime.nn_max_dist}, "
          f"voxel_min_frac={runtime.voxel_min_frac}, max_anchor_frac={runtime.max_anchor_frac}",
          flush=True)
    return runtime


def question_id_from_scanqa(question: dict):
    """3DZip question files reuse the ScanQA ``question_id`` verbatim."""
    return question.get('question_id')


_SCENE_RE = re.compile(r'scene\d{4}_\d{2}')


def scene_id_from_video(video_field: str) -> Optional[str]:
    match = _SCENE_RE.search(str(video_field or ''))
    return match.group(0) if match else None
