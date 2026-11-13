import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .video_processor import RGBDVideoProcessor
from .spatial_aware_module import SpatialAwareModule
from .unproject import backprojector_dataloader, voxelize, adaptive_voxelize
from .dtc import dynamic_token_compression
# GT Query-Relevant Object Oracle (diagnostic upper bound, not a method).
# Everything oracle-specific lives in gt_oracle.py; this import plus the small
# hooks in the 3DZip branch below are the only entry points.
from .gt_oracle import ORACLE_POOLING
from .vispruner import apply_vispruner_to_tokens, vispruner_select_indices
from .segpruner import apply_segpruner_to_tokens, segpruner_select_indices
from .divprune import (apply_divprune_to_tokens, divprune_score_features,
                       divprune_select_indices, resolve_divprune_space)
from .visionzip import visionzip_compress
from .tosa import tosa_compress
from .zip3d import cosine_kernel, dpp_greedy_map, merge_into_anchors
# Any POOLING that is not a built-in branch names a method package at the repo
# root (one directory per version); see method_registry.py.
from .method_registry import is_external, load_method, method_flag
from torch_scatter import scatter_mean, scatter_add
from .position_encodings import PositionEmbeddingLearnedMLP



def _sample_id(key):
    """The scene / question the evaluators published, when one is bound.

    Imported lazily so the vision path keeps no hard dependency on the
    decoder-side pruning module.
    """
    try:
        from ..language_model.llm_token_pruning import PRUNE_STATE
        return PRUNE_STATE.get(key)
    except Exception:
        return None


def _token_selection_export():
    """The Fig.4 selection exporter, imported only when TOKEN_SEL asks for it.

    It lives in the repo-root ``visualization/`` package rather than under
    ``llava/`` so that drawing figures never becomes a dependency of running
    the model. A missing or broken exporter must not take inference down with
    it, hence the swallowed import error.
    """
    try:
        import importlib
        import sys
        root = str(Path(__file__).resolve().parents[3])
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module('visualization.export_token_selection')
    except Exception as exc:  # visualization must never break inference
        print(f"[TokenSel][WARN] exporter unavailable: {exc}", flush=True)
        return None


# The pooling branches that replace LLaVA-3D's token construction with a
# compression method. ``voxelize`` is the baseline itself and ``3dzip``
# voxelizes as its own Stage 1, so neither is listed.
COMPRESSION_POOLINGS = ('dtc', 'vispruner', 'divprune', 'segpruner',
                        'visionzip', 'tosa')


def _compression_inputs(units, p2v, geo_features, clip_features, token_xyz,
                        attn=None, merge_metric=None, attn_pool='sum'):
    """The token sequence a compression method runs on, and its per-token inputs.

    ``units='voxel'`` (the default) pools everything onto LLaVA-3D's voxel grid
    first, so every method starts from the **voxelized 3D tokens the
    uncompressed model feeds the LLM** -- the baseline token construction the
    3DZip paper's comparison is defined against. FastV and SparseVLM already
    prune that sequence inside the decoder, so this puts the vision-path methods
    on the same footing: same inputs, same starting count, only the selection
    rule differs.

    ``units='patch'`` keeps the raw V*576 CLIP patch grid instead, which is
    where these methods run in their own papers, and is the ablation row. It is
    a much harsher reduction (K out of ~13k rather than out of ~1.3k) and it
    takes VisionZip's contextual step far outside the regime its one
    hyperparameter was tuned for -- measured at 0.4562 against 0.5124 on
    SQA3D K=128.

    Features, keys and XYZ are pooled with ``scatter_mean``, so a voxel token
    is exactly the voxelize baseline's token.

    The CLS attention is pooled with ``COMPRESS_ATTN_POOL``, and ``sum`` is the
    default because ``mean`` is measurably biased here. Upstream's attention is
    a per-image distribution over 576 patches; a voxel is seen by a varying
    number of patches, so its mean is an average over a varying sample. A voxel
    seen once takes that patch's value verbatim while a voxel seen twenty times
    regresses to the middle, and the distribution is heavy-tailed -- so top-k by
    the mean over-selects thinly-observed voxels. Simulated on this split's
    shape (2257 voxels from ~14k patches, 18% of voxels seen by a single
    patch): the top-64 by mean is 36% single-patch voxels with a median of 2
    patches, against 1.6% and 19 for sum. Those are scene edges and noisy
    depth, not salient objects. ``sum`` measures the total attention mass a
    region received across the views it appears in, which is the scene-level
    reading of upstream's per-image saliency; ``max`` is the most-salient
    single view of it.

    Batching: the voxel count differs per scene, so pooling only supports
    B == 1, which is what the evaluators run. A larger batch keeps the patch
    grid and warns.
    """
    if units not in ('voxel', 'patch'):
        raise ValueError(f"Unknown COMPRESS_UNITS '{units}' (expected voxel or patch)")
    if units == 'patch' or p2v is None:
        return geo_features, clip_features, token_xyz, attn, merge_metric
    if geo_features.shape[0] != 1:
        print('[Compress][WARN] voxel units need batch size 1; keeping the patch '
              'grid for this batch.', flush=True)
        return geo_features, clip_features, token_xyz, attn, merge_metric

    def pool(tensor):
        if tensor is None:
            return None
        return scatter_mean(tensor[0], p2v, dim=0).unsqueeze(0)

    def pool_attn(tensor):
        if tensor is None:
            return None
        values = tensor[0].unsqueeze(-1)
        if attn_pool == 'mean':
            pooled = scatter_mean(values, p2v, dim=0)
        elif attn_pool == 'sum':
            pooled = scatter_add(values, p2v, dim=0)
        elif attn_pool == 'max':
            n_vox = int(p2v.max()) + 1
            pooled = values.new_full((n_vox, 1), float('-inf'))
            pooled.scatter_reduce_(0, p2v.unsqueeze(-1), values, reduce='amax')
        else:
            raise ValueError(
                f"Unknown COMPRESS_ATTN_POOL '{attn_pool}' (expected sum, mean or max)")
        return pooled.squeeze(-1).unsqueeze(0)

    return (pool(geo_features), pool(clip_features), pool(token_xyz),
            pool_attn(attn), pool(merge_metric))


class PromptEncoder(nn.Module):

    def __init__(self, latent_dim=4096):
        super(PromptEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.pos_emb3d = PositionEmbeddingLearnedMLP(dim=3, num_pos_feats=latent_dim)

    def encode_pe(self, xyz=None):
        return self.pos_emb3d(xyz)

    def forward(self, clicks):
        pos_embed = self.encode_pe(clicks)
        return pos_embed


class RGBDVideoTower(nn.Module):
    def __init__(self, vision_tower, video_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.num_frames = args.num_frames
        self.num_sample_tokens = args.num_sample_tokens
        self.pooling = os.environ.get('POOLING', 'voxelize')
        self.voxel_size = float(os.environ.get('VOXEL_SIZE', '0.2'))
        self.adaptive_ratio = float(os.environ.get('ADAPTIVE_RATIO', '0'))
        self.vision_tower_name = vision_tower
        self.video_tower_name = video_tower
        # A query-dependent method must never be served from the per-scene
        # feature cache (scene_cache.py reads this).
        self.query_dependent = method_flag(self.pooling, 'QUERY_DEPENDENT')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_video_tower', False):
            self.load_model()
        else:
            self.cfg_only = None

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        print(f"[VideoTower] num_frames={self.num_frames}", flush=True)
        self.video_processor = RGBDVideoProcessor(self.vision_tower_name, self.num_frames)
        if self.video_tower_name == 'SpatialAwareModule':
            self.video_tower = SpatialAwareModule()
        else:
            raise NotImplementedError

        self.prompt_encoder = PromptEncoder()
        self.is_loaded = True

    def forward(self, features, depths, poses, intrinsics, lengths=None, cls_attn=None, text_sim=None,
                vz_metric=None, projector=None, query_features=None, query_texts=None):
        """
        Args:
            - features: (B, V, 1024, 24, 24), image token features
            - depths: (B, V, H, W), depth images
            - poses: (B, V, 4, 4), camera-to-world poses
            - intrinsics: (B, V, 4, 4), camera intrinsics
            - lengths: (B,), view count per scene
            - vz_metric: (B, V, 576, head_dim), VisionZip merge metric; None otherwise
            - projector: the model's ``mm_projector``, which normally runs after
              this tower. DivPrune alone needs it, because upstream scores the
              tokens in the LLM's embedding space; see ``divprune.py``.
            - query_features: (B, V*H*W, C_clip), CLIP semantic patch tokens
              (last layer + post_layernorm) in the same (V, H, W) order as
              ``features``; None unless the method asks for them.
            - query_texts: list of B question strings (the raw user query);
              None when the caller does not pass one. Built-ins ignore both.

        Returns:
            - pooled_video_features: (Bn, 1024)
            - batch_offset: (B,) int32
        """
        B, V, C, H, W = features.shape
        _log = not getattr(self, '_logged_once', False)
        assert intrinsics.dim() == 4

        # Depth backprojection -> 3D world coordinates. The poses arriving here
        # are already multiplied by ScanNet's axisAlignment (see
        # video_processor), so every XYZ below -- and therefore every voxel the
        # Fig.4 export records -- is in the axis-aligned ScanNet frame the
        # <scene>_vh_clean_2.ply mesh can be moved into.
        selection_export = _token_selection_export() if os.environ.get('TOKEN_SEL', '0') not in ('0', 'false', 'False', '') else None

        feat_xyz, _dense_xyz = backprojector_dataloader([features.flatten(0, 1)], depths, poses, intrinsics)
        # Spatial-aware 3D position embedding
        video_features = self.video_tower([features.flatten(0, 1)], [feat_xyz.flatten(0, 1)], (B, V))[0]
        video_xyz = feat_xyz.reshape(B, V * H * W, 3)
        visual_2d_features = features.permute(0, 1, 3, 4, 2).reshape(B, V * H * W, C)
        if cls_attn is not None:
            vispruner_attn = cls_attn.reshape(B, V * H * W)
        else:
            vispruner_attn = None
        if vz_metric is not None:
            vz_metric = vz_metric.reshape(B, V * H * W, vz_metric.shape[-1])
        # The uncompressed token XYZ, kept for callers that report on what the
        # compression started from.
        self._pre_compress_xyz = [video_xyz[b].detach() for b in range(B)]
        if lengths is not None:
            lengths = lengths * H * W

        # --- LLaVA-3D's voxelization pooling, as the baseline token
        # --- construction every compression method starts from.
        #
        # The 3DZip paper's comparison replaces only the compression step: the
        # input, vision encoder, 3D token construction, LLM and evaluation all
        # stay LLaVA-3D's. So the sequence handed to a compression branch is the
        # voxelized 3D tokens, not the raw patch grid -- which is also where
        # FastV and SparseVLM prune, inside the decoder.
        #
        # From here to the end of the branch chain these five names refer to
        # that sequence. They are the per-patch tensors only under
        # COMPRESS_UNITS=patch, which is the ablation. ``_pre_compress_xyz``
        # above still records the patch grid, since it reports what the
        # compression started from.
        compress_units = os.environ.get('COMPRESS_UNITS', 'voxel')
        attn_pool = os.environ.get('COMPRESS_ATTN_POOL', 'sum')
        if self.pooling in COMPRESSION_POOLINGS:
            compress_p2v = (voxelize(feat_xyz, self.voxel_size)[0]
                            if compress_units == 'voxel' and self.voxel_size > 0 else None)
            (video_features, visual_2d_features, video_xyz,
             vispruner_attn, vz_metric) = _compression_inputs(
                compress_units, compress_p2v, video_features, visual_2d_features,
                video_xyz, attn=vispruner_attn, merge_metric=vz_metric,
                attn_pool=attn_pool)
            num_compress_tokens = video_features.shape[1]
            if _log:
                print(f'[Compress] {self.pooling} runs on {num_compress_tokens} '
                      f'{compress_units} tokens (patch grid was {V * H * W}), '
                      f'attn_pool={attn_pool}', flush=True)
        else:
            num_compress_tokens = V * H * W

        # Adaptive voxel size is consumed by the voxelize baseline. 3DZip and
        # DTC use ADAPTIVE_RATIO as an explicit target-token budget instead.
        if self.pooling == 'voxelize' and self.adaptive_ratio > 0:
            target = int(self.adaptive_ratio) if self.adaptive_ratio >= 1 else int(self.adaptive_ratio * V * H * W)
            p2v, found_vs = adaptive_voxelize(feat_xyz, target_tokens=target)
        else:
            p2v = None

        if self.pooling == 'voxelize':
            # Baseline: scatter_mean per voxel grid
            # The display grid is voxel_size only when voxel_size built p2v.
            p2v_is_fixed = p2v is None and self.voxel_size > 0
            if p2v is None:
                p2v = voxelize(feat_xyz, self.voxel_size)
            pooled_video_features = torch.cat([scatter_mean(video_features[b], p2v[b], dim=0) for b in range(B)])
            xyz_list = [scatter_mean(video_xyz[b], p2v[b], dim=0) for b in range(B)]
            batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
            # Fig.4 'full': the uncompressed baseline. Skipped under FastV /
            # SparseVLM, which also run on voxelize and record their own pruning.
            if (selection_export is not None and p2v_is_fixed
                    and os.environ.get('LLM_PRUNE', 'none') in ('', 'none')):
                selection_export.record_full(
                    scene_key=getattr(self, 'scene_key', None),
                    voxel_size=self.voxel_size, token_xyz=video_xyz[0],
                    point_to_voxel=p2v[0], voxel_xyz=xyz_list[0],
                    num_patch_tokens=V * H * W)

        elif self.pooling in ('3dzip', ORACLE_POOLING):
            # 3DZip: Voxelize + DPP greedy MAP selection (Cholesky) + cosine merge
            # ORACLE_POOLING runs the identical pipeline; only anchor selection
            # is overridden, and only when a GT context has been set.
            gt_oracle = getattr(self, '_gt_oracle', None)
            if gt_oracle is not None and getattr(gt_oracle, 'context', None) is None:
                gt_oracle = None   # no question bound -> stock 3DZip
            oracle_mode = self.pooling == ORACLE_POOLING
            if self.voxel_size > 0:
                p2v_fix = voxelize(feat_xyz, self.voxel_size)
            else:
                N_tok = V * H * W
                p2v_fix = torch.arange(N_tok, device=video_features.device).unsqueeze(0).expand(B, -1)
            MERGE_CUTOFF = float(os.environ.get('MERGE_CUTOFF', '5.0'))
            target_k = int(self.adaptive_ratio) if self.adaptive_ratio >= 1 else (int(self.adaptive_ratio * V * H * W) if self.adaptive_ratio > 0 else 0)
            pooled_list, xyz_list, counts = [], [], []
            self._last_anchor_indices = []
            for b in range(B):
                vox_feat = scatter_mean(video_features[b], p2v_fix[b], dim=0)  # (n_vox, C)
                vox_xyz = scatter_mean(video_xyz[b], p2v_fix[b], dim=0)        # (n_vox, 3)
                n_vox = vox_feat.shape[0]
                k = min(target_k, n_vox) if target_k > 0 else n_vox
                # GT-relevant voxels are computed in BOTH modes: the baseline
                # needs them for its GT-retention statistics.
                gt_targets = None
                if gt_oracle is not None:
                    if b > 0:
                        raise RuntimeError('GT oracle context is defined for a single question; '
                                           'batch size must be 1.')
                    gt_targets = gt_oracle.compute_voxel_targets(video_xyz[b], p2v_fix[b], n_vox)
                if k < n_vox and k >= 2:
                    # --- Stage 1a (oracle only): force GT-relevant voxels in ---
                    sel_idx = None
                    if oracle_mode and gt_oracle is not None and gt_targets is not None:
                        sel_idx = gt_oracle.select_anchors(vox_feat, gt_targets, k)
                    if sel_idx is not None:
                        assert sel_idx.numel() == k, (
                            f'GT oracle must keep the budget: {sel_idx.numel()} != {k}')
                        selected = sel_idx.tolist()
                    else:
                        # --- Stage 1: DPP greedy MAP selection (Cholesky-based) ---
                        L = cosine_kernel(vox_feat)  # cosine similarity kernel
                        sel_idx = dpp_greedy_map(L, k)

                    if gt_oracle is not None and b == 0:
                        gt_oracle.record_stats(sel_idx, gt_targets, k, n_vox,
                                               oracle_applied=oracle_mode and gt_targets is not None
                                               and gt_targets.n_gt_voxels > 0)

                    # --- Stage 2: Merge remaining voxels into nearest anchor ---
                    merged_feat, anchor_xyz, merge_trace = merge_into_anchors(
                        vox_feat, vox_xyz, sel_idx, k, self.voxel_size, MERGE_CUTOFF)

                    pooled_list.append(merged_feat)
                    xyz_list.append(anchor_xyz)
                    self._last_anchor_indices.append(sel_idx.detach())
                    if selection_export is not None and b == 0:
                        # Stage 1 voxels, the Stage 2 anchors and the Stage 3
                        # merges that passed the spatial cutoff -- the indices
                        # the forward pass just used, not a reconstruction.
                        selection_export.record_3dzip(
                            scene_key=getattr(self, 'scene_key', None),
                            voxel_size=self.voxel_size if self.voxel_size > 0
                            else selection_export.display_voxel_size(),
                            target_tokens=k, unit_xyz=vox_xyz, anchor_indices=sel_idx,
                            remain_idx=None if merge_trace is None else merge_trace[0],
                            nearest_anchor=None if merge_trace is None else merge_trace[1],
                            close_mask=None if merge_trace is None else merge_trace[2],
                            unit_voxel=selection_export.voxel_grid_per_unit(
                                video_xyz[b], p2v_fix[b], self.voxel_size, n_vox)
                            if self.voxel_size > 0 else None,
                            merge_cutoff=MERGE_CUTOFF, num_patch_tokens=V * H * W)
                else:
                    pooled_list.append(vox_feat)
                    xyz_list.append(vox_xyz)
                    self._last_anchor_indices.append(torch.arange(n_vox, device=vox_feat.device))
                    if gt_oracle is not None and b == 0:
                        # No selection happened: every voxel survives as a token.
                        gt_oracle.record_stats(
                            torch.arange(n_vox, device=vox_feat.device), gt_targets,
                            n_vox, n_vox, oracle_applied=False)
                    if selection_export is not None and b == 0:
                        # k >= n_vox: no DPP or merge ran, every voxel is a token.
                        selection_export.record_3dzip(
                            scene_key=getattr(self, 'scene_key', None),
                            voxel_size=self.voxel_size if self.voxel_size > 0
                            else selection_export.display_voxel_size(),
                            target_tokens=n_vox, unit_xyz=vox_xyz,
                            anchor_indices=torch.arange(n_vox, device=vox_feat.device),
                            unit_voxel=selection_export.voxel_grid_per_unit(
                                video_xyz[b], p2v_fix[b], self.voxel_size, n_vox)
                            if self.voxel_size > 0 else None,
                            merge_cutoff=MERGE_CUTOFF, num_patch_tokens=V * H * W)
                counts.append(pooled_list[-1].shape[0])

            pooled_video_features = torch.cat(pooled_list, dim=0)
            batch_offset = torch.tensor(counts, dtype=torch.int32, device=video_features.device).cumsum(0)

        elif self.pooling == 'dtc':
            # DTC baseline: dynamic voxel-constrained matching over the same
            # LLaVA-3D geometry-aware tokens used by voxelize/3dzip.
            target_k = int(self.adaptive_ratio) if self.adaptive_ratio >= 1 else (
                int(self.adaptive_ratio * num_compress_tokens)
                if self.adaptive_ratio > 0 else num_compress_tokens
            )
            exact_budget = os.environ.get('DTC_EXACT_BUDGET', '1') not in ('0', 'false', 'False')
            dtc_trace = {} if selection_export is not None else None
            pooled_video_features, xyz_list, batch_offset, retained_indices = dynamic_token_compression(
                video_features,
                video_xyz,
                target_tokens=target_k,
                provenance_out=dtc_trace,
                v_init=float(os.environ.get('DTC_V_INIT', '0.1')),
                delta_v=float(os.environ.get('DTC_DELTA_V', '0.02')),
                edge_ratio=float(os.environ.get('DTC_EDGE_RATIO', '0.4')),
                seed=int(os.environ.get('DTC_SEED', '0')),
                exact_budget=exact_budget,
                debug=os.environ.get('DTC_DEBUG', '0') not in ('0', 'false', 'False'),
                max_iter=int(os.environ.get('DTC_MAX_ITER', '200')),
            )
            self._dtc_retained_indices = retained_indices
            if selection_export is not None:
                # DTC keeps whole patch tokens and drops the rest: no voxel
                # aggregation, and no dropped token's feature reaches a
                # survivor. So the regions it can be credited with are exactly
                # its K kept tokens -- `represented_by` is stored alongside as
                # bookkeeping, not as evidence the LLM received.
                extra = {}
                if dtc_trace and dtc_trace.get('represented_by'):
                    extra['dtc_represented_by'] = dtc_trace['represented_by'][0]
                selection_export.record_selection_only(
                    scene_key=getattr(self, 'scene_key', None), method='dtc',
                    voxel_size=selection_export.display_voxel_size(self.voxel_size),
                    target_tokens=target_k, token_xyz=video_xyz[0],
                    kept_indices=retained_indices[0], num_patch_tokens=V * H * W,
                    extra=extra)

        elif self.pooling == 'vispruner':
            # VisPruner baseline: select visual tokens using 2D CLIP visual
            # attention/diversity, then pass the corresponding LLaVA-3D
            # geometry-aware features and XYZ to the LLM.
            target_k = int(os.environ.get('VISPRUNER_TOKENS', '0'))
            if target_k <= 0:
                target_k = int(self.adaptive_ratio) if self.adaptive_ratio >= 1 else (
                    int(self.adaptive_ratio * num_compress_tokens)
                    if self.adaptive_ratio > 0 else num_compress_tokens
                )
            vispruner_trace = {} if selection_export is not None else None
            selected_indices = vispruner_select_indices(
                visual_2d_features,
                vispruner_attn,
                target_tokens=target_k,
                provenance_out=vispruner_trace,
                important_ratio=float(os.environ.get('VISPRUNER_IMPORTANT_RATIO', '0.5')),
                debug=os.environ.get('VISPRUNER_DEBUG', '0') not in ('0', 'false', 'False'),
            )
            pooled_video_features, xyz_list, batch_offset = apply_vispruner_to_tokens(
                video_features,
                video_xyz,
                selected_indices,
            )
            self._vispruner_selected_indices = selected_indices
            if selection_export is not None:
                # VisPruner only selects 2D tokens; a kept token's feature is its own.
                extra = {}
                if vispruner_trace and vispruner_trace.get('selection_kind'):
                    extra['vispruner_selection_kind'] = vispruner_trace['selection_kind'][0]
                selection_export.record_selection_only(
                    scene_key=getattr(self, 'scene_key', None), method='vispruner',
                    voxel_size=selection_export.display_voxel_size(self.voxel_size),
                    target_tokens=target_k, token_xyz=video_xyz[0],
                    kept_indices=selected_indices[0], num_patch_tokens=V * H * W,
                    extra=extra)

        elif self.pooling == 'divprune':
            # DivPrune baseline: greedy max-min diversity (MMDP) over the
            # visual tokens. Selection only -- nothing merges, every kept token
            # is its own.
            target_k = int(os.environ.get('DIVPRUNE_TOKENS', '0'))
            if target_k <= 0:
                target_k = int(self.adaptive_ratio) if self.adaptive_ratio >= 1 else (
                    int(self.adaptive_ratio * num_compress_tokens)
                    if self.adaptive_ratio > 0 else num_compress_tokens
                )
            # Upstream measures its distances after mm_projector, on the
            # embeddings the LLM will read, so DIVPRUNE_SPACE defaults to
            # 'projected'. In LLaVA-3D that space carries the 3D positional
            # embedding, but that embedding is part of the backbone every
            # method here shares -- the rule DivPrune must not break is adding
            # a *separate* xyz term to its score, which it does not. 'clip'
            # (PE-free 2D patch features) and 'geo' (pre-projector) stay
            # available as the ablation. Either way this is a scoring space
            # only: the tokens handed on stay un-projected, because llava_arch
            # projects them right after.
            # DIVPRUNE_FEATURE_SPACE is the reproduction spec's name for the same
            # switch; DIVPRUNE_SPACE stays valid.
            space = os.environ.get('DIVPRUNE_FEATURE_SPACE',
                                   os.environ.get('DIVPRUNE_SPACE', 'projected'))
            space = resolve_divprune_space(space)
            score_features = divprune_score_features(
                video_features, clip_features=visual_2d_features, projector=projector,
                space=space, warn=_log)
            divprune_debug = os.environ.get(
                'DIVPRUNE_DEBUG', '0') not in ('0', 'false', 'False')
            divprune_trace = {} if (selection_export is not None or divprune_debug) else None
            selected_indices = divprune_select_indices(
                score_features,
                target_tokens=target_k,
                provenance_out=divprune_trace,
                precision=os.environ.get('DIVPRUNE_PRECISION', 'fp32'),
                order=os.environ.get('DIVPRUNE_ORDER', 'original'),
                scene_tag='',
                debug=False,          # the branch prints the full block below
            )
            pooled_video_features, xyz_list, batch_offset = apply_divprune_to_tokens(
                video_features,
                video_xyz,
                selected_indices,
                target_tokens=target_k,
            )
            if divprune_debug:
                print(f'  output feature shape           = {tuple(pooled_video_features.shape)}'
                      f' (projected to {projector[-1].out_features if hasattr(projector, "__getitem__") else "?"} by llava_arch)',
                      flush=True)
            self._divprune_selected_indices = selected_indices
            if divprune_debug:
                trace = (divprune_trace or {}).get('trace', [{}])[0]
                before = trace.get('order_before_sort', [])
                after = trace.get('order_after_sort', [])
                scene = getattr(self, 'scene_key', None)
                print(
                    '[DivPrune Debug]'
                    f"\n  scene_id                       = "
                    f"{_sample_id('scene_id') or (os.path.basename(str(scene)) if scene else '?')}"
                    f"\n  question_id                    = {_sample_id('question_id')}"
                    f"\n  raw visual tokens              = {V * H * W}"
                    f"\n  after fixed {self.voxel_size}m voxelization = {num_compress_tokens}"
                    f"\n  DivPrune feature space         = {space}"
                    f" ({'post_projector' if space == 'projected' else 'pre_projector' if space == 'geo' else space})"
                    f"\n  DivPrune feature dim           = {score_features.shape[-1]}"
                    f"\n  target K                       = {target_k}"
                    f"\n  selected K                     = {selected_indices[0].numel()}"
                    f"\n  adaptive voxelization          = False (only POOLING=voxelize uses it)"
                    f"\n  merge after pruning            = False (selection only)"
                    f"\n  precision / order              = "
                    f"{os.environ.get('DIVPRUNE_PRECISION', 'fp32')} / "
                    f"{os.environ.get('DIVPRUNE_ORDER', 'original')}"
                    f"\n  first selected token           = {trace.get('first_index')}"
                    f"\n  its nearest-neighbour distance = {trace.get('first_nearest_distance'):.4f}"
                    f"\n  indices before sorting         = {before[:8]} ... (n={len(before)})"
                    f"\n  indices after sorting          = {after[:8]} ... (n={len(after)})"
                    f"\n  min pairwise cosine distance   = {trace.get('min_pairwise_distance'):.4f}"
                    f"\n  mean pairwise cosine distance  = {trace.get('mean_pairwise_distance'):.4f}",
                    flush=True)
            if selection_export is not None:
                # Selection only: the regions DivPrune can be credited with are
                # exactly its K kept tokens. `greedy_rank` rides along as
                # bookkeeping -- the order the MMDP loop picked them in.
                extra = {}
                if divprune_trace and divprune_trace.get('greedy_rank'):
                    extra['divprune_greedy_rank'] = divprune_trace['greedy_rank'][0]
                selection_export.record_selection_only(
                    scene_key=getattr(self, 'scene_key', None), method='divprune',
                    voxel_size=selection_export.display_voxel_size(self.voxel_size),
                    target_tokens=target_k, token_xyz=video_xyz[0],
                    kept_indices=selected_indices[0], num_patch_tokens=V * H * W,
                    extra=extra)

        elif self.pooling == 'segpruner':
            # SeGPruner baseline: attention-salient tokens plus geometry-aware
            # FPS over the rest, then the matching LLaVA-3D geometry-aware
            # features and XYZ go to the LLM. Selection only -- nothing merges.
            # No Fig.4 export yet: TOKEN_SEL=1 records nothing for this branch.
            target_k = int(os.environ.get('SEGPRUNER_TOKENS', '0'))
            if target_k <= 0:
                target_k = int(self.adaptive_ratio) if self.adaptive_ratio >= 1 else (
                    int(self.adaptive_ratio * num_compress_tokens)
                    if self.adaptive_ratio > 0 else num_compress_tokens
                )
            selected_indices = segpruner_select_indices(
                visual_2d_features,
                vispruner_attn,
                video_xyz,
                target_tokens=target_k,
                important_ratio=float(os.environ.get('SEGPRUNER_IMPORTANT_RATIO', '0.5')),
                lam=float(os.environ.get('SEGPRUNER_LAMBDA', '0.5')),
                debug=os.environ.get('SEGPRUNER_DEBUG', '0') not in ('0', 'false', 'False'),
            )
            pooled_video_features, xyz_list, batch_offset = apply_segpruner_to_tokens(
                video_features,
                video_xyz,
                selected_indices,
            )
            self._segpruner_selected_indices = selected_indices

        elif self.pooling == 'visionzip':
            # VisionZip baseline: dominant tokens by CLS attention, plus
            # contextual tokens that fold the remainder into strided merge
            # targets by CLIP-key similarity.
            target_k = int(os.environ.get('VISIONZIP_TOKENS', '0'))
            if target_k <= 0:
                target_k = int(self.adaptive_ratio) if self.adaptive_ratio >= 1 else (
                    int(self.adaptive_ratio * num_compress_tokens)
                    if self.adaptive_ratio > 0 else num_compress_tokens
                )
            if vz_metric is None and _log:
                print('[VisionZip][WARN] no CLIP key metric captured; merging on '
                      'patch features instead. Was POOLING=visionzip set before '
                      'the model was loaded?', flush=True)
            # The sequence is already the voxelized one (see COMPRESS_UNITS
            # above). That matters most for this method: VisionZip's contextual
            # step is sized by a hyperparameter (10/64 of the budget) assuming
            # each merged token absorbs ~23 others -- 468 residual over 20
            # targets on one 576-patch image. On the voxelized sequence the
            # ratio is ~35-105; on the raw V*576 patch grid it is ~570, 24x
            # outside that regime. Measured on SQA3D K=128: 0.5124 voxel vs
            # 0.4562 patch, against the 3DZip paper's 0.515.
            visionzip_trace = {} if selection_export is not None else None
            pooled_video_features, xyz_list, batch_offset = visionzip_compress(
                video_features,
                vz_metric if vz_metric is not None else visual_2d_features,
                vispruner_attn,
                video_xyz,
                target_tokens=target_k,
                contextual_ratio=float(os.environ.get('VISIONZIP_CONTEXTUAL_RATIO', '0.15625')),
                debug=os.environ.get('VISIONZIP_DEBUG', '0') not in ('0', 'false', 'False'),
                provenance_out=visionzip_trace,
                # Merge the CLIP part and re-add one PE at the merged position,
                # so a contextual token carries a single positional embedding
                # like every other token. See visionzip.py.
                clip_features=visual_2d_features,
                pe_fn=self.video_tower.encode_pe,
                merge_space=os.environ.get('VISIONZIP_MERGE_SPACE', 'clip'),
            )
            if selection_export is not None and visionzip_trace:
                # Unlike the selection-only methods, VisionZip folds every
                # leftover token into a contextual target, so the record carries
                # a real merge relation and almost nothing stays unused. The
                # units are whatever VisionZip ran on -- voxels by default.
                selection_export.record_merge_selection(
                    scene_key=getattr(self, 'scene_key', None), method='visionzip',
                    voxel_size=selection_export.display_voxel_size(self.voxel_size),
                    target_tokens=target_k, token_xyz=video_xyz[0],
                    anchor_indices=visionzip_trace['anchor_indices'][0],
                    merge_assignment=visionzip_trace['merge_assignment'][0],
                    num_patch_tokens=V * H * W)

        elif self.pooling == 'tosa':
            # ToSA baseline: ToMe bipartite soft matching whose score mixes
            # CLIP-key similarity with similarity of a sinusoidal 3D position
            # code, iterated to the token budget.
            # No Fig.4 export yet: TOKEN_SEL=1 records nothing for this branch.
            target_k = int(os.environ.get('TOSA_TOKENS', '0'))
            if target_k <= 0:
                target_k = int(self.adaptive_ratio) if self.adaptive_ratio >= 1 else (
                    int(self.adaptive_ratio * num_compress_tokens)
                    if self.adaptive_ratio > 0 else num_compress_tokens
                )
            if vz_metric is None and _log:
                print('[ToSA][WARN] no CLIP key metric captured; merging on '
                      'patch features instead. Was POOLING=tosa set before '
                      'the model was loaded?', flush=True)
            pooled_video_features, xyz_list, batch_offset = tosa_compress(
                video_features,
                vz_metric if vz_metric is not None else visual_2d_features,
                video_xyz,
                target_tokens=target_k,
                alpha=float(os.environ.get('TOSA_ALPHA', '0.5')),
                alpha_style=os.environ.get('TOSA_ALPHA_STYLE', 'uniform'),
                pe_bins=int(os.environ.get('TOSA_PE_BINS', '27')),
                pe_channels=int(os.environ.get('TOSA_PE_CHANNELS', '24')),
                schedule=os.environ.get('TOSA_SCHEDULE', 'uniform'),
                rounds=int(os.environ.get('TOSA_ROUNDS', '26')),
                debug=os.environ.get('TOSA_DEBUG', '0') not in ('0', 'false', 'False'),
            )

        elif is_external(self.pooling):
            # A method package (one directory per version at the repo root).
            # It gets everything computed above and returns the same triple as
            # the built-in branches.
            ctx = dict(
                B=B, V=V, C=C, H=H, W=W,
                features=features, feat_xyz=feat_xyz,
                video_features=video_features, video_xyz=video_xyz,
                visual_2d_features=visual_2d_features, cls_attn=vispruner_attn,
                vz_metric=vz_metric, projector=projector,
                query_features=query_features, query_texts=query_texts,
                selection_export=selection_export, log=_log,
            )
            pooled_video_features, xyz_list, batch_offset = load_method(self.pooling).compress(self, ctx)

        else:
            raise NotImplementedError(f"Unknown pooling: '{self.pooling}'")

        self._voxel_xyz = xyz_list
        self._logged_once = True
        return pooled_video_features, batch_offset

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
