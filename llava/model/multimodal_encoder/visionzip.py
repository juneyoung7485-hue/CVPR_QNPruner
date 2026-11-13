"""VisionZip adapted to LLaVA-3D's multi-view geometry-aware tokens.

Upstream: ``no_query_based/2d/VisionZip``, whose
``visionzip/clip_encoder.py::CLIPVisionTower_VisionZip.forward`` is the method:

  1. **Dominant tokens** -- the top ``N_d`` patches by CLS->patch attention,
     taken from the penultimate CLIP layer.
  2. **Contextual tokens** -- of everything left over, every ``step``-th token
     is made a merge target; each remaining token is assigned to its most
     similar target by *key* similarity, and the mean of an assignment group is
     **added** to its target (``target + mean(assigned)``, upstream's residual
     form, not an average).
  3. The two sets are concatenated and handed to the LLM.

The port keeps that arithmetic and changes only what the host model forces:

* **Metric.** Upstream patches ``CLIPAttention.forward`` to leak the
  head-averaged key states of layer ``-2`` and merges by their cosine
  similarity. Monkey-patching transformers is avoidable here: ``CLIPVisionTower``
  hooks the same layer's ``k_proj`` and exposes the identical tensor as
  ``_vz_metric`` (see ``clip_encoder.py``). If the hook produced nothing, the
  CLIP patch features stand in and a warning is printed.
* **What gets merged.** The *decision* is made on 2D CLIP keys, exactly as
  upstream. The *values* are the split VisPruner uses in this repo: the
  decision comes from 2D, the tokens handed on are LLaVA-3D's.

  The sum needs one extra step here, and it is the one thing in this port that
  changes what the LLM reads. Upstream adds ``mean(group)`` into a CLIP hidden
  state. LLaVA-3D's tokens are ``clip + PE(xyz)`` -- ``SpatialAwareModule``
  adds the 3D positional embedding -- so summing *those* gives a contextual
  token carrying ``PE_target + mean(PE_group)``, two positional embeddings
  instead of one. Measured on a stand-in PE of the same shape: the positional
  part comes out **1.95x** a single embedding, and the whole token **1.50x** a
  dominant one. The *direction* is largely fine -- that stand-in put the
  blended code at cosine 0.988 from the centroid's own code, since one MLP's
  outputs are strongly correlated across positions -- so this is a magnitude
  and distribution problem, not a scrambled position. It still means ~15% of
  the budget arrives at a scale the LLM never saw in training.

  Nothing upstream has this problem: single-image LLaVA has no additive
  positional embedding on its visual tokens. Nor does any other merging method
  here -- voxelize, 3DZip and ToSA all *average*, which keeps one PE at unit
  magnitude.

  So ``VISIONZIP_MERGE_SPACE=clip`` (the default) runs upstream's arithmetic on
  the CLIP features and adds a single PE at the merged token's own centroid,
  which is the token layout the LLM was trained on. ``geo`` is the literal sum
  on geometry-aware tokens, kept because it is what a naive port does and the
  difference is worth an ablation row. Only the ~15% of the budget that is
  contextual differs; dominant tokens are untouched either way.
* **Merged XYZ.** A contextual token is several patches in one, so it is placed
  at their mean position, matching what ``scatter_mean`` does for the voxelize
  baseline. Dominant tokens keep their own XYZ.
* **Scene-level budget.** Upstream compresses one image at a time. LLaVA-3D
  back-projects all views into one world frame before compression, so the
  budget is spent over the whole scene -- the convention DTC, VisPruner, 3DZip
  and SeGPruner all follow here, and the only one under which a K-token
  comparison means anything.

Budget split: VisionZip's README runs ``visionzip(model, dominant=54,
contextual=10)``, and its published configurations scale that to 108+20 and
162+30 -- a constant 10/64 = 0.15625 contextual share, the default below.
Upstream spends one of those dominant slots on CLS (``apply_info`` is called
with ``dominant_num=dominant-1``); here all of them are patches, because a CLS
token has no 3D position to give the LLM.

``tests/test_upstream_equivalence.py`` checks this module against the upstream
code transcribed line-for-line, at all three published splits.
"""
import torch
import torch.nn.functional as F


def visionzip_compress(
    geo_features,
    visual_metric,
    visual_attn,
    token_xyz,
    target_tokens,
    contextual_ratio=0.15625,
    debug=False,
    provenance_out=None,
    clip_features=None,
    pe_fn=None,
    merge_space='clip',
):
    """Compress [B, N, C] geometry-aware tokens to ~``target_tokens`` per scene.

    Args:
        geo_features: [B, N, C] LLaVA-3D geometry-aware tokens (the values).
        visual_metric: [B, N, C_m] merge metric, normally the head-averaged
            ``k_proj`` output of CLIP layer -2. ``None`` falls back to
            ``geo_features``.
        visual_attn: [B, N] CLS->patch attention picking the dominant tokens.
            ``None`` falls back to feature norm.
        token_xyz: [B, N, 3] world-frame XYZ of each patch token.
        target_tokens: total tokens to keep per scene.
        contextual_ratio: share of the budget spent on merged contextual tokens.
        provenance_out: debug-only dict. When given, filled with
            ``anchor_indices`` and ``merge_assignment``, one entry per scene:
            the index relation the forward pass produced, for the Fig.4 export.
            ``anchor_indices[j]`` is the patch token that output token ``j`` was
            built on (dominant tokens first, then the contextual targets), and
            ``merge_assignment[i]`` is the output token patch ``i`` reached, or
            -1. Unlike the selection-only methods, a merged patch's colour
            genuinely reaches the LLM: VisionZip *adds* the group mean to its
            target, so nothing here is a nearest-neighbour guess.

    Returns:
        ``(pooled_features, xyz_list, batch_offset)`` in the layout the video
        tower returns: features concatenated over the batch, one XYZ tensor per
        scene, and the cumulative per-scene token counts.
    """
    if geo_features.dim() != 3:
        raise ValueError(f"VisionZip expects geo_features [B, N, C], got {tuple(geo_features.shape)}")
    if token_xyz.dim() != 3 or token_xyz.shape[-1] != 3:
        raise ValueError(f"VisionZip expects token_xyz [B, N, 3], got {tuple(token_xyz.shape)}")
    if geo_features.shape[:2] != token_xyz.shape[:2]:
        raise ValueError(
            f"VisionZip feature/xyz shape mismatch: "
            f"{tuple(geo_features.shape)} vs {tuple(token_xyz.shape)}"
        )
    if visual_metric is None:
        visual_metric = geo_features
    if visual_metric.shape[:2] != geo_features.shape[:2]:
        raise ValueError(
            f"VisionZip feature/metric shape mismatch: "
            f"{tuple(geo_features.shape)} vs {tuple(visual_metric.shape)}"
        )
    if visual_attn is None:
        visual_attn = visual_metric.float().norm(dim=-1)
    if visual_attn.shape != geo_features.shape[:2]:
        raise ValueError(
            f"VisionZip feature/attention shape mismatch: "
            f"{tuple(geo_features.shape)} vs {tuple(visual_attn.shape)}"
        )

    batch_size, num_tokens, _ = geo_features.shape
    target_tokens = int(target_tokens)
    if target_tokens <= 0 or target_tokens > num_tokens:
        target_tokens = num_tokens
    contextual_ratio = min(max(float(contextual_ratio), 0.0), 1.0)

    if merge_space not in ('clip', 'geo'):
        raise ValueError(f"Unknown VisionZip merge space '{merge_space}' (expected clip or geo)")
    if merge_space == 'clip' and (clip_features is None or pe_fn is None):
        print('[VisionZip][WARN] no CLIP features or position encoder passed; '
              'merging on geometry-aware tokens, which doubles the 3D positional '
              'embedding of every contextual token.', flush=True)
        merge_space = 'geo'

    pooled_list, xyz_list, counts = [], [], []
    for b in range(batch_size):
        feats, xyz = _visionzip_one_scene(
            geo_features[b], visual_metric[b], visual_attn[b], token_xyz[b],
            target_tokens, contextual_ratio, debug=debug, batch_index=b,
            provenance_out=provenance_out,
            clip_features=None if clip_features is None else clip_features[b],
            pe_fn=pe_fn, merge_space=merge_space,
        )
        pooled_list.append(feats)
        xyz_list.append(xyz)
        counts.append(feats.shape[0])

    pooled = torch.cat(pooled_list, dim=0)
    batch_offset = torch.tensor(counts, dtype=torch.int32, device=geo_features.device).cumsum(0)
    return pooled, xyz_list, batch_offset


def _record_visionzip_provenance(provenance_out, num_tokens, anchor_indices, merge_assignment):
    if provenance_out is None:
        return
    provenance_out.setdefault('anchor_indices', []).append(anchor_indices)
    provenance_out.setdefault('merge_assignment', []).append(merge_assignment)


def _visionzip_one_scene(geo_features, metric, attn, xyz, target_tokens,
                         contextual_ratio, debug=False, batch_index=0,
                         provenance_out=None, clip_features=None, pe_fn=None,
                         merge_space='clip'):
    num_tokens = geo_features.shape[0]
    contextual_num = min(int(round(target_tokens * contextual_ratio)), target_tokens)
    dominant_num = target_tokens - contextual_num

    # -- Dominant tokens: top-k by CLS->patch attention -------------------
    dominant_idx = torch.topk(attn.float(), k=dominant_num, dim=0).indices if dominant_num > 0 \
        else torch.empty(0, dtype=torch.long, device=geo_features.device)
    dominant_idx = torch.sort(dominant_idx).values

    keep_mask = torch.zeros(num_tokens, dtype=torch.bool, device=geo_features.device)
    keep_mask[dominant_idx] = True
    residual_idx = torch.where(~keep_mask)[0]
    num_residual = residual_idx.numel()
    contextual_num = min(contextual_num, num_residual)

    dominant_feat = geo_features[dominant_idx]
    dominant_xyz = xyz[dominant_idx]
    if contextual_num <= 0:
        if debug:
            print(f"[VisionZip] batch={batch_index} N={num_tokens} target={target_tokens} "
                  f"dominant={dominant_feat.shape[0]} contextual=0", flush=True)
        assignment = torch.full((num_tokens,), -1, dtype=torch.long, device=geo_features.device)
        assignment[dominant_idx] = torch.arange(dominant_idx.numel(), device=geo_features.device)
        _record_visionzip_provenance(provenance_out, num_tokens, dominant_idx, assignment)
        return dominant_feat, dominant_xyz

    # -- Contextual tokens: uniformly strided merge targets ---------------
    metric_norm = F.normalize(metric[residual_idx].float(), dim=-1)
    step = max(1, num_residual // contextual_num)
    target_local = torch.arange(0, num_residual, step, device=geo_features.device)[:contextual_num]

    merge_mask = torch.ones(num_residual, dtype=torch.bool, device=geo_features.device)
    merge_mask[target_local] = False
    merge_local = torch.where(merge_mask)[0]

    # What gets summed. Upstream adds into CLIP hidden states; LLaVA-3D's
    # tokens are ``clip + PE(xyz)``, so adding those instead would put a second
    # positional embedding inside every contextual token (see the module
    # docstring). ``merge_space='clip'`` sums the CLIP part and re-adds one
    # PE at the merged position, which is the token layout the LLM was trained
    # to read; ``'geo'`` is the literal sum, kept for the ablation.
    merge_source = geo_features if merge_space == 'geo' else clip_features
    target_feat = merge_source[residual_idx[target_local]]
    target_xyz = xyz[residual_idx[target_local]]

    if merge_local.numel() > 0:
        similarity = metric_norm[merge_local] @ metric_norm[target_local].T
        assign = similarity.argmax(dim=1)

        merge_feat = merge_source[residual_idx[merge_local]].float()
        merge_xyz = xyz[residual_idx[merge_local]].float()
        ones = torch.ones(merge_local.numel(), device=geo_features.device, dtype=torch.float32)

        counts = torch.zeros(contextual_num, device=geo_features.device, dtype=torch.float32)
        counts.index_add_(0, assign, ones)
        feat_sum = torch.zeros(contextual_num, merge_feat.shape[-1],
                               device=geo_features.device, dtype=torch.float32)
        feat_sum.index_add_(0, assign, merge_feat)
        xyz_sum = torch.zeros(contextual_num, 3, device=geo_features.device, dtype=torch.float32)
        xyz_sum.index_add_(0, assign, merge_xyz)

        # Upstream adds the group mean to the target token; a target with no
        # assignments keeps its own feature (counts clamped, sum is zero).
        aggregated = feat_sum / counts.clamp(min=1).unsqueeze(-1)
        contextual_feat = target_feat + aggregated.to(target_feat.dtype)
        # The centroid of the patches the token stands for, target included.
        # Only ``geo`` reports it; see the xyz choice below.
        centroid_xyz = ((xyz_sum + target_xyz.float()) / (counts + 1.0).unsqueeze(-1)).to(target_xyz.dtype)
    else:
        contextual_feat = target_feat
        centroid_xyz = target_xyz

    if merge_space == 'clip':
        # The 2D method ran on 2D features; the 3D embedding goes back in at
        # the *same index* -- the target patch the token was built on, which is
        # the index upstream's `target + mean(group)` keeps. So the token is
        # that patch, enriched with its group, and it sits where that patch
        # sits. Its xyz follows the embedding for the same reason: the two must
        # agree, or the position the LLM reads and the position the record
        # draws are different places.
        contextual_xyz = target_xyz
        contextual_feat = contextual_feat + pe_fn(
            contextual_xyz.to(geo_features.dtype)).to(contextual_feat.dtype)
    else:
        contextual_xyz = centroid_xyz

    if debug:
        # ``absorbed`` is the diagnostic that matters for this port: upstream
        # sized the contextual step for ~23 tokens per merged token. Far above
        # that and each one stops being a local summary.
        absorbed = merge_local.numel() / max(1, contextual_feat.shape[0])
        print(f"[VisionZip] batch={batch_index} N={num_tokens} target={target_tokens} "
              f"dominant={dominant_feat.shape[0]} contextual={contextual_feat.shape[0]} "
              f"merged={merge_local.numel()} absorbed_per_contextual={absorbed:.1f} "
              f"(upstream ~23)", flush=True)

    if provenance_out is not None:
        # Output order is [dominant..., contextual...], so slot j < D is
        # dominant_idx[j] and slot D + i is the i-th contextual target.
        device = geo_features.device
        num_dominant = dominant_idx.numel()
        target_original = residual_idx[target_local]
        anchor_indices = torch.cat([dominant_idx, target_original], dim=0)
        assignment = torch.full((num_tokens,), -1, dtype=torch.long, device=device)
        assignment[dominant_idx] = torch.arange(num_dominant, device=device)
        assignment[target_original] = num_dominant + torch.arange(
            target_original.numel(), device=device)
        if merge_local.numel() > 0:
            assignment[residual_idx[merge_local]] = num_dominant + assign
        _record_visionzip_provenance(provenance_out, num_tokens, anchor_indices, assignment)

    return (torch.cat([dominant_feat, contextual_feat], dim=0),
            torch.cat([dominant_xyz, contextual_xyz], dim=0))
