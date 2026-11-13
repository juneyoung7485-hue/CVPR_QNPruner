"""DTC -- Voxel-based Dynamic Token Compression, for LLaVA-3D tokens.

Paper: Huang et al., *Zero-shot 3D Question Answering via Voxel-based Dynamic
Token Compression*, CVPR 2025.
https://openaccess.thecvf.com/content/CVPR2025/html/Huang_Zero-shot_3D_Question_Answering_via_Voxel-based_Dynamic_Token_Compression_CVPR_2025_paper.html

No reference implementation was released, so this follows the paper's Sec. 3.2,
which specifies the iteration as five steps:

  1. Randomly partition the visual tokens within each voxel into two sets,
     A and B, of roughly equal size.
  2. Draw one edge from each token in A to its most similar token in B.
  3. Keep the most r% similar edges **in the whole 3D space** -- the paper is
     explicit that the ranking is global, not per voxel.
  4. Compress the tokens still connected by edges. "We simply keep one token
     from the token pair ... the one that comes first in 2D image order":
     a drop, not an average. Their Table 6 ablates this (keep-one 54.1 >
     weighted average 52.9 > average pooling 51.7 LLM-Match on OpenEQA).
  5. Increase the voxel size by delta_v and iterate, "until the number of
     visual tokens meets the desired amount".

Paper hyperparameters, which are this module's defaults: ``v_init`` 0.1 m,
``delta_v`` 0.02 m, ``edge_ratio`` 40%.

Two things the paper leaves open, decided here:

* **Exact budgets.** The paper stops once the count "meets the desired
  amount", so it lands near the target rather than on it. ``exact_budget``
  (``DTC_EXACT_BUDGET=1``, the default) caps each iteration's removals so the
  run finishes on exactly K tokens, which is what makes the K-token comparison
  against the other baselines in this repo well posed. Set it to 0 for the
  paper's own stopping rule.
* **Edge conflicts.** A token can be the survivor of one selected edge and the
  dropped end of another. The paper does not say; ``_drop_later_tokens`` skips
  such edges so the "keep the earlier token" rule is never violated.

"2D image order" maps onto LLaVA-3D's token order, which is (view, row,
column) flattened -- raster order within a view, view order across views --
so the survivor is the token with the smaller flat index. Under
``COMPRESS_UNITS=voxel`` the tokens are voxel-pooled first and that order
becomes the voxel id order, which is still the order the LLM reads them in.

**What the tokens are.** The 3DZip paper's comparison applies every method
after LLaVA-3D's voxelization pooling, so by default this runs on the
voxelized 3D tokens rather than the raw patch grid. One consequence worth
knowing: those tokens already sit on a 0.2 m grid, so DTC's own voxel size
starts *below* it -- at ``v_init`` 0.1 m each DTC voxel holds at most one
token and the first few iterations find no edges. The loop grows ``v`` by
``delta_v`` until it does, so the paper's hyperparameters are kept and the
only cost is a handful of no-op iterations before the first merge.
"""
import math
import warnings

import torch
import torch.nn.functional as F

from .unproject import voxelization


def compute_dtc_similarity(features_a, features_b):
    """Cosine similarity for DTC matching; kept separate for easy replacement."""
    feat_a = F.normalize(features_a.float(), dim=-1)
    feat_b = F.normalize(features_b.float(), dim=-1)
    return feat_a @ feat_b.T


def _flatten_xyz(xyz):
    if xyz.dim() == 5:
        bsz, views, height, width, _ = xyz.shape
        return xyz.reshape(bsz, views * height * width, 3)
    if xyz.dim() == 3:
        return xyz
    raise ValueError(f"DTC expects xyz with shape [B, N, 3] or [B, V, H, W, 3], got {tuple(xyz.shape)}")


def _make_generator(device, seed):
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _build_dtc_candidate_edges(cur_features, cur_xyz, voxel_size, generator):
    """Return DTC candidate edges as local indices into the current alive token set."""
    device = cur_features.device
    p2v = voxelization(cur_xyz.unsqueeze(0), voxel_size)[0]
    # One stable sort instead of a torch.where (= one GPU sync) per voxel: each
    # voxel's slice holds the same ascending indices torch.where returned, and the
    # voxels are visited in the same order, so the randperm stream is unchanged.
    order = torch.argsort(p2v, stable=True)
    occupied_voxels, voxel_counts = torch.unique_consecutive(p2v[order], return_counts=True)

    edge_src = []
    edge_dst = []
    edge_sim = []

    start = 0
    for count in voxel_counts.tolist():
        voxel_indices = order[start:start + count]
        start += count
        if count < 2:
            continue

        perm = voxel_indices[torch.randperm(voxel_indices.numel(), device=device, generator=generator)]
        half = perm.numel() // 2
        if half == 0 or half == perm.numel():
            continue

        group_a = perm[:half]
        group_b = perm[half:]
        if group_b.numel() == 0:
            continue

        sim = compute_dtc_similarity(cur_features[group_a], cur_features[group_b])
        best_sim, best_b = sim.max(dim=1)

        edge_src.append(group_a)
        edge_dst.append(group_b[best_b])
        edge_sim.append(best_sim)

    if not edge_sim:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        return empty_long, empty_long, empty_float, int(occupied_voxels.numel())

    return (
        torch.cat(edge_src, dim=0),
        torch.cat(edge_dst, dim=0),
        torch.cat(edge_sim, dim=0),
        int(occupied_voxels.numel()),
    )


def _select_dtc_edges(edge_src, edge_dst, edge_sim, edge_ratio):
    if edge_sim.numel() == 0:
        return edge_src, edge_dst, edge_sim

    n_select = max(1, int(math.ceil(edge_sim.numel() * edge_ratio)))
    n_select = min(n_select, edge_sim.numel())

    # IMPORTANT: DTC ranks edges globally across the whole scene,
    # not independently within each voxel.
    order = torch.argsort(edge_sim, descending=True)
    selected = order[:n_select]
    return edge_src[selected], edge_dst[selected], edge_sim[selected]


def _drop_later_tokens(selected_src, selected_dst, alive, max_remove=None):
    """Returns (dropped tokens, the surviving token each was matched against)."""
    drop_original = []
    keep_original = []
    drop_seen = set()

    alive_list = alive.tolist()     # one host copy, not two .item() syncs per edge
    for src_local, dst_local in zip(selected_src.tolist(), selected_dst.tolist()):
        src_orig = alive_list[src_local]
        dst_orig = alive_list[dst_local]
        keep_orig = min(src_orig, dst_orig)
        drop_orig = max(src_orig, dst_orig)

        # Edge conflicts can make a token both the earlier token in one selected
        # pair and the later token in another. Skip those conflicts so the DTC
        # "keep the earlier original-order token" rule is never violated.
        if keep_orig in drop_seen or drop_orig in drop_seen:
            continue

        drop_seen.add(drop_orig)
        drop_original.append(drop_orig)
        keep_original.append(keep_orig)
        if max_remove is not None and len(drop_original) >= max_remove:
            break

    return drop_original, keep_original


def _resolve_representatives(represented_by):
    """Path-compress the drop chains into the final surviving token per entry.

    ``represented_by[i] < i`` always holds (DTC keeps the earlier token), so the
    chains are acyclic and this terminates.
    """
    for i in range(represented_by.numel()):
        root = int(represented_by[i])
        while root != int(represented_by[root]):
            root = int(represented_by[root])
        represented_by[i] = root
    return represented_by


def dynamic_token_compression(
    features,
    xyz,
    target_tokens,
    v_init=0.1,
    delta_v=0.02,
    edge_ratio=0.4,
    seed=0,
    exact_budget=True,
    debug=False,
    max_iter=200,
    provenance_out=None,
):
    """
    Training-free Dynamic Token Compression (DTC) for LLaVA-3D tokens.

    Args:
        features: [B, N, D] geometry-aware visual tokens.
        xyz: [B, N, 3] or [B, V, H, W, 3] world-coordinate token positions.
        target_tokens: desired output tokens per scene for fixed-budget mode.
        v_init: initial DTC voxel size in meters, default 0.1.
        delta_v: voxel-size increase per iteration in meters, default 0.02.
        edge_ratio: global scene-level fraction of candidate edges to select.
        seed: DTC bipartition seed; uses a local torch.Generator.
        exact_budget: True for 3DZip fixed-budget comparison mode.
        debug: print per-iteration DTC diagnostics.
        max_iter: safety cap; voxel size keeps increasing until progress occurs.
        provenance_out: debug-only dict. When given, filled with
            ``represented_by``: per scene, an [N] tensor mapping every original
            token to the retained token it was dropped in favour of (itself if
            retained). DTC *drops* duplicates rather than averaging them, so this
            is a "stood in for" relation, not a feature contribution.

    Returns:
        compressed_features: concatenated [sum(K_b), D].
        compressed_xyz: list of [K_b, 3] tensors.
        batch_offset: int32 cumulative token counts.
        retained_indices: list of original-order retained token indices.
    """
    if features.dim() != 3:
        raise ValueError(f"DTC expects features with shape [B, N, D], got {tuple(features.shape)}")
    if not 0 < edge_ratio <= 1:
        raise ValueError(f"DTC edge_ratio must be in (0, 1], got {edge_ratio}")

    xyz_flat = _flatten_xyz(xyz)
    if features.shape[:2] != xyz_flat.shape[:2]:
        raise ValueError(
            f"DTC features/xyz token counts differ: features={tuple(features.shape)}, xyz={tuple(xyz_flat.shape)}"
        )

    batch_size, num_tokens, _ = features.shape
    device = features.device

    if target_tokens is None or int(target_tokens) <= 0:
        target_tokens = num_tokens
    target_tokens = int(target_tokens)
    if target_tokens > num_tokens:
        target_tokens = num_tokens

    all_features = []
    all_xyz = []
    retained_indices = []
    counts = []

    with torch.no_grad():
        for batch_idx in range(batch_size):
            feat = features[batch_idx]
            pos = xyz_flat[batch_idx]
            alive = torch.arange(num_tokens, dtype=torch.long, device=device)
            voxel_size = float(v_init)
            generator = _make_generator(device, int(seed) + batch_idx)
            represented_by = (torch.arange(num_tokens, dtype=torch.long, device=device)
                              if provenance_out is not None else None)

            for iteration in range(int(max_iter)):
                tokens_before = int(alive.numel())
                if tokens_before <= target_tokens:
                    break

                cur_features = feat[alive]
                cur_xyz = pos[alive]
                edge_src, edge_dst, edge_sim, num_voxels = _build_dtc_candidate_edges(
                    cur_features, cur_xyz, voxel_size, generator
                )
                selected_src, selected_dst, _ = _select_dtc_edges(edge_src, edge_dst, edge_sim, edge_ratio)

                remaining_to_remove = tokens_before - target_tokens if exact_budget else None
                drop_original, keep_original = _drop_later_tokens(
                    selected_src, selected_dst, alive, max_remove=remaining_to_remove
                )

                if drop_original:
                    drop_tensor = torch.tensor(drop_original, dtype=torch.long, device=device)
                    if represented_by is not None:
                        represented_by[drop_tensor] = torch.tensor(
                            keep_original, dtype=torch.long, device=device)
                    keep_mask = torch.ones(tokens_before, dtype=torch.bool, device=device)
                    keep_mask[torch.isin(alive, drop_tensor)] = False
                    alive = alive[keep_mask]

                if debug:
                    print(
                        "[DTC] "
                        f"iter={iteration} voxel={voxel_size:.2f} "
                        f"tokens_before={tokens_before} "
                        f"num_occupied_voxels={num_voxels} "
                        f"num_candidate_edges={edge_sim.numel()} "
                        f"num_selected_edges={selected_src.numel()} "
                        f"num_unique_removed={len(drop_original)} "
                        f"tokens_after={alive.numel()}",
                        flush=True,
                    )

                voxel_size += float(delta_v)

            if exact_budget and alive.numel() != target_tokens:
                warnings.warn(
                    "DTC could not reach the exact fixed token budget without arbitrary pruning: "
                    f"target={target_tokens}, final={alive.numel()}, batch={batch_idx}.",
                    RuntimeWarning,
                )

            alive, _ = torch.sort(alive)
            all_features.append(feat[alive])
            all_xyz.append(pos[alive])
            retained_indices.append(alive)
            counts.append(int(alive.numel()))
            if represented_by is not None:
                provenance_out.setdefault("represented_by", []).append(
                    _resolve_representatives(represented_by))

            if debug:
                print(f"[DTC] final tokens={alive.numel()}", flush=True)

    compressed_features = torch.cat(all_features, dim=0)
    batch_offset = torch.tensor(counts, dtype=torch.int32, device=device).cumsum(0)
    return compressed_features, all_xyz, batch_offset, retained_indices
