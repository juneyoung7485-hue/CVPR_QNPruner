import torch
import torch.nn.functional as F


def vispruner_select_indices(
    visual_features,
    visual_attn,
    target_tokens,
    important_ratio=0.5,
    debug=False,
    provenance_out=None,
):
    """
    VisPruner-style token selection for LLaVA-3D.

    Upstream picks tokens out of one image's 576 CLIP patches. Here the
    sequence is whatever ``COMPRESS_UNITS`` selected in ``video_encoder.py`` --
    by default LLaVA-3D's voxelized 3D tokens, since that is the baseline token
    construction the 3DZip paper's comparison replaces only the compression
    step of. The rule itself is unchanged and still reads 2D cues only: the CLS
    attention and the CLIP features, both pooled onto those tokens.

    Args:
        visual_features: [B, N, C] visual features, without the 3D positional
            embedding -- CLIP patch features, or their voxel means.
        visual_attn: [B, N] visual attention scores, usually CLIP CLS->patch,
            averaged over the patches of each token.
        target_tokens: number of visual tokens to keep per scene.
        important_ratio: fraction selected directly by attention.
        provenance_out: debug-only dict. When given, filled with
            ``selection_kind``: per scene, an [N] int8 tensor labelling each
            original token 0 = kept as attention-important, 1 = kept as diverse,
            -1 = pruned. VisPruner only *selects*, so a kept token's feature is
            its own -- no pruned token contributes to it.

    Returns:
        selected_indices: list of [K_b] original 2D token indices per scene.
    """
    if visual_features.dim() != 3:
        raise ValueError(f"VisPruner expects visual_features [B, N, C], got {tuple(visual_features.shape)}")
    if visual_attn is None:
        visual_attn = visual_features.float().norm(dim=-1)
    if visual_attn.dim() != 2:
        raise ValueError(f"VisPruner expects visual_attn [B, N], got {tuple(visual_attn.shape)}")
    if visual_features.shape[:2] != visual_attn.shape:
        raise ValueError(
            f"VisPruner feature/attention shape mismatch: "
            f"{tuple(visual_features.shape)} vs {tuple(visual_attn.shape)}"
        )

    batch_size, num_tokens, _ = visual_features.shape
    target_tokens = int(target_tokens)
    if target_tokens <= 0 or target_tokens > num_tokens:
        target_tokens = num_tokens

    important_ratio = float(important_ratio)
    important_ratio = min(max(important_ratio, 0.0), 1.0)
    important_token_num = min(target_tokens, int(target_tokens * important_ratio))
    diverse_token_num = target_tokens - important_token_num

    selected_per_batch = []
    feature_norm = F.normalize(visual_features.float(), dim=-1)

    for b in range(batch_size):
        token_order = torch.argsort(visual_attn[b].float(), descending=True)
        important_indices = token_order[:important_token_num]
        residual_indices = token_order[important_token_num:]

        # VisPruner removes duplicate-looking residual tokens by iterative
        # bipartite matching, leaving visually diverse residual tokens. This
        # follows upstream's loop exactly, including the detail that with an
        # odd residual count the extra even-position token is still scored
        # (``a`` holds ceil(R/2) rows, ``b`` floor(R/2)) and so can itself be
        # pruned.
        while diverse_token_num > 0 and residual_indices.numel() > diverse_token_num:
            residual_count = residual_indices.numel()
            prune_count = min(8, residual_count - diverse_token_num)
            if prune_count <= 0:
                break

            even_indices = residual_indices[::2]
            odd_indices = residual_indices[1::2]
            if odd_indices.numel() == 0:
                # Upstream would reduce over an empty dimension here; with one
                # residual token left there is nothing to match against anyway.
                break

            scores = feature_norm[b, even_indices] @ feature_norm[b, odd_indices].T
            duplicate_scores = scores.max(dim=-1).values

            keep_even_order = torch.argsort(duplicate_scores, descending=True)[prune_count:]
            keep_even = even_indices[keep_even_order]

            residual_indices = torch.cat([keep_even, odd_indices], dim=0)

        if diverse_token_num > 0:
            diverse_indices = residual_indices[:diverse_token_num]
            selected = torch.cat([important_indices, diverse_indices], dim=0)
        else:
            diverse_indices = residual_indices[:0]
            selected = important_indices

        selected = torch.sort(selected[:target_tokens]).values
        selected_per_batch.append(selected)

        if provenance_out is not None:
            kind = torch.full((num_tokens,), -1, dtype=torch.int8, device=selected.device)
            kind[important_indices] = 0
            kind[diverse_indices] = 1
            kept = torch.zeros(num_tokens, dtype=torch.bool, device=selected.device)
            kept[selected] = True
            kind[~kept] = -1                      # truncated by the target budget
            provenance_out.setdefault("selection_kind", []).append(kind)

        if debug:
            print(
                f"[VisPruner] batch={b} N={num_tokens} target={target_tokens} "
                f"important={important_indices.numel()} diverse={max(0, selected.numel() - important_indices.numel())} "
                f"kept={selected.numel()}",
                flush=True,
            )

    return selected_per_batch


def apply_vispruner_to_tokens(features, xyz, selected_indices):
    """Gather [B, N, C] features and [B, N, 3] xyz by per-scene indices."""
    kept_features = []
    kept_xyz = []
    counts = []

    for b, indices in enumerate(selected_indices):
        kept_features.append(features[b, indices])
        kept_xyz.append(xyz[b, indices])
        counts.append(int(indices.numel()))

    output_features = torch.cat(kept_features, dim=0)
    batch_offset = torch.tensor(counts, dtype=torch.int32, device=features.device).cumsum(0)
    return output_features, kept_xyz, batch_offset
