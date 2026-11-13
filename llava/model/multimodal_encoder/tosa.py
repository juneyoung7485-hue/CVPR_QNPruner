"""ToSA (IROS 2025) adapted to LLaVA-3D's multi-view geometry-aware tokens.

Upstream: ``no_query_based/3d/ToSA``, whose
``llava/model/multimodal_encoder/siglip_encoder_3D_tome.py`` is the method.
ToSA is ToMe (bipartite soft matching) with one change, in
``bipartite_soft_matching_w_depth``::

    scores = alpha * cos(key_states) + (1 - alpha) * cos(spatial_pe)

i.e. two tokens are merge candidates when they are *both* semantically and
spatially alike. The spatial term is a sinusoidal positional encoding of a
quantised 3D index, built in the run scripts as::

    pe = PositionalEncoding1D(channels=24) over a length-27 axis
    depth_state(patch) = concat(pe[x], pe[y], pe[z])        # 72 dims

with ``x, y`` the patch's grid position and ``z`` its mean depth quantised into
27 bins. Merging itself is stock ToMe: split the sequence into even/odd halves,
score every even token against every odd one, merge the ``r`` best-matching
pairs, and average by ``merge_wavg`` weighted by how many original tokens each
one already stands for.

What the port changes, and why
------------------------------
* **Where it runs.** Upstream merges ``r`` tokens *inside every SigLIP layer*,
  26 times, and re-derives the key metric from each layer. LLaVA-3D's CLIP
  tower has to emit a full 24x24 grid per view -- ``backprojector_dataloader``
  and ``SpatialAwareModule`` attach an XYZ to each grid cell -- so merging
  inside it would leave the surviving tokens with no 3D position to be given.
  The port therefore runs ToSA's merge rule *after* the views are fused, over
  the scene's V*H*W tokens, and **iterates it to the K-token budget** instead
  of once per layer: each round removes ``min(t - K, t // 2)`` tokens, so
  11520 -> 128 takes ~7 rounds. ToMe's 50%-per-round cap is kept.
* **Semantic metric.** Upstream uses the head-averaged key states of the layer
  it is merging in. There is one pass here, so the port uses the head-averaged
  ``k_proj`` output of CLIP layer -2 -- the same tensor VisionZip merges on,
  captured by the same hook in ``clip_encoder.py`` -- and re-merges it every
  round alongside the features, exactly as upstream re-merges ``depth_states``.
* **Spatial metric.** Upstream's ``x, y`` are *per-image* patch indices, so the
  same (x, y) in two different views encodes two unrelated places; that is
  sound for single-image ToSA and meaningless once views are fused. The port
  keeps the encoding (same sinusoidal table, same 27 bins, same 3x24 = 72 dims)
  and feeds it the world XYZ LLaVA-3D already back-projected, quantised per
  axis over the scene's extent. This is the same information ToSA wants, in the
  one frame where it is comparable across views.
* **Merged XYZ.** ``merge_wavg``-weighted mean of the members, matching what
  ``scatter_mean`` gives the voxelize baseline. Reported only; it never reaches
  the LLM.
* **Metric widths.** Upstream asserts its two metrics have equal shape, which
  holds only for its own backbone: SigLIP so400m's ``head_dim`` is 72 and its
  spatial code is 3 x 24 = 72. CLIP ViT-L/14-336 has ``head_dim`` 64, and the
  assert is not an algorithmic requirement -- each metric is normalized and
  scored on its own, and both score matrices come out [N/2, N/2] regardless --
  so the port keeps upstream's per-axis 24-channel code and compares token
  counts only.

``alpha`` follows upstream's ``alpha_style``: ``uniform`` (0.5, the value its
OpenEQA 3D-QA script uses), ``decrease`` (1 -> 0 across rounds) or ``increase``
(0 -> 1). ``alpha = 1`` is vanilla ToMe; ``alpha = 0`` is purely spatial.

``tests/test_tosa.py`` checks the merge against the upstream functions
transcribed line-for-line, and the sinusoidal table against the
``positional_encodings`` package ToSA imports.
"""
import math

import torch


# -- ToSA's spatial metric -------------------------------------------------
def sinusoidal_1d_table(length, channels, device=None, dtype=torch.float32):
    """``positional_encodings.torch_encodings.PositionalEncoding1D``, inlined.

    Reproduced rather than imported so that running the model never depends on
    a package only one baseline needs; ``tests/test_tosa.py`` asserts the two
    agree exactly whenever the package is installed.
    """
    padded = int(math.ceil(channels / 2) * 2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, padded, 2, dtype=torch.float32, device=device) / padded))
    pos = torch.arange(length, dtype=torch.float32, device=device)
    sin_inp = torch.einsum("i,j->ij", pos, inv_freq)
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1).flatten(-2, -1)
    return emb[:, :channels].to(dtype)


def tosa_spatial_metric(xyz, bins=27, channels=24):
    """ToSA's ``depth_states`` for [N, 3] world coordinates -> [N, 3*channels].

    Each axis is quantised into ``bins`` uniform cells over the scene's own
    extent -- upstream's ``x, y`` run over the 27x27 patch grid and ``z`` over
    27 depth bins -- and the three sinusoidal codes are concatenated.
    """
    if xyz.dim() != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"ToSA expects xyz [N, 3], got {tuple(xyz.shape)}")
    table = sinusoidal_1d_table(bins, channels, device=xyz.device, dtype=torch.float32)
    coords = xyz.float()
    low = coords.min(dim=0, keepdim=True).values
    high = coords.max(dim=0, keepdim=True).values
    span = (high - low).clamp(min=1e-6)
    idx = ((coords - low) / span * bins).floor().long().clamp(0, bins - 1)
    return torch.cat([table[idx[:, 0]], table[idx[:, 1]], table[idx[:, 2]]], dim=-1)


# -- ToMe, with ToSA's two-term score --------------------------------------
def bipartite_soft_matching_w_depth(metric1, metric2, r, alpha):
    """Transcribed from ToSA's ``bipartite_soft_matching_w_depth``.

    ``metric1`` is the semantic metric and ``metric2`` the spatial one, both
    [B, N, C]; ``r`` tokens are removed. Returns the ``merge`` closure. The
    ``class_token`` / ``distill_token`` branches are dropped: LLaVA-3D's scene
    tokens are all patches.
    """
    assert metric1.shape[:2] == metric2.shape[:2]
    num_tokens = metric1.shape[1]
    r = min(int(r), num_tokens // 2)
    if r <= 0:
        return None, 0

    with torch.no_grad():
        m1 = metric1 / metric1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        a, b = m1[..., ::2, :], m1[..., 1::2, :]
        scores1 = a @ b.transpose(-1, -2)

        m2 = metric2 / metric2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        a, b = m2[..., ::2, :], m2[..., 1::2, :]
        scores2 = a @ b.transpose(-1, -2)

        scores = alpha * scores1 + (1 - alpha) * scores2

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]   # unmerged tokens
        src_idx = edge_idx[..., :r, :]   # merged tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x, mode="mean"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    return merge, r


def merge_wavg(merge, x, size=None):
    """Transcribed from ToSA's ``merge_wavg``: size-weighted average."""
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    return x / size, size


# -- scene-level driver ----------------------------------------------------
def resolve_alpha(alpha, alpha_style, round_index, num_rounds):
    """Upstream's ``alpha_style``, read over merge rounds instead of layers."""
    if alpha_style == 'uniform':
        return float(alpha)
    if num_rounds <= 1:
        return 1.0 if alpha_style == 'decrease' else 0.0
    fraction = round_index / (num_rounds - 1)
    if alpha_style == 'decrease':
        return 1.0 - fraction
    if alpha_style == 'increase':
        return fraction
    raise ValueError(f"Unknown TOSA_ALPHA_STYLE '{alpha_style}' "
                     f"(expected uniform, decrease or increase)")


def plan_rounds(num_tokens, target_tokens, schedule='uniform', rounds=26):
    """Tokens removed per round, as a list.

    ``uniform`` mirrors upstream: ToMe removes a *constant* ``r`` per step and
    upstream takes one step in each of SigLIP's layers, so the port spreads the
    same constant over ``rounds`` steps (26 by default, that layer count). Any
    remainder goes into the last steps so the budget is hit exactly, and each
    step is still clipped to ToMe's 50% cap.

    ``halving`` always removes the cap, reaching the budget in ~log2 steps.
    Much cheaper, but far more aggressive than anything upstream does.
    """
    remaining = int(num_tokens)
    target_tokens = int(target_tokens)
    if remaining <= target_tokens:
        return []

    if schedule == 'halving':
        plan = []
        while remaining > target_tokens:
            r = min(remaining - target_tokens, remaining // 2)
            if r <= 0:
                break
            plan.append(r)
            remaining -= r
        return plan

    if schedule != 'uniform':
        raise ValueError(f"Unknown TOSA_SCHEDULE '{schedule}' (expected uniform or halving)")

    rounds = max(1, int(rounds))
    plan = []
    for step in range(rounds):
        left = remaining - target_tokens
        if left <= 0:
            break
        # Spread what is left over the steps that remain, so rounding never
        # leaves the budget unmet.
        share = -(-left // (rounds - step))          # ceil division
        r = min(share, remaining // 2)
        if r <= 0:
            break
        plan.append(r)
        remaining -= r
    # The 50% cap can stop `uniform` short of the budget; finish with halving.
    while remaining > target_tokens:
        r = min(remaining - target_tokens, remaining // 2)
        if r <= 0:
            break
        plan.append(r)
        remaining -= r
    return plan


def tosa_compress(geo_features, semantic_metric, token_xyz, target_tokens,
                  alpha=0.5, alpha_style='uniform', pe_bins=27, pe_channels=24,
                  schedule='uniform', rounds=26, debug=False):
    """Compress [B, N, C] geometry-aware tokens to ``target_tokens`` per scene.

    Args:
        geo_features: [B, N, C] LLaVA-3D geometry-aware tokens (the values).
        semantic_metric: [B, N, C_m] head-averaged CLIP layer -2 keys.
            ``None`` falls back to ``geo_features``.
        token_xyz: [B, N, 3] world-frame XYZ, the spatial metric's input.
        target_tokens: tokens to keep per scene.
        alpha: weight of the semantic term (1 = vanilla ToMe, 0 = spatial only).
        alpha_style: ``uniform`` | ``decrease`` | ``increase``.
        schedule: ``uniform`` (constant r over ``rounds`` steps, as upstream
            does over its layers) or ``halving`` (50% per step).
        rounds: number of merge steps for the ``uniform`` schedule.

    Returns:
        ``(pooled_features, xyz_list, batch_offset)``, the layout the video
        tower returns.
    """
    if geo_features.dim() != 3:
        raise ValueError(f"ToSA expects geo_features [B, N, C], got {tuple(geo_features.shape)}")
    if token_xyz.dim() != 3 or token_xyz.shape[-1] != 3:
        raise ValueError(f"ToSA expects token_xyz [B, N, 3], got {tuple(token_xyz.shape)}")
    if geo_features.shape[:2] != token_xyz.shape[:2]:
        raise ValueError(f"ToSA feature/xyz shape mismatch: "
                         f"{tuple(geo_features.shape)} vs {tuple(token_xyz.shape)}")
    if semantic_metric is None:
        semantic_metric = geo_features
    if semantic_metric.shape[:2] != geo_features.shape[:2]:
        raise ValueError(f"ToSA feature/metric shape mismatch: "
                         f"{tuple(geo_features.shape)} vs {tuple(semantic_metric.shape)}")

    batch_size, num_tokens, _ = geo_features.shape
    target_tokens = int(target_tokens)
    if target_tokens <= 0 or target_tokens > num_tokens:
        target_tokens = num_tokens

    pooled_list, xyz_list, counts = [], [], []
    for b in range(batch_size):
        feats, xyz = _tosa_one_scene(
            geo_features[b], semantic_metric[b], token_xyz[b], target_tokens,
            alpha, alpha_style, pe_bins, pe_channels, schedule, rounds,
            debug=debug, batch_index=b)
        pooled_list.append(feats)
        xyz_list.append(xyz)
        counts.append(feats.shape[0])

    pooled = torch.cat(pooled_list, dim=0)
    batch_offset = torch.tensor(counts, dtype=torch.int32, device=geo_features.device).cumsum(0)
    return pooled, xyz_list, batch_offset


def _tosa_one_scene(geo_features, semantic_metric, token_xyz, target_tokens,
                    alpha, alpha_style, pe_bins, pe_channels, schedule, rounds,
                    debug=False, batch_index=0):
    out_dtype = geo_features.dtype
    # ToMe divides by accumulated sizes that reach the hundreds; bfloat16 has
    # ~3 significant digits, so the running average is kept in float32.
    x = geo_features.float().unsqueeze(0)
    m1 = semantic_metric.float().unsqueeze(0)
    m2 = tosa_spatial_metric(token_xyz, bins=pe_bins, channels=pe_channels).unsqueeze(0)
    xyz = token_xyz.float().unsqueeze(0)
    size = torch.ones_like(x[..., 0, None])

    plan = plan_rounds(x.shape[1], target_tokens, schedule, rounds)
    for round_index, r in enumerate(plan):
        alpha_r = resolve_alpha(alpha, alpha_style, round_index, len(plan))
        merge, applied = bipartite_soft_matching_w_depth(m1, m2, r, alpha_r)
        if merge is None:
            break
        # Upstream merges depth_states with the size from *before* this round
        # and lets the hidden-state merge be the one that advances it.
        m2, _ = merge_wavg(merge, m2, size)
        m1, _ = merge_wavg(merge, m1, size)
        xyz, _ = merge_wavg(merge, xyz, size)
        x, size = merge_wavg(merge, x, size)
        if debug:
            print(f"[ToSA] batch={batch_index} round={round_index} alpha={alpha_r:.3f} "
                  f"removed={applied} tokens={x.shape[1]}", flush=True)

    if debug:
        print(f"[ToSA] batch={batch_index} N={geo_features.shape[0]} "
              f"target={target_tokens} schedule={schedule} rounds={len(plan)} kept={x.shape[1]}", flush=True)

    return x.squeeze(0).to(out_dtype), xyz.squeeze(0).to(token_xyz.dtype)
