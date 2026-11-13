"""SeGPruner adapted to LLaVA-3D's multi-view geometry-aware tokens.

Upstream: ``no_query_based/3d/SegPruner`` (LLaVA-OneVision + SigLIP), whose
``farthest_point_sample`` in ``llava/model/llava_arch.py`` is the whole method:

  1. **Saliency-aware Token Selector** -- keep the top ``T * r`` tokens by
     visual attention.
  2. **Geometry-aware Token Diversifier** -- farthest point sampling over the
     *remaining* tokens under a fused distance that trades 3D spread against
     semantic novelty::

         fused = lam * (d_3d / d_max) + (1 - lam) * (1 - cos_sim)

  3. Concatenate the two sets and hand them to the LLM.

Two things change in the port, both forced by the host model, and neither
touches the algorithm:

* **Attention source.** SigLIP has no CLS token, so upstream averages the
  attention each token *receives* (``attentions[-1].mean(dim=-2)``). CLIP does
  have one, and LLaVA-3D already extracts CLS->patch attention for VisPruner
  (``CLIPVisionTower._cls_attn``), so that is the importance score here.
* **Scene-level budget.** Upstream runs the selection per view because its
  pipeline never merges views into one frame. LLaVA-3D back-projects every view
  into one world frame before compression, so the budget is spent over the whole
  scene -- the same convention DTC, VisPruner and 3DZip use in this repo, and
  the only one that makes the K-token comparison meaningful. It also gives the
  geometry term what SeGPruner actually wants: true scene-wide 3D distances
  rather than per-view ones.

Upstream writes its picks into a boolean ``index_masks`` and gathers with it, so
its output is in ascending token order too -- the ``[important..., diverse...]``
concatenation does not survive. This port sorts for the same reason.

``tests/test_upstream_equivalence.py`` checks this module against the upstream
FPS loop transcribed line-for-line; the selected indices match exactly.
"""
import torch
import torch.nn.functional as F


def segpruner_select_indices(
    visual_features,
    visual_attn,
    token_xyz,
    target_tokens,
    important_ratio=0.5,
    lam=0.5,
    debug=False,
    provenance_out=None,
):
    """SeGPruner token selection.

    Args:
        visual_features: [B, N, C] pre-geometry 2D visual patch features; the
            semantic half of the fused distance.
        visual_attn: [B, N] visual attention scores (CLIP CLS->patch). ``None``
            falls back to feature norm, as in ``vispruner.py``.
        token_xyz: [B, N, 3] world-frame XYZ of each patch token; the geometric
            half of the fused distance.
        target_tokens: total tokens to keep per scene (``T``).
        important_ratio: fraction taken by attention alone (``r``).
        lam: weight of the 3D distance term against semantic novelty.
        provenance_out: debug-only dict. When given, filled with
            ``selection_kind``: per scene an [N] int8 tensor, 0 = kept as
            attention-salient, 1 = kept as geometry-diverse, -1 = pruned.
            SeGPruner only *selects*, so a kept token's feature is its own.

    Returns:
        selected_indices: list of [K_b] original token indices per scene,
        ascending (see the ordering note in ``vispruner.py``).
    """
    if visual_features.dim() != 3:
        raise ValueError(f"SeGPruner expects visual_features [B, N, C], got {tuple(visual_features.shape)}")
    if token_xyz.dim() != 3 or token_xyz.shape[-1] != 3:
        raise ValueError(f"SeGPruner expects token_xyz [B, N, 3], got {tuple(token_xyz.shape)}")
    if visual_attn is None:
        visual_attn = visual_features.float().norm(dim=-1)
    if visual_attn.dim() != 2:
        raise ValueError(f"SeGPruner expects visual_attn [B, N], got {tuple(visual_attn.shape)}")
    if visual_features.shape[:2] != visual_attn.shape:
        raise ValueError(
            f"SeGPruner feature/attention shape mismatch: "
            f"{tuple(visual_features.shape)} vs {tuple(visual_attn.shape)}"
        )
    if visual_features.shape[:2] != token_xyz.shape[:2]:
        raise ValueError(
            f"SeGPruner feature/xyz shape mismatch: "
            f"{tuple(visual_features.shape)} vs {tuple(token_xyz.shape)}"
        )

    batch_size, num_tokens, _ = visual_features.shape
    target_tokens = int(target_tokens)
    if target_tokens <= 0 or target_tokens > num_tokens:
        target_tokens = num_tokens

    important_ratio = min(max(float(important_ratio), 0.0), 1.0)
    lam = min(max(float(lam), 0.0), 1.0)
    important_token_num = min(target_tokens, int(target_tokens * important_ratio))
    diverse_token_num = target_tokens - important_token_num

    feature_norm = F.normalize(visual_features.float(), dim=-1)
    selected_per_batch = []

    for b in range(batch_size):
        token_order = torch.argsort(visual_attn[b].float(), descending=True)
        important_indices = token_order[:important_token_num]
        residual_indices = token_order[important_token_num:]

        n_diverse = min(diverse_token_num, residual_indices.numel())
        if n_diverse > 0:
            local = _geometry_aware_fps(
                token_xyz[b, residual_indices].float(),
                feature_norm[b, residual_indices],
                n_diverse,
                lam,
            )
            diverse_indices = residual_indices[local]
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
                f"[SeGPruner] batch={b} N={num_tokens} target={target_tokens} lam={lam} "
                f"important={important_indices.numel()} diverse={diverse_indices.numel()} "
                f"kept={selected.numel()}",
                flush=True,
            )

    return selected_per_batch


def _geometry_aware_fps(xyz, feature_norm, num_samples, lam):
    """Farthest point sampling under SeGPruner's fused geometric/semantic distance.

    ``xyz`` is [R, 3], ``feature_norm`` is [R, C] already L2-normalized, and the
    returned tensor holds ``num_samples`` indices into those R rows.

    Follows the upstream loop exactly: seed at row 0 (the highest-attention
    residual token, since the caller passes the rows in attention order), and
    normalize every later 3D distance by ``d_max``, the largest distance seen
    from that seed. An already-picked row scores 0 on both terms, so it can
    never be picked twice and no explicit mask is needed.
    """
    device = xyz.device
    num_residual = xyz.shape[0]
    centroids = torch.zeros(num_samples, dtype=torch.long, device=device)
    distance = torch.full((num_residual,), 1e10, device=device, dtype=torch.float32)
    farthest = torch.zeros((), dtype=torch.long, device=device)
    d_max = None

    for i in range(num_samples):
        centroids[i] = farthest
        dist = torch.norm(xyz - xyz[farthest].unsqueeze(0), dim=-1)
        scores = feature_norm @ feature_norm[farthest]
        if d_max is None:
            d_max = dist.max().clamp(min=1e-8)
        fused = lam * (dist / d_max) + (1.0 - lam) * (1.0 - scores)
        distance = torch.minimum(distance, fused)
        farthest = distance.argmax()

    return centroids


def apply_segpruner_to_tokens(features, xyz, selected_indices):
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
