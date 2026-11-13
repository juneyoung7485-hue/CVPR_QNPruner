"""DivPrune adapted to LLaVA-3D's multi-view geometry-aware tokens.

Upstream: ``no_query_based/2d/divprune``, whose
``LLaVA/llava/model/llava_arch.py::LlavaMetaForCausalLM.DivPrune`` is the whole
method. It is the Max-Min Diversity Problem (MMDP) solved greedily over the
visual tokens:

  1. Build the pairwise **cosine distance** matrix ``1 - cos_sim`` over the
     visual tokens.
  2. Pick the most isolated token first -- the one whose nearest neighbour is
     farthest away (upstream's ``topk(m2, 2, dim=0, largest=False).values[1]``
     takes the second-smallest entry of each column, skipping the token's own
     zero on the diagonal).
  3. Then repeatedly add the token whose distance to the *already selected set*
     is largest (``min`` over the selected rows, then ``argmax``), until the
     budget is spent.

Nothing else: no attention, no text, no training, no merging. Every kept token
keeps its own feature, and a pruned token contributes to nothing -- the same
selection-only contract DTC, VisPruner and SeGPruner follow in this repo.

3DZip Supplement B.6 describes it as the baseline that uses feature-space
diversity and no spatial structure, and D.1 applies every 2D pruning baseline
after LLaVA-3D's voxelization pooling. Both hold here: the branch runs on the
voxel sequence (``COMPRESS_UNITS=voxel``), and the rule reads no coordinate, no
question and no GT, and merges nothing -- see the conformance tests at the end
of ``tests/test_divprune.py``. ``docs/divprune_llava3d_port.md`` has the table.

What the port changes, and why
------------------------------
* **Where it runs.** Upstream prunes in ``prepare_inputs_labels_for_multimodal``,
  after the tokens are spliced into the sequence, and physically removes rows
  from ``new_input_embeds`` (survivors keep their original position ids).
  LLaVA-3D's visual tokens are produced by the video tower, so the selection
  happens there instead, and the tower simply returns the K tokens it kept.
  The sequence is then built from those -- equivalent, and it keeps the budget
  bookkeeping with every other method in this repo.

* **Selection space.** Upstream scores the tokens *after* ``mm_projector``,
  i.e. in the LLM's embedding space -- which, in single-image LLaVA, carries no
  positional embedding at all. LLaVA-3D's equivalent does:
  ``mm_projector(CLIP_patch + PE(xyz))`` has ``SpatialAwareModule``'s 3D
  embedding inside it, so scoring there would let geometry into a decision
  upstream makes on appearance alone.

  ``DIVPRUNE_SPACE=projected`` (the default) keeps upstream's measurement
  point: the post-projector embeddings, in the LLM's dimension. The 3D
  positional embedding riding inside them is part of the backbone every method
  in this comparison shares, so using it is not what would turn this into a
  3D-aware variant -- adding a *separate* xyz distance term to the score would
  be, and DivPrune here does not. ``clip`` (the PE-free 2D patch features, the
  space VisPruner and VisionZip decide in) and ``geo`` (pre-projector) remain
  as the ablation.

* **Precision.** Upstream runs on fp16 embeddings, where ``1 - cos_sim``
  quantises to ~1e-3 and the greedy ``argmax`` becomes arbitrary among the many
  near-ties a 824-token scene produces. The distances here are computed in
  float32, as everywhere else in this repo.

* **Budget.** Upstream takes a ratio (``SUBSET_RATIO``, e.g. 0.1 of 576 = 58
  tokens); this takes the absolute count K that all the baselines here are
  compared at. ``K = round(ratio * N)`` converts between them.

* **Scene-level budget.** Upstream prunes one image's tokens. LLaVA-3D
  back-projects all views into one world frame first, so the budget is spent
  over the whole scene -- the convention every method here follows, and the
  only one under which a K-token comparison means anything. Diversity is
  therefore measured across views too: a token is dropped for being
  feature-redundant with a token in *another* view, which is exactly the
  redundancy multi-view 3D input creates.

``tests/test_divprune.py`` checks this module against upstream's ``DivPrune``
transcribed line-for-line.
"""
import torch
import torch.nn.functional as F


DIVPRUNE_PRECISIONS = ('fp32', 'fp16')


def divprune_cosine_distance(features, precision='fp32'):
    """Upstream's ``1.0 - pairwise_cosine_similarity(x)``.

    Upstream normalises with ``matrix.norm(dim=1, keepdim=True)``, which for a
    2D [N, C] token matrix is the per-token norm -- the same thing
    ``F.normalize(..., dim=-1)`` computes.

    ``precision`` picks the arithmetic:

    ``fp32`` (the default here) computes the whole matrix in float32.

    ``fp16`` reproduces upstream's numerics, which run on the host's half
    precision embeddings. It matters more than it sounds: ``1 - cos_sim`` then
    quantises hard -- measured on 2257 random unit vectors of width 1024, the
    ~5.1M pairwise distances take only 444 distinct fp16 values -- so the
    greedy ``argmax`` has to break ties that float32 would have resolved. Half
    matmul needs CUDA; on CPU this raises rather than silently computing
    something else.
    """
    if precision not in DIVPRUNE_PRECISIONS:
        raise ValueError(
            f"Unknown DivPrune precision '{precision}' (expected one of "
            f"{', '.join(DIVPRUNE_PRECISIONS)})")
    if precision == 'fp16':
        if features.device.type != 'cuda':
            raise RuntimeError(
                "DIVPRUNE_PRECISION=fp16 needs CUDA: torch has no half matmul on CPU, "
                "and emulating it in float32 would not reproduce upstream's numerics.")
        normalized = F.normalize(features.float(), dim=-1).half()
        return (1.0 - normalized @ normalized.T).half()
    normalized = F.normalize(features.float(), dim=-1)
    return 1.0 - normalized @ normalized.T


DIVPRUNE_SPACES = ('projected', 'geo', 'clip')

# 3DZip does not say whether its DivPrune baseline measured before or after
# ``mm_projector``, and the two upstreams point different ways: DivPrune's own
# code scores the post-projector embeddings, while 3DZip's DPP selection runs on
# the pre-projector voxel features inside the video tower. Both are therefore
# reachable, under the names the reproduction spec uses.
DIVPRUNE_SPACE_ALIASES = {
    'post_projector': 'projected',   # DivPrune upstream's own measurement point
    'pre_projector': 'geo',          # where 3DZip's own selection runs
}


def resolve_divprune_space(space):
    """Accept either the internal names or the pre/post_projector aliases."""
    space = (space or '').strip().lower()
    return DIVPRUNE_SPACE_ALIASES.get(space, space)


def divprune_score_features(geo_features, clip_features=None, projector=None,
                            space='projected', warn=False):
    """The tokens DivPrune measures its distances on.

    ``projected`` is the default, here and in the pipeline, and runs the host's
    ``mm_projector`` so the distances are upstream's: computed on the embeddings
    the LLM will read. It is a scoring space only -- the caller keeps handing on
    the un-projected tokens, which ``llava_arch`` projects itself a moment
    later.

    ``geo`` skips the projector and scores the geometry-aware tokens, the space
    DTC and SeGPruner decide in. ``clip`` goes back further, to the 2D CLIP
    patch features *before* ``SpatialAwareModule`` adds the 3D positional
    embedding -- the only one of the three that is geometry-free, and the space
    VisPruner and VisionZip decide in.
    """
    space = resolve_divprune_space(space)
    if space not in DIVPRUNE_SPACES:
        raise ValueError(
            f"Unknown DivPrune space '{space}' (expected one of {', '.join(DIVPRUNE_SPACES)})")
    if space == 'clip':
        if clip_features is None:
            raise ValueError("DivPrune space 'clip' needs the pre-geometry CLIP features")
        return clip_features
    if space == 'geo':
        return geo_features
    if projector is None:
        if warn:
            print('[DivPrune][WARN] no projector was passed; scoring on geometry-aware '
                  'tokens instead of the LLM embedding space upstream uses.', flush=True)
        return geo_features
    with torch.no_grad():
        return projector(geo_features)


def divprune_select_indices(
    features,
    target_tokens,
    debug=False,
    provenance_out=None,
    scene_tag='',
    precision='fp32',
    order='original',
):
    """Greedy max-min diversity selection over [B, N, C] visual tokens.

    This is upstream's loop with the redundant work removed: instead of
    re-reducing ``cosine_matrix[s[:i]]`` from scratch every iteration, the
    running minimum over the selected rows is carried forward. ``min`` is exact,
    so the two agree bit-for-bit -- ``tests/test_divprune.py`` asserts it.

    Args:
        features: [B, N, C] tokens to select over. Upstream scores the
            post-projector embeddings; see ``DIVPRUNE_SPACE``.
        target_tokens: number of visual tokens to keep per scene.
        scene_tag: optional ``scene_id=... question_id=...`` prefix for the
            debug line, so a log can be traced back to the sample.
        precision: ``fp32`` (default) or ``fp16``, upstream's own arithmetic.
        order: ``original`` (default) restores the input sequence order, which is
            what upstream does (``keep_indexs.sort().values``). ``selection``
            keeps the MMDP pick order instead and exists only so the audit can
            measure what token ordering is worth; it is not upstream behaviour.
        provenance_out: debug-only dict. When given, filled with
            ``greedy_rank``: per scene, an [N] int32 tensor giving the
            iteration at which each token was picked, or -1 if it was pruned.

    Returns:
        selected_indices: list of [K_b] original token indices per scene,
        ascending -- DivPrune only selects, so token order is preserved.
    """
    if features.dim() != 3:
        raise ValueError(f"DivPrune expects features [B, N, C], got {tuple(features.shape)}")

    batch_size, num_tokens, _ = features.shape
    target_tokens = int(target_tokens)
    if target_tokens <= 0 or target_tokens > num_tokens:
        target_tokens = num_tokens

    selected_per_batch = []
    with torch.no_grad():
        for b in range(batch_size):
            if num_tokens < 2 or target_tokens >= num_tokens:
                selected_per_batch.append(
                    torch.arange(num_tokens, dtype=torch.long, device=features.device))
                continue

            distance = divprune_cosine_distance(features[b], precision=precision)

            # Step 1: the most isolated token. Upstream reads the second
            # smallest entry of each column, skipping the token's own zero.
            nearest = torch.topk(distance, 2, dim=0, largest=False).values[1, :]
            # ``distance`` may be fp16 here on purpose: that is what creates the
            # ties upstream's greedy pass has to resolve.
            first = int(torch.argmax(nearest))
            order_list = [first]
            # Step 2..K: farthest from everything already selected. A selected
            # token scores ~0 and so is never the argmax -- except in a scene of
            # near-duplicates, where fp error could let one be picked twice and
            # quietly return fewer than K tokens. -inf makes that impossible.
            distance = distance.float()          # the ranking itself stays exact
            running_min = distance[first].clone()
            running_min[first] = float('-inf')
            for _ in range(1, target_tokens):
                nxt = int(torch.argmax(running_min))
                order_list.append(nxt)
                running_min = torch.minimum(running_min, distance[nxt])
                running_min[nxt] = float('-inf')

            picked = torch.tensor(order_list, dtype=torch.long, device=features.device)
            if order == 'original':
                selected = torch.sort(picked).values      # upstream's own ordering
            elif order == 'selection':
                selected = picked
            else:
                raise ValueError(
                    f"Unknown DivPrune order '{order}' (expected original or selection)")
            selected_per_batch.append(selected)

            if provenance_out is not None:
                rank = torch.full((num_tokens,), -1, dtype=torch.int32, device=features.device)
                rank[picked] = torch.arange(picked.numel(), dtype=torch.int32,
                                            device=features.device)
                provenance_out.setdefault('greedy_rank', []).append(rank)
                kept_norm = F.normalize(features[b, selected].float(), dim=-1)
                pair = 1.0 - kept_norm @ kept_norm.T
                eye = torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)
                provenance_out.setdefault('trace', []).append({
                    'first_index': int(picked[0]),
                    'first_nearest_distance': float(nearest[int(picked[0])]),
                    'order_before_sort': picked.detach().cpu().tolist(),
                    'order_after_sort': torch.sort(picked).values.detach().cpu().tolist(),
                    'min_pairwise_distance': float(pair[~eye].min()),
                    'mean_pairwise_distance': float(pair[~eye].mean()),
                })

            if debug:
                # What MMDP is meant to maximise, measured on what it kept.
                kept_norm = F.normalize(features[b, selected].float(), dim=-1)
                pair = 1.0 - kept_norm @ kept_norm.T
                eye = torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)
                off = pair[~eye]
                print(f"[DivPrune] {scene_tag}batch={b} N={num_tokens} "
                      f"target={target_tokens} order={order} "
                      f"kept={selected.numel()} unique={torch.unique(selected).numel()} "
                      f"min_pair_dist={off.min():.4f} mean_pair_dist={off.mean():.4f} "
                      f"position=pre_llm query_aware=false",
                      flush=True)

    return selected_per_batch


def apply_divprune_to_tokens(features, xyz, selected_indices, target_tokens=None):
    """Gather [B, N, C] features and [B, N, 3] xyz by per-scene indices.

    Pure gather: an output row **is** an input row. DivPrune is a
    selection/pruning method, so nothing here averages, merges or otherwise
    folds a dropped token into a survivor -- the unselected tokens are simply
    discarded. ``test_divprune_output_rows_are_untouched_input_rows`` holds
    that line.

    ``target_tokens`` turns the budget into an assertion: every scene comes out
    at exactly ``K``, or at its own token count when the scene has fewer than
    ``K`` to begin with (a no-op, never padded up by duplicating tokens).
    """
    kept_features = []
    kept_xyz = []
    counts = []

    for b, indices in enumerate(selected_indices):
        kept_features.append(features[b, indices])
        kept_xyz.append(xyz[b, indices])
        counts.append(int(indices.numel()))
        if target_tokens is not None:
            expected = min(int(target_tokens), features.shape[1])
            assert counts[-1] == expected, (
                f'DivPrune budget broken for scene {b}: kept {counts[-1]} tokens, '
                f'expected {expected} (input had {features.shape[1]})')
            assert int(torch.unique(indices).numel()) == counts[-1], (
                f'DivPrune selected a duplicate token for scene {b}')
            assert int(indices.min()) >= 0 and int(indices.max()) < features.shape[1], (
                f'DivPrune index out of range for scene {b}: '
                f'[{int(indices.min())}, {int(indices.max())}] vs N={features.shape[1]}')

    output_features = torch.cat(kept_features, dim=0)
    batch_offset = torch.tensor(counts, dtype=torch.int32, device=features.device).cumsum(0)
    return output_features, kept_xyz, batch_offset
