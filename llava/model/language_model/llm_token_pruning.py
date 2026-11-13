"""FastV and SparseVLM adapted to LLaVA-3D, as query-based decoder-side pruning.

Both upstreams differ from every compressor already in this repo: they do not
touch the vision path at all. The full set of visual tokens enters the LLM, and
tokens are dropped *inside the decoder*, scored by how much the **text** attends
to them. That is what makes them query-based -- the same scene compresses
differently for different questions.

FastV (``query_based/2d/FastV``)
-------------------------------
``modeling_llama_fastv_kvcache.py``: run decoder layer ``AGG_LAYER`` (2 by
default) normally, average its attention over heads, take the **last prompt
token's** attention row restricted to the visual span, keep the top ``R``
visual tokens, and drop the rest for every later layer.

SparseVLM (``query_based/2d/SparseVLMs``)
-----------------------------------------
``modelling_sparse_llama.py`` + ``score.py``:

1. *Text raters* -- before the first layer, score text tokens by
   ``(v @ t.T).softmax(-1).mean(over visual)`` and keep those above the mean.
   Only these "rater" tokens get a vote.
2. At each of layers ``[2, 6, 15]``, average attention over heads, take the
   rater rows restricted to the still-alive visual tokens, average over raters,
   and keep the top ``sparse_token_list[stage]``.
3. *Token recycling* -- the top 30% of the just-pruned tokens by that same
   score are clustered (density-peak clustering, ``cluster_num = n/10 + 1``)
   and the cluster averages are re-inserted.

What the port changes, and why
------------------------------
* **Dropping is done by masking.** Upstream physically removes rows from
  ``hidden_states`` and surgically rewrites the KV cache, position ids and
  attention mask -- fragile machinery that has to be re-derived for every
  transformers version. This port instead masks the pruned columns out of the
  attention from the pruning layer onward. No later query can see a pruned
  token, so the model's output is the same; what is lost is the *speedup*,
  since the tokens still occupy their slots. FastV ships exactly this mode
  itself (``USE_FAST_V_Token_Drop = False``). **These baselines therefore
  measure accuracy at a token budget, not latency.**
* **Token recycling re-uses the cluster centre's slot.** Masking cannot append
  tokens, so each cluster's density-peak centre stays alive and its features
  are replaced by the cluster average -- "merge into the centre" rather than
  "append a new token". Set ``SPARSEVLM_RECYCLE=0`` for upstream's own v2.0
  behaviour, which cancels the merge outright (``s_flag = False``).
* **The reconstructed tokens are inside the budget, not on top of it.**
  Upstream never has to decide this: its ``v_token_num`` after a stage is
  ``pred_score_vis.sum() + cluster_num``, and the ``RETAIN_TOKN`` label is only
  a name for a tuned table. A fixed-budget comparison does have to decide, and
  "K" here means **K visual tokens reach the LLM**::

      effective = kept originals + reconstructed
      kept_original=119 + reconstructed=9 = effective 128

  Keeping 128 originals and then re-inserting the recycled clusters would put
  137 visual tokens in front of the decoder while the row is labelled 128,
  which is not the budget the vision-path baselines run at. So the schedule is
  solved against ``effective``: ``sparsevlm_recycle_count`` predicts what
  ``_recycle`` will add, and ``sparsevlm_keep_for_effective`` inverts
  ``keep + C(alive - keep)`` -- it rises in unit steps, so the bisection lands
  on the target exactly. In ``retained`` mode the final stage is re-solved
  inside the forward pass against the count that really survived, which is
  SparseVLM 3.4's exact-budget stage; the ranking stays the rater attention,
  never a new criterion. **This is an adaptation for the fixed-budget
  comparison, not something upstream does.**
* **Budget.** Decoder-side pruning has no single token count: the number of
  live visual tokens changes with depth. The budget here is the **average over
  decoder layers**, which is what SparseVLM's own ``RETAIN_TOKN`` labels denote
  (its 192 schedule is ``[300, 200, 110]``, averaging ~192 over 32 layers).
  ``LLM_PRUNE_BUDGET=retained`` instead reads K as the final effective count,
  and is the default here. Both the average and the final count are reported.
  The layer average counts reconstructed tokens too, since they are live
  visual tokens for every layer after the stage that made them.

  **At K=128 on this data ``average`` is not usable, and the reason is easy to
  miss.** The layers before the first pruning layer always see every token, so
  they put a floor under the average: ``(prune_from * N + tail) / num_layers``.
  SQA3D's entering counts measured over the split are min 446, median 1248,
  mean 1334, max 2560 -- so for FastV (``prune_from=3``) a target of 128 is
  *below the floor* for **41.9%** of questions, and the median retained count
  over the whole split is **12 tokens**. SparseVLM (first stage at layer 2) is
  unreachable for 12.1%. Both used to clamp to a single token in silence;
  ``_warn_infeasible`` now says so once per scene size.

      convention (K=128)   fastv retained        sparsevlm    avg/layer
      average              median 12, 42% at 1   unreachable 12%   --
      retained             128                   128          median 233

  A vision-path method at K=128 keeps 128 tokens in *every* layer, so it sits
  at 128 on both counts. ``average`` is the FLOPs-matched idea and the
  convention the SparseVLM paper's tables use, but it only means anything while
  the target stays above the floor -- on 576-token single images it does, on
  scene-sized inputs at a small K it does not. ``retained`` matches the token
  count instead, at a higher layer average. Report which one you used; a K=128
  table needs ``retained``, which is why it is the default.

  One wrinkle in ``retained``: anchoring the last stage rescales the earlier
  ones, and upstream's 128 table has a large first stage, so it clamps --
  ``(824, 391, 128)`` means layer 2 prunes nothing at all. The 192 profile
  degrades more gracefully (``(349, 232, 128)``) if that matters.
* **Schedule.** SparseVLM's per-budget tables are tuned for exactly 576 visual
  tokens. LLaVA-3D emits a scene-dependent count (462-1078 on SQA3D), so the
  port takes the table upstream published *for the requested budget* -- 128 is
  ``[303, 110, 36]``, not a scaled ``[300, 200, 110]`` -- reads it as fractions
  of 576, and rescales those per scene. A budget upstream never published falls
  back to the 192 profile, and the install line says which table was used.
  ``SPARSEVLM_STAGES`` overrides it with explicit counts. Note that upstream's
  labels are rounded: its tables average a little under them (192 -> 188.2,
  128 -> 123.9, 96 -> 93.1, 64 -> 61.7). Keeping the profile while hitting the
  requested average exactly therefore scales the stages up by 2.5% (at 192) to
  8.9% (at 64) -- the first layers always see every token, so closing a small
  gap in the average takes a larger correction in the stages.

* **Causality at the scoring layers.** Asking one layer for its attention
  makes transformers fall back from sdpa to its eager attention, which needs
  the 4D causal mask that the sdpa path had already optimised away. The pruner
  therefore materialises that mask itself -- see ``_make_layer_pre_hook``.

``tests/test_llm_token_pruning.py`` checks the scoring rules against the
upstream functions transcribed line-for-line.
"""
import math
import os

import torch

# Set by ``llava_arch.prepare_inputs_labels_for_multimodal`` right after the
# visual tokens are spliced into the sequence: where they start and how many
# there are. A plain module global because the pruning runs inside the decoder,
# several frames below the code that knows the layout. The evaluators also drop
# ``scene_id``/``question_id`` in here so a debug line can name the sample it
# belongs to; both are optional and the pruner works without them.
PRUNE_STATE = {}


def _sample_tag():
    """``scene_id=... question_id=...`` for the debug lines, when known."""
    parts = [f'{key}={PRUNE_STATE[key]}' for key in ('scene_id', 'question_id')
             if PRUNE_STATE.get(key) is not None]
    return (' '.join(parts) + ' ') if parts else ''

# (method, num_visual) pairs already warned about, so an infeasible budget
# says so once per scene size instead of once per question.
_INFEASIBLE_WARNED = set()

# Upstream SparseVLM's per-budget stage schedules (``score.py``'s
# ``sparse_token_dict``, the v1.0 lists), keyed by the ``RETAIN_TOKN`` label
# they ship under. They are *not* rescalings of one another -- the 128 schedule
# starts *higher* than the 192 one and then falls off a cliff -- so the budget
# picks the table, and only the visual-token count gets rescaled.
# Only SparseVLM **V1** is ported. Upstream selects the version with
# ``USE_VERSION`` in ``score.py``; everything guarded by ``V2_0`` there -- the
# v2.0 token lists, the RoPE attention-gravity correction (SparseVLM+ 4.2), the
# priority-head selection (4.3) and the cancel-merge -- is deliberately absent
# from this module. ``assert_v1_only`` checks that nothing has crept back in.
SPARSEVLM_VERSION = 'v1'

SPARSEVLM_SCHEDULES = {
    192: (300, 200, 110),
    128: (303, 110, 36),
    96: (238, 48, 26),
    64: (66, 30, 17),
}
# The v2.0 lists, kept only so the assertion can prove they are not in use.
SPARSEVLM_V2_SCHEDULES = {
    192: (300, 200, 118),
    128: (238, 108, 60),
    96: (246, 54, 28),
    64: (66, 34, 20),
}
SPARSEVLM_UPSTREAM_VISUAL = 576
SPARSEVLM_DEFAULT_BUDGET = 192
SPARSEVLM_LAYERS = (2, 6, 15)

# ``official_v1`` runs upstream's own schedule and reports whatever token count
# it lands on. ``fixed_budget`` is the 3DZip comparison adaptation, which solves
# the schedule so the effective count is exactly K.
SPARSEVLM_MODES = ('official_v1', 'fixed_budget')


def assert_v1_only():
    """Fail loudly if a v2.0 table or code path is in play.

    Cheap, and it makes ``[SparseVLM Version] version=v1`` in the startup log a
    claim about the code rather than a comment.
    """
    for budget, counts in SPARSEVLM_SCHEDULES.items():
        assert counts != SPARSEVLM_V2_SCHEDULES[budget], (
            f'SparseVLM v2.0 schedule for {budget} is in SPARSEVLM_SCHEDULES')
    for banned in ('select_attn_head_by_sum', 'rope_weights', 'apply_rotary_pos_emb'):
        assert banned not in globals(), f'SparseVLM+ symbol {banned} is defined'
    return SPARSEVLM_VERSION


def sparsevlm_shape(target, budget_mode='average'):
    """Upstream's stage profile, as fractions of its 576 visual tokens.

    **Which table depends on how K is read, and getting this wrong is worth
    several accuracy points.** Upstream selects ``sparse_token_dict[RETAIN_TOKN]``
    and its own comment prices that label as a *layer average*
    (``# 2*576 4*300 10*200 16*110`` averages to 191 over 32 layers, and the
    128 table likewise averages 126). The tables are tuned for that reading.

    In ``average`` mode the requested K is that same label, so the table for K
    is upstream's own answer and is used directly.

    In ``retained`` mode K is the **final** count, which upstream never labels.
    Anchoring a table's last stage on an absolute K rescales the whole profile
    by ``K / (frac_last * N)``, and the small-K tables have a deliberately tiny
    last fraction -- 36/576 = 6.25% at K=128 -- precisely because it is the
    average being controlled. Anchoring there inflates every earlier stage by
    ~16x the target: for 2257 entering tokens the 128 table gives stages
    ``(1077, 391, 128)``, so half the decoder still runs on 400-2200 tokens and
    the layer average is 464 -- against FastV's 327 at the same nominal budget.
    Measured on SQA3D that is +4.2 EM1 over the published SparseVLM row.

    So ``retained`` scales upstream's **reference** profile (the 192 schedule,
    the one its paper headlines) instead: ``(349, 233, 128)``, a layer average
    of 318, which is compute-comparable to FastV at the same K.

    ``SPARSEVLM_PROFILE`` overrides the choice with an explicit table key.
    ``install_llm_pruning`` prints which one was used.
    """
    forced = os.environ.get('SPARSEVLM_PROFILE', 'auto').strip().lower()
    if forced not in ('', 'auto'):
        budget = int(forced)
        if budget not in SPARSEVLM_SCHEDULES:
            raise ValueError(
                f"Unknown SPARSEVLM_PROFILE '{forced}' (expected auto or one of "
                f"{sorted(SPARSEVLM_SCHEDULES)})")
    elif budget_mode == 'retained':
        budget = SPARSEVLM_DEFAULT_BUDGET
    else:
        budget = int(target) if int(target) in SPARSEVLM_SCHEDULES else SPARSEVLM_DEFAULT_BUDGET
    counts = SPARSEVLM_SCHEDULES[budget]
    return tuple(c / SPARSEVLM_UPSTREAM_VISUAL for c in counts), budget


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_flag(name, default='0'):
    return os.environ.get(name, default) not in ('0', 'false', 'False', '')


def prune_config():
    """Read the LLM-pruning knobs from the environment."""
    method = os.environ.get('LLM_PRUNE', 'none').lower()
    if method in ('', 'none', 'off'):
        return None
    if method not in ('fastv', 'sparsevlm'):
        raise ValueError(f"Unknown LLM_PRUNE '{method}' (expected fastv, sparsevlm or none)")
    stages = os.environ.get('SPARSEVLM_STAGES', '').strip()
    mode = os.environ.get('SPARSEVLM_MODE', 'official_v1').lower()
    if mode not in SPARSEVLM_MODES:
        raise ValueError(f"Unknown SPARSEVLM_MODE '{mode}' (expected one of "
                         f"{', '.join(SPARSEVLM_MODES)})")
    return {
        'method': method,
        'tokens': _env_int('LLM_PRUNE_TOKENS', 128),
        'budget': os.environ.get('LLM_PRUNE_BUDGET', 'average'),
        'debug': _env_flag('LLM_PRUNE_DEBUG'),
        'fastv_layer': _env_int('FASTV_LAYER', 2),
        'sparsevlm_layers': tuple(int(x) for x in
                                  os.environ.get('SPARSEVLM_LAYERS', '2,6,15').split(',')),
        'sparsevlm_recycle': _env_flag('SPARSEVLM_RECYCLE', '1'),
        # Upstream V1's ``cluster_and_merge`` divides by the cluster size. The
        # paper's 3.3 says element-wise sum; ``sum`` opts into that reading.
        'sparsevlm_reconstruct': os.environ.get('SPARSEVLM_RECONSTRUCT', 'mean').lower(),
        'sparsevlm_stages': tuple(int(x) for x in stages.split(',')) if stages else None,
        'sparsevlm_mode': mode,
        # V1 truncates position_ids after a prune, so survivors are renumbered
        # 0..L-1. Keeping the original ids is what V2 added as "idea1".
        'sparsevlm_position_ids': os.environ.get('SPARSEVLM_POSITION_IDS', 'renumber').lower(),
        'sparsevlm_version': assert_v1_only(),
    }


# -- budget solving --------------------------------------------------------
def min_achievable_average(num_visual, num_layers, prune_from):
    """The smallest layer-average this schedule can reach: one token in the tail.

    The first ``prune_from`` layers always see every token, so they alone put a
    floor under the average. Ask for less and no retained count exists.
    """
    if num_layers <= 0:
        return 0.0
    return (prune_from * num_visual + (num_layers - prune_from)) / num_layers


def _warn_infeasible(method, num_visual, target, floor):
    key = (method, int(num_visual))
    if key in _INFEASIBLE_WARNED:
        return
    _INFEASIBLE_WARNED.add(key)
    print(f"[LLMPrune][WARN] {method}: a layer-average of {int(target)} is out of "
          f"reach with {num_visual} visual tokens -- the layers before the first "
          f"pruning layer already average {floor:.0f}. Retaining the minimum "
          f"instead, so this run is NOT at the requested budget. Use "
          f"LLM_PRUNE_BUDGET=retained to set the retained count directly.",
          flush=True)


def fastv_retained(num_visual, target, num_layers, prune_from, budget='average'):
    """Visual tokens FastV keeps, so the layer-average hits ``target``.

    Layers ``[0, prune_from)`` see all ``num_visual`` tokens; the rest see the
    retained count. When the target is below what those first layers already
    force, there is no solution and this warns rather than silently returning
    a single token -- which is what a scene-dependent token count makes easy to
    miss: 824 entering tokens leave room for a 128 average, 2257 do not.
    """
    if budget == 'retained':
        return max(1, min(int(target), num_visual))
    if budget != 'average':
        raise ValueError(f"Unknown LLM_PRUNE_BUDGET '{budget}' (expected average or retained)")
    tail = num_layers - prune_from
    if tail <= 0:
        return num_visual
    retained = (int(target) * num_layers - prune_from * num_visual) / tail
    if retained < 1:
        _warn_infeasible('fastv', num_visual, target,
                         min_achievable_average(num_visual, num_layers, prune_from))
    return max(1, min(int(round(retained)), num_visual))


def sparsevlm_recycle_count(alive, keep, recycle=True):
    """Reconstructed tokens ``_recycle`` adds when ``alive`` is pruned to ``keep``.

    Mirrors ``_recycle`` exactly, which is upstream's SparseVLM 3.3: the top
    30% (+1) of the just-pruned tokens are clustered into ``take / 10 + 1``
    centres. Having it as a plain function is what lets the budget be solved
    *before* the forward pass, so the reconstructed tokens land inside the
    budget rather than on top of it.
    """
    dropped = int(alive) - int(keep)
    if not recycle or dropped <= 0:
        return 0
    take = min(int(dropped * 0.3) + 1, dropped)
    return min(int(take / 10) + 1, take)


def sparsevlm_effective(alive, keep, recycle=True):
    """Visual tokens the LLM actually sees after this stage: kept + reconstructed."""
    return int(keep) + sparsevlm_recycle_count(alive, keep, recycle)


def sparsevlm_keep_for_effective(alive, target, recycle=True):
    """The ``keep`` whose kept+reconstructed count is exactly ``target``.

    ``effective(keep) = keep + C(alive - keep)`` rises by 0 or 1 for each extra
    kept token -- ``C`` can never fall by more than one per token -- so it is
    monotone with unit steps and a bisection lands on ``target`` exactly
    whenever the target is reachable. Selection is untouched: this only sets
    *how many* of the rater-ranked tokens survive.

    Keeping a single token still hands on that step's recycled clusters, so
    ``1 + C(alive - 1)`` is a floor. Below it the return is 1, and the caller
    reports the count it actually got rather than the one it asked for.
    """
    alive, target = int(alive), int(target)
    if alive <= 1 or target >= alive:
        return alive
    lo, hi = 1, alive
    while lo < hi:
        mid = (lo + hi) // 2
        if sparsevlm_effective(alive, mid, recycle) >= target:
            hi = mid
        else:
            lo = mid + 1
    return max(1, min(lo, alive))


def _sparsevlm_walk(num_visual, shape, scale, recycle):
    """Run the stage profile at one scale, tracking what stays live.

    ``alive`` carries the *effective* count forward, so a later stage prunes
    the sequence the previous stage really left behind -- reconstructed tokens
    included.
    """
    keeps, effectives = [], []
    alive = int(num_visual)
    for frac in shape:
        keep = int(round(frac * num_visual * scale))
        keep = max(1, min(keep, alive - 1)) if alive > 1 else 1
        effective = sparsevlm_effective(alive, keep, recycle)
        keeps.append(keep)
        effectives.append(effective)
        alive = effective
    return tuple(keeps), tuple(effectives)


def _sparsevlm_effectives(num_visual, keeps, recycle=True):
    """Effective counts for an explicit ``SPARSEVLM_STAGES`` schedule."""
    effectives, alive = [], int(num_visual)
    for keep in keeps:
        keep = max(1, min(int(keep), alive - 1)) if alive > 1 else 1
        alive = sparsevlm_effective(alive, keep, recycle)
        effectives.append(alive)
    return tuple(effectives)


def sparsevlm_official_v1_schedule(num_visual, target, recycle=True):
    """Upstream V1's own schedule, used verbatim -- adaptation **A**, absolute.

    ``score.py`` picks ``sparse_token_dict[RETAIN_TOKN]`` and feeds each stage's
    count straight into ``topk(relation_vis, min(count, v_token_num - 1))``.
    Those counts are absolute, not fractions, so porting the method "as is" to a
    scene-dependent token count means using them as they are written. A budget
    upstream never published falls back to its 192 table.

    Nothing here is solved against a target: the final effective count is
    whatever the schedule lands on, and the caller reports it. Forcing it to K
    is a *different* mode (``fixed_budget``), because a post-hoc top-K is not
    part of upstream V1.
    """
    budget = int(target) if int(target) in SPARSEVLM_SCHEDULES else SPARSEVLM_DEFAULT_BUDGET
    counts, effectives = [], []
    alive = int(num_visual)
    for count in SPARSEVLM_SCHEDULES[budget]:
        # Upstream's own clamp: ``min(sparse_token_list[stage], v_token_num - 1)``.
        keep = max(1, min(int(count), alive - 1)) if alive > 1 else 1
        alive = sparsevlm_effective(alive, keep, recycle)
        counts.append(keep)
        effectives.append(alive)
    return tuple(counts), tuple(effectives), budget


def sparsevlm_schedule(num_visual, target, num_layers, layers, budget='average',
                       recycle=True):
    """Per-stage ``(keeps, effectives)``, keeping upstream's stage profile.

    ``keeps`` is how many original tokens each stage retains; ``effectives`` is
    what the next layers actually see, ``keep + reconstructed``. **The budget is
    read against ``effectives``**, because that is the visual sequence the LLM
    receives -- retaining ``target`` originals and then re-inserting the
    recycled clusters would put more than ``target`` visual tokens in front of
    the decoder, which is not the budget the comparison is run at.

    ``retained``: the last stage is solved so its effective count *is* the
    target, and the earlier stages keep upstream's profile.
    ``average``: the scale is bisected so the layer-average of ``effectives``
    is the target.
    """
    shape = sparsevlm_shape(target, budget)[0][:len(layers)]

    if budget == 'retained':
        anchor = shape[-1] * num_visual
        scale = int(target) / anchor if anchor > 0 else 1.0
        keeps, effectives = _sparsevlm_walk(num_visual, shape, scale, recycle)
        # Re-solve only the final stage, against the count that actually
        # reaches it, so the budget is exact rather than approximately right.
        alive = effectives[-2] if len(effectives) > 1 else num_visual
        keep = sparsevlm_keep_for_effective(alive, int(target), recycle)
        keeps = keeps[:-1] + (keep,)
        effectives = effectives[:-1] + (sparsevlm_effective(alive, keep, recycle),)
        return keeps, effectives

    if budget != 'average':
        raise ValueError(f"Unknown LLM_PRUNE_BUDGET '{budget}' (expected average or retained)")

    def average_at(scale):
        return layer_average(num_visual, _sparsevlm_walk(num_visual, shape, scale, recycle)[1],
                             layers, num_layers)

    floor = average_at(0.0)
    if floor > int(target):
        _warn_infeasible('sparsevlm', num_visual, target,
                         min_achievable_average(num_visual, num_layers, layers[0]))
        return _sparsevlm_walk(num_visual, shape, 0.0, recycle)

    lo, hi = 0.0, 1.0
    while average_at(hi) < int(target) and hi < 1e6:
        hi *= 2.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if average_at(mid) < int(target):
            lo = mid
        else:
            hi = mid
    return _sparsevlm_walk(num_visual, shape, hi, recycle)


def sparsevlm_stage_counts(num_visual, target, num_layers, layers, budget='average',
                           recycle=True):
    """The kept-original counts of :func:`sparsevlm_schedule`."""
    return sparsevlm_schedule(num_visual, target, num_layers, layers, budget, recycle)[0]


def layer_average(num_visual, counts, layers, num_layers):
    """Average live visual tokens per decoder layer, for reporting."""
    total = layers[0] * num_visual
    for i, count in enumerate(counts):
        end = layers[i + 1] if i + 1 < len(layers) else num_layers
        total += (end - layers[i]) * count
    return total / num_layers


# -- upstream scoring rules ------------------------------------------------
def fastv_scores(attn_weights, v_start, v_num):
    """FastV: head-averaged attention of the **last** query token to the visuals.

    Transcribed from ``modeling_llama_fastv_kvcache.py``::

        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
        ...[SYS_LENGTH:SYS_LENGTH + IMAGE_TOKEN_LENGTH]
    """
    return attn_weights.mean(dim=1)[0][-1][v_start:v_start + v_num]


def sparsevlm_rater_indices(hidden_states, v_start, v_end):
    """SparseVLM 3.2's text raters, from ``modelling_sparse_llama.py``::

        m_v_t = v_t @ t_t.transpose(1, 2)
        m_v_t = m_v_t.softmax(2).mean(1)
        t_token_idx = torch.where(m_v_t > m_v_t.mean())

    Returns absolute sequence positions of the rater text tokens.
    """
    v_t = hidden_states[:, v_start:v_end, :]
    t_t = hidden_states[:, v_end:, :]
    if t_t.shape[1] == 0:
        return None
    m_v_t = v_t.float() @ t_t.float().transpose(1, 2)
    m_v_t = m_v_t.softmax(2).mean(1)
    rater = torch.where(m_v_t > m_v_t.mean())[1]
    if rater.numel() == 0:
        rater = torch.arange(t_t.shape[1], device=hidden_states.device)
    return rater + v_end


def sparsevlm_scores(attn_weights, rater_abs, v_start, v_num):
    """SparseVLM: rater-averaged attention to the visuals, from ``score.py``::

        self_attn_weights = self_attn_weights.mean(1)
        relation_vis_text = self_attn_weights[:, t_token_idx, v_start:v_start + v_num]
        relation_vis_text = relation_vis_text.mean(1)
    """
    averaged = attn_weights.mean(1)
    relation = averaged[:, rater_abs, v_start:v_start + v_num]
    return relation.mean(1)[0]


def cluster_and_merge(x, cluster_num, assignment_out=None, reduce='mean'):
    """Density-peak clustering, transcribed from SparseVLM's ``utils.py``.

    ``x`` is [B, N, C]. Returns ``(merged [B, cluster_num, C], centre indices
    [B, cluster_num])``; upstream returns only the first, but the port needs
    the centres to know which slots the merged tokens take over.

    ``assignment_out``, when given, also receives ``idx_cluster`` -- which
    cluster each input row landed in. Only the Fig.4 export reads it, so that
    the record can say which tokens a recycled token actually stands for.
    """
    B, N, C = x.shape
    distance = torch.cdist(x, x)
    dist_matrix = distance / (C ** 0.5)
    dist_nearest, _ = torch.topk(dist_matrix, k=min(cluster_num, N), dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

    mask = density[:, None, :] > density[:, :, None]
    mask = mask.type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

    score = dist * density
    _, index_down = torch.topk(score, k=cluster_num, dim=-1)

    centre_dist = torch.gather(
        dist_matrix, 1, index_down[..., None].expand(B, cluster_num, N))
    idx_cluster = centre_dist.argmin(dim=1)

    idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
    idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    token_weight = x.new_ones(B, N, 1)
    idx = idx_cluster + torch.arange(B, device=x.device)[:, None] * cluster_num
    if reduce == 'sum':
        # SparseVLM 3.3 as the paper states it: the reconstructed token is the
        # element-wise SUM of its cluster. Upstream's released code divides by
        # the cluster size instead (``norm_weight`` below), so the two differ by
        # a per-cluster scale -- and a summed token enters the residual stream
        # with several times the norm of a real one, which the LLM does notice.
        norm_weight = token_weight
    elif reduce == 'mean':
        all_weight = token_weight.new_zeros(B * cluster_num, 1)
        all_weight.index_add_(0, idx.reshape(B * N), token_weight.reshape(B * N, 1))
        all_weight = all_weight + 1e-6
        norm_weight = token_weight / all_weight[idx]
    else:
        raise ValueError(f"Unknown reconstruction '{reduce}' (expected sum or mean)")

    merged = x.new_zeros(B * cluster_num, C)
    merged.index_add_(0, idx.reshape(B * N), (x * norm_weight).reshape(B * N, C).type(x.dtype))
    if assignment_out is not None:
        assignment_out['idx_cluster'] = idx_cluster
    return merged.reshape(B, cluster_num, C), index_down


# -- the controller --------------------------------------------------------
class LLMTokenPruner:
    """Installs FastV / SparseVLM on a loaded LLaVA-3D decoder.

    Everything is done with module hooks: the scoring layers are asked for
    their attention, and from the pruning layer onward the 4D attention mask
    gets the pruned visual columns set to the dtype minimum. No transformers
    source is replaced. One pre-hook per layer owns the mask, so a layer that
    was handed ``output_attentions=True`` still gets a causal mask.
    """

    def __init__(self, language_model, config):
        self.lm = language_model
        self.cfg = config
        self.num_layers = len(language_model.layers)
        self.handles = []
        self.reset()
        # ``retained`` is the effective count -- kept originals plus the
        # reconstructed tokens, which is the visual sequence the LLM gets.
        self.stats = {'entering': [], 'retained': [], 'average': [],
                      'kept_original': [], 'reconstructed': []}
        # SparseVLM must decide during prefill only; these two prove it.
        self.prefill_sparse_calls = 0
        self.decode_sparse_calls = 0

    def reset(self):
        self.dropped = None          # absolute positions masked out
        self.rater_abs = None
        self.counts = None           # kept originals per stage
        self.effectives = None       # kept + reconstructed per stage
        self.schedule_table = None   # which upstream RETAIN_TOKN table was used
        self.stage_log = []          # (stage, alive_before, kept, reconstructed)
        self.v_start = None
        self.v_num = None
        self.prompt_len = None
        # (member, centre) local visual indices for SparseVLM's recycled
        # clusters: a member's content reaches the LLM through its centre.
        self.recycled_pairs = []

    # -- installation ------------------------------------------------------
    def install(self):
        method = self.cfg['method']
        if method == 'fastv':
            source_layers = [self.cfg['fastv_layer']]
            self.first_pruned_layer = self.cfg['fastv_layer'] + 1
        else:
            source_layers = list(self.cfg['sparsevlm_layers'])
            self.first_pruned_layer = source_layers[0] + 1
        self.source_layers = set(source_layers)

        # One pre-hook per layer, from the first scoring layer to the end. It
        # does both jobs -- ask for attention, and hand the layer a 4D mask --
        # because on a scoring layer the two are inseparable: see
        # ``_make_layer_pre_hook``.
        for index in range(min(source_layers), self.num_layers):
            self.handles.append(self.lm.layers[index].register_forward_pre_hook(
                self._make_layer_pre_hook(index in self.source_layers), with_kwargs=True))

        for index in source_layers:
            self.handles.append(self.lm.layers[index].register_forward_hook(
                self._make_consume_attention(index), with_kwargs=True))

        # Layer 0 sees the untouched sequence: the right place to locate the
        # visual span and pick SparseVLM's raters.
        self.handles.append(self.lm.layers[0].register_forward_pre_hook(
            self._make_prefill_setup(), with_kwargs=True))
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    # -- hooks -------------------------------------------------------------
    def _make_prefill_setup(self):
        def hook(_module, args, kwargs):
            hidden_states = args[0] if args else kwargs['hidden_states']
            if hidden_states.shape[1] <= 1:
                return None                      # a generation step: keep the decision
            self.reset()
            start = PRUNE_STATE.get('img_start')
            length = PRUNE_STATE.get('img_len')
            if start is None or length is None or length <= 1:
                return None
            self.v_start, self.v_num = int(start), int(length)
            self.prompt_len = int(hidden_states.shape[1])
            if self.cfg['method'] == 'sparsevlm':
                self.rater_abs = sparsevlm_rater_indices(
                    hidden_states, self.v_start, self.v_start + self.v_num)
                recycle = self.cfg['sparsevlm_recycle']
                if self.cfg['sparsevlm_stages']:
                    self.counts = self.cfg['sparsevlm_stages']
                    self.effectives = _sparsevlm_effectives(
                        self.v_num, self.counts, recycle)
                elif self.cfg['sparsevlm_mode'] == 'official_v1':
                    self.counts, self.effectives, self.schedule_table = \
                        sparsevlm_official_v1_schedule(
                            self.v_num, self.cfg['tokens'], recycle)
                else:
                    self.counts, self.effectives = sparsevlm_schedule(
                        self.v_num, self.cfg['tokens'], self.num_layers,
                        self.cfg['sparsevlm_layers'], self.cfg['budget'], recycle)
            else:
                self.counts = (fastv_retained(
                    self.v_num, self.cfg['tokens'], self.num_layers,
                    self.first_pruned_layer, self.cfg['budget']),)
                self.effectives = self.counts
            self.stats['entering'].append(self.v_num)
            self.stats['kept_original'].append(self.counts[-1])
            self.stats['reconstructed'].append(self.effectives[-1] - self.counts[-1])
            self.stats['retained'].append(self.effectives[-1])
            layers = (self.cfg['sparsevlm_layers'] if self.cfg['method'] == 'sparsevlm'
                      else (self.first_pruned_layer,))
            self.stats['average'].append(
                layer_average(self.v_num, self.effectives, layers, self.num_layers))
            if self.cfg['debug']:
                head = f"[LLMPrune] {_sample_tag()}{self.cfg['method']}"
                if self.cfg['method'] == 'sparsevlm':
                    raters = 0 if self.rater_abs is None else int(self.rater_abs.numel())
                    text_tokens = int(hidden_states.shape[1]) - (self.v_start + self.v_num)
                    print(f"{head} version={self.cfg['sparsevlm_version']} "
                          f"mode={self.cfg['sparsevlm_mode']} "
                          f"table=RETAIN_TOKN_{self.schedule_table} "
                          f"initial_visual_tokens={self.v_num} "
                          f"num_text_tokens={text_tokens} num_raters={raters} "
                          f"schedule_kept={self.counts} schedule_effective={self.effectives}",
                          flush=True)
                else:
                    print(f"{head} visual={self.v_num} kept={self.counts} "
                          f"effective={self.effectives} "
                          f"avg/layer={self.stats['average'][-1]:.1f}", flush=True)
            return None
        return hook

    def _make_layer_pre_hook(self, is_source_layer):
        """Ask a scoring layer for its attention, and keep every layer causal.

        The two have to happen together. transformers picks the mask format
        once, before the layer loop, from the *model-level* ``output_attentions``
        (``LlamaModel.forward``): with the default sdpa backend and no padding,
        ``_prepare_4d_causal_attention_mask_for_sdpa`` returns ``None`` and
        relies on SDPA's own ``is_causal``. Forcing ``output_attentions=True``
        on one layer makes ``LlamaSdpaAttention`` fall back to the eager path,
        which applies a mask only ``if attention_mask is not None`` -- so that
        layer would run **bidirectionally**, and its hidden states feed every
        later layer. Both upstreams are causal there (SparseVLM builds a
        ``tril`` bias explicitly before its softmax), so the mask is
        materialised here whenever it is missing.
        """
        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get('hidden_states')
            if hidden_states is None:
                return None
            prefill = hidden_states.shape[1] > 1
            want_attention = is_source_layer and prefill and self.v_num is not None
            columns = (self.dropped if self.dropped is not None
                       else torch.empty(0, dtype=torch.long, device=hidden_states.device))
            if not want_attention and columns.numel() == 0:
                return None                      # nothing to force, nothing to mask

            if want_attention:
                kwargs['output_attentions'] = True   # sdpa falls back to eager here
            mask = self._materialise_mask(module, hidden_states, kwargs)
            if columns.numel():
                columns = columns[columns < mask.shape[-1]]
                mask[..., columns] = torch.finfo(mask.dtype).min
            kwargs['attention_mask'] = mask
            if columns.numel():
                self._renumber_positions(kwargs, hidden_states, prefill)
            return args, kwargs
        return hook

    def _materialise_mask(self, module, hidden_states, kwargs):
        """The layer's 4D additive mask, built from scratch if it is absent."""
        mask = kwargs.get('attention_mask')
        if mask is not None:
            return mask.clone()

        batch, q_len, _ = hidden_states.shape
        past = kwargs.get('past_key_value', kwargs.get('past_key_values'))
        past_len = 0
        if past is not None and hasattr(past, 'get_usable_length'):
            try:
                past_len = past.get_usable_length(q_len, module.self_attn.layer_idx)
            except Exception:
                past_len = 0
        kv_len = past_len + q_len
        dtype = hidden_states.dtype
        # sdpa was relying on is_causal; materialise the causal mask so the
        # pruned columns can be removed from it.
        mask = torch.full((q_len, kv_len), torch.finfo(dtype).min, dtype=dtype,
                          device=hidden_states.device)
        mask = mask.triu(diagonal=1 + past_len)
        return mask[None, None, :, :].expand(batch, 1, q_len, kv_len).clone()

    def _renumber_positions(self, kwargs, hidden_states, prefill):
        """SparseVLM V1's ``position_ids = position_ids[:, :new_length]``.

        Upstream slices the pruned rows out and then *truncates* the position
        ids, so every survivor is renumbered 0..L-1 and the text after the
        visual span slides down by however many tokens were dropped. Keeping
        the original ids is what upstream added in V2 ("idea1: Keep Position
        ID"), so a V1 port has to renumber.

        This port masks instead of slicing, so the renumbering is done in
        place: a survivor gets its rank among the survivors, and a pruned token
        gets whatever falls out -- nothing attends to it, so its position never
        reaches another token's attention.
        """
        if (self.cfg['method'] != 'sparsevlm'
                or self.cfg['sparsevlm_position_ids'] != 'renumber'
                or self.prompt_len is None):
            return
        position_ids = kwargs.get('position_ids')
        if position_ids is None:
            return
        dropped = int(self.dropped.numel())
        if prefill:
            alive = torch.ones(self.prompt_len, dtype=torch.bool,
                               device=hidden_states.device)
            alive[self.dropped] = False
            ranks = torch.cumsum(alive.long(), 0) - 1          # survivor j -> j
            kwargs['position_ids'] = ranks.clamp(min=0)[None].expand_as(position_ids)
        else:
            # A generated token sits at (compacted prompt length + steps so far).
            kwargs['position_ids'] = (position_ids - dropped).clamp(min=0)

    def _make_consume_attention(self, layer_index):
        def hook(_module, args, kwargs, output):
            hidden_states = args[0] if args else kwargs.get('hidden_states')
            if hidden_states is None or self.v_num is None:
                return None
            if hidden_states.shape[1] <= 1:
                # A decode step reached a scoring layer: upstream V1 also skips
                # here (``hidden_states.shape[1] != 1`` guards its whole block),
                # so this must stay at zero.
                self.decode_sparse_calls += 1
                return None
            self.prefill_sparse_calls += 1
            attn = output[1] if len(output) > 1 else None
            if attn is None:
                return None
            alive = self._alive_visual_positions(hidden_states.device)
            stage = 0
            if self.cfg['method'] == 'fastv':
                scores = fastv_scores(attn.float(), self.v_start, self.v_num)
                keep = self.counts[0]
            else:
                stage = self.cfg['sparsevlm_layers'].index(layer_index)
                if self.rater_abs is None:
                    return None
                scores = sparsevlm_scores(attn.float(), self.rater_abs, self.v_start, self.v_num)
                keep = self.counts[stage]
            # Only tokens still alive can be kept, and upstream never keeps
            # more than ``v_token_num - 1``.
            masked_scores = torch.full_like(scores, float('-inf'))
            local_alive = alive - self.v_start
            masked_scores[local_alive] = scores[local_alive]
            alive_now = int(local_alive.numel())
            last_stage = stage == len(self.counts) - 1
            if (self.cfg['method'] == 'sparsevlm' and last_stage
                    and self.cfg['sparsevlm_mode'] == 'fixed_budget'
                    and self.cfg['budget'] == 'retained'
                    and not self.cfg['sparsevlm_stages']):
                # Exact-budget control, SparseVLM 3.4's final stage. The count
                # is re-solved against what really survived, so kept originals
                # plus reconstructed tokens land on the target; the ranking is
                # still the rater attention, never a new criterion.
                keep = sparsevlm_keep_for_effective(
                    alive_now, self.cfg['tokens'], self.cfg['sparsevlm_recycle'])
            keep = max(1, min(keep, alive_now - 1)) if alive_now > 1 else 1
            kept_local = masked_scores.topk(keep).indices

            newly_dropped = torch.ones(self.v_num, dtype=torch.bool, device=scores.device)
            newly_dropped[kept_local] = False
            newly_dropped[(masked_scores == float('-inf'))] = False   # already dropped
            new_hidden = None
            recycle_candidates = 0
            if (self.cfg['method'] == 'sparsevlm' and self.cfg['sparsevlm_recycle']
                    and newly_dropped.any()):
                n_dropped = int(newly_dropped.sum())
                recycle_candidates = min(int(n_dropped * 0.3) + 1, n_dropped)
                new_hidden, recycled = self._recycle(
                    output[0], scores, newly_dropped)
                if recycled is not None:
                    newly_dropped[recycled] = False       # centres stay alive

            dropped_local = torch.where(newly_dropped)[0]
            dropped_abs = dropped_local + self.v_start
            self.dropped = (dropped_abs if self.dropped is None
                            else torch.cat([self.dropped, dropped_abs]))

            if self.cfg['method'] == 'sparsevlm':
                effective = alive_now - int(dropped_local.numel())
                self.stage_log.append((stage, alive_now, keep, effective - keep))
                if last_stage and self.stats['retained']:
                    # What the LLM really got, replacing the prediction.
                    self.stats['kept_original'][-1] = keep
                    self.stats['reconstructed'][-1] = effective - keep
                    self.stats['retained'][-1] = effective
                if self.cfg['debug']:
                    live = scores[local_alive].float()
                    print(f"[LLMPrune] {_sample_tag()}stage={stage} layer={layer_index} "
                          f"visual_tokens_before={alive_now} "
                          f"num_raters={int(self.rater_abs.numel())} "
                          f"schedule_value={self.counts[stage]} "
                          f"kept_original={keep} deleted={int(dropped_local.numel())} "
                          f"recycle_candidates={recycle_candidates} "
                          f"reconstructed={effective - keep} "
                          f"after_effective={effective} "
                          f"score_min={live.min():.3e} score_max={live.max():.3e} "
                          f"score_mean={live.mean():.3e} score_std={live.std():.3e}",
                          flush=True)
            if new_hidden is not None:
                return (new_hidden,) + tuple(output[1:])
            return None
        return hook

    def _recycle(self, hidden_states, scores, newly_dropped):
        """SparseVLM 3.3, with each cluster written back into its centre's slot."""
        dropped_local = torch.where(newly_dropped)[0]
        take = int(dropped_local.shape[0] * 0.3) + 1
        take = min(take, dropped_local.shape[0])
        order = scores[dropped_local].topk(take).indices
        chosen_local = dropped_local[order]
        cluster_num = min(int(take / 10) + 1, take)
        if cluster_num <= 0:
            return None, None
        chosen_abs = chosen_local + self.v_start
        assignment = {}
        merged, centres = cluster_and_merge(
            hidden_states[:, chosen_abs, :].float(), cluster_num,
            assignment_out=assignment, reduce=self.cfg['sparsevlm_reconstruct'])
        centre_abs = chosen_abs[centres[0]]
        new_hidden = hidden_states.clone()
        new_hidden[:, centre_abs, :] = merged[0].to(hidden_states.dtype)
        centre_local = chosen_local[centres[0]]
        idx_cluster = assignment.get('idx_cluster')
        if idx_cluster is not None:
            # Who each recycled token now speaks through. A centre maps to
            # itself; the export drops those, since a centre is not merged.
            self.recycled_pairs.append(
                (chosen_local.detach().cpu(), centre_local[idx_cluster[0]].detach().cpu()))
        return new_hidden, centre_local

    def _alive_visual_positions(self, device):
        alive = torch.ones(self.v_num, dtype=torch.bool, device=device)
        if self.dropped is not None:
            alive[self.dropped - self.v_start] = False
        return torch.where(alive)[0] + self.v_start

    def selection_snapshot(self):
        """What the last prefill decided, for the Fig.4 export.

        Valid from the end of a prefill until the next one resets it, so the
        evaluators read it right after ``generate`` returns. These two methods
        are query-based: unlike every vision-path method in this repo, this
        answers a *question*, not a scene, and it changes from question to
        question over the same scene.

        ``kept_local`` indexes the visual tokens that entered the decoder --
        for these methods the voxel-pooled LLaVA-3D baseline tokens, not patch
        tokens. Pruning is done by masking, so "kept" means no later layer was
        stopped from attending to it.
        """
        if self.v_num is None or self.v_start is None:
            return None
        alive = torch.ones(self.v_num, dtype=torch.bool)
        if self.dropped is not None and self.dropped.numel():
            local = (self.dropped - self.v_start).detach().cpu()
            local = local[(local >= 0) & (local < self.v_num)]
            alive[local] = False
        pairs = []
        for members, centres in self.recycled_pairs:
            for member, centre in zip(members.tolist(), centres.tolist()):
                if member != centre and alive[centre]:
                    pairs.append((member, centre))
        return {
            'method': self.cfg['method'],
            'num_visual': int(self.v_num),
            'visual_start': int(self.v_start),
            'kept_local': torch.where(alive)[0],
            'recycled_pairs': pairs,
            'stage_counts': tuple(int(c) for c in (self.counts or ())),
            'prune_layers': (tuple(self.cfg['sparsevlm_layers'])
                             if self.cfg['method'] == 'sparsevlm'
                             else (self.cfg['fastv_layer'],)),
        }

    def summary(self):
        if not self.stats['entering']:
            return None
        def mean(values):
            return sum(values) / len(values)
        if self.cfg['method'] == 'sparsevlm':
            label = (f"sparsevlm {self.cfg['sparsevlm_version']}/"
                     f"{self.cfg['sparsevlm_mode']} RETAIN_TOKN={self.cfg['tokens']}")
        else:
            label = f"{self.cfg['method']} budget={self.cfg['tokens']} ({self.cfg['budget']})"
        return (f"[LLM Prune] {label}: "
                f"initial_visual_tokens={mean(self.stats['entering']):.1f}, "
                f"final_kept_original={mean(self.stats['kept_original']):.1f}, "
                f"final_reconstructed={mean(self.stats['reconstructed']):.1f}, "
                f"final_effective_visual_tokens={mean(self.stats['retained']):.1f}, "
                f"avg/layer={mean(self.stats['average']):.1f}, "
                f"n={len(self.stats['entering'])}, "
                f"prefill_sparse_calls={self.prefill_sparse_calls}, "
                f"decode_sparse_calls={self.decode_sparse_calls}")


_INSTALLED = None


def install_llm_pruning(model):
    """Install FastV / SparseVLM on ``model`` if ``LLM_PRUNE`` asks for it."""
    global _INSTALLED
    config = prune_config()
    if config is None:
        return None
    language_model = model.get_model()
    _INSTALLED = LLMTokenPruner(language_model, config).install()
    if config['method'] != 'sparsevlm':
        print(f"[LLMPrune] installed {config['method']} "
              f"(budget {config['tokens']} {config['budget']})", flush=True)
        return _INSTALLED

    print(f"[SparseVLM Version] version = {config['sparsevlm_version'].upper()}", flush=True)
    print('[SparseVLM Version] SparseVLM V2 / SparseVLM+ enabled: False', flush=True)
    schedule = ''
    if config['sparsevlm_stages'] is not None:
        schedule = f", schedule={config['sparsevlm_stages']} (SPARSEVLM_STAGES)"
    elif config['sparsevlm_mode'] == 'official_v1':
        table = (config['tokens'] if config['tokens'] in SPARSEVLM_SCHEDULES
                 else SPARSEVLM_DEFAULT_BUDGET)
        schedule = (f", schedule={SPARSEVLM_SCHEDULES[table]} (upstream V1's {table}-token "
                    f"table, used as absolute counts"
                    f"{'' if table == config['tokens'] else '; no table for this budget'})")
    else:
        table = sparsevlm_shape(config['tokens'], config['budget'])[1]
        schedule = (f", profile={SPARSEVLM_SCHEDULES[table]} ({table}-token table, "
                    f"rescaled to hit the {config['budget']} budget)")
    print(f"[LLMPrune] installed sparsevlm mode={config['sparsevlm_mode']} "
          f"RETAIN_TOKN={config['tokens']} layers={config['sparsevlm_layers']} "
          f"recycle={config['sparsevlm_recycle']} "
          f"reconstruct={config['sparsevlm_reconstruct']}{schedule}", flush=True)
    if config['sparsevlm_mode'] == 'official_v1':
        print("[LLMPrune] official_v1: upstream's counts are used as written and the "
              'final token count is NOT forced to RETAIN_TOKN. Use '
              'SPARSEVLM_MODE=fixed_budget for the K-token comparison row.', flush=True)
    return _INSTALLED


def installed():
    return _INSTALLED
