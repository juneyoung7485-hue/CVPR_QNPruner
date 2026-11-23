"""QNPruner end to end: scene voxel features -> query-noun selection of exactly K voxel tokens -> NL2 merged rows.

Called by LLaVA-3D's video tower (``POOLING=qnpruner``) through ``compress(tower, ctx)``; see method_registry.py.
"""
import json
import os

import torch
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean, scatter_min

from llava.model.multimodal_encoder.unproject import voxelize
from llava.sample_context import SAMPLE

from . import merge as MG
from . import relevance as REL
from . import stage1 as S1
from . import stage2 as S2
from . import text as TX

NAME = 'qnpruner'


# ------------------------------------------------------------------------------------------------ selection
def select(text, entry, projector, embed_tokens, tok, tok_name, k):
    """(selected voxel ids sorted (list), Stage-1 anchor ids (list), record)."""
    with torch.no_grad():
        P2d = projector(entry['feat_2d'].to(next(projector.parameters()).dtype)).float().cpu()
    entry['P2d'] = P2d
    v_cpu = entry['v'].detach().float().cpu()
    valid = (v_cpu.norm(dim=1) > 0) & (P2d.norm(dim=1) > 0)
    k_t = min(k, int(valid.sum()))

    # Step 0: parse (question sentence; the whole prompt when it cannot be isolated, see text.split_text)
    split = TX.split_text(text)
    nouns = TX.question_nouns(split)
    phrases = TX.attribute_phrases(split, nouns)
    qtype = TX.question_type(split)
    k_c = int(round(TX.RHO[qtype] * k_t))

    # Step 1: relevance (questions with a noun only; without one the selection is query-agnostic max-min)
    emb_n = [(w, REL.word_embedding(w, embed_tokens, tok, tok_name)) for w in nouns]
    emb_n = [(w, e) for w, e in emb_n if e is not None]
    emb_p = [(p, h, REL.word_embedding(p, embed_tokens, tok, tok_name)) for p, h in phrases]
    emb_p = [(p, h, e) for p, h, e in emb_p if e is not None]
    rec = {'question': text, 'nouns': [w for w, _ in emb_n], 'phrases': [p for p, _, _ in emb_p], 'qtype': qtype,
           'num_valid_voxels': int(valid.sum()), 'K_target': k_t, 'K_stage1': k_c}
    if emb_n:
        rel_all = REL.word_relevance(P2d, [e for _, e in emb_n] + [e for _, _, e in emb_p])
        score, phrase_score = rel_all[:, :len(emb_n)], rel_all[:, len(emb_n):]
        rel = score.masked_fill(~valid[:, None], -1.0).max(1).values
        r_hat = ((rel + 1) / 2).masked_fill(~valid, 0.0)

    # Step 2: Stage 1
    xyz = entry['vox_xyz'].float().cpu()
    if emb_n:
        z = F.normalize(entry['feat_2d'].float().cpu(), dim=1)
        base = S1.stage1(score, phrase_score, [h for _, h, _ in emb_p], [w.lower() for w, _ in emb_n],
                         valid, k_c, xyz, z) if k_c > 0 else []
    else:                                                      # no noun: max-min K tokens = generic NL2 anchors
        base = S2.maxmin(F.normalize(v_cpu, dim=1), valid, k_t)
        rec['K_stage1'] = len(base)                            # all k_t tokens are the generic anchors
    # Step 3: Stage 2 (skipped without a noun: max-min already chose all K tokens)
    if not emb_n:
        keep = list(base)
    else:
        keep = S2.stage2(entry, valid, r_hat, base, k_t) if k_t - k_c > 0 else list(base)
    if len(keep) != k_t or len(set(keep)) != k_t or not bool(valid[torch.tensor(keep)].all()):
        raise AssertionError(f'{NAME}: {len(keep)} picks ({len(set(keep))} unique) != {k_t}')
    sel = sorted(keep)
    rec.update(cluster_indices=sorted(base), selected_indices=sel, qa_order=sel)
    return sel, sorted(base), rec


# ------------------------------------------------------------------------------------------------ tower hook
def _grid(feat_xyz_b, p2v_b, voxel_size, n):
    cell = torch.round(feat_xyz_b.reshape(-1, 3) / voxel_size).long()
    hi = scatter_max(cell, p2v_b, dim=0, dim_size=n)[0]
    lo = scatter_min(cell, p2v_b, dim=0, dim_size=n)[0]
    assert torch.equal(hi, lo), f'{NAME}: the points of one voxel fall in different grid cells'
    return hi


def compress(tower, ctx):
    """LLaVA-3D tower hook -> (pooled (K, C) rows, [xyz (K, 3)], batch_offset). Batch size 1."""
    assert ctx['B'] == 1, f'{NAME}: batch size 1 only'
    if tower.voxel_size <= 0:
        raise ValueError(f'{NAME} selects among voxel tokens (VOXEL_SIZE > 0)')
    k = int(tower.target_tokens)
    if k < 1:
        raise ValueError(f'{NAME}: token budget K={tower.target_tokens} must be a positive integer (TARGET_TOKENS)')
    text = (ctx['query_texts'] or [None])[0]
    if text is None:
        raise RuntimeError(f'{NAME}: no question text reached the tower')
    embed = getattr(tower, '_llm_embed_tokens', None)
    if embed is None:
        raise RuntimeError(f'{NAME}: the LLM embedding table was not handed to the video tower')
    tok_name = os.environ.get('OURS_TOKENIZER', 'ChaimZhu/LLaVA-3D-7B')
    tok = REL.tokenizer(tok_name)
    projector, video_features = ctx['projector'], ctx['video_features']

    # LLaVA-3D voxelization: the voxel tokens the uncompressed model would feed the LLM
    p2v = voxelize(ctx['feat_xyz'], tower.voxel_size)[0].long()
    n = int(p2v.max()) + 1
    feat_fused = scatter_mean(video_features[0], p2v, dim=0, dim_size=n)          # CLIP L-2 + 3D PE, per voxel
    feat_2d = scatter_mean(ctx['visual_2d_features'][0], p2v, dim=0, dim_size=n)  # CLIP L-2, per voxel
    vox_xyz = scatter_mean(ctx['video_xyz'][0], p2v, dim=0, dim_size=n)
    with torch.no_grad():
        v = projector(feat_fused.to(next(projector.parameters()).dtype)).float()
    entry = {'feat_fused': feat_fused, 'feat_2d': feat_2d, 'vox_xyz': vox_xyz, 'v': v,
             'grid': _grid(ctx['feat_xyz'][0], p2v, tower.voxel_size, n)}

    sel, s1, rec = select(text, entry, projector, embed[0], tok, tok_name, k)
    keep = torch.tensor(sel, dtype=torch.long, device=v.device)
    valid = (v.float().norm(dim=1) > 0) & (entry['P2d'].to(v.device).norm(dim=1) > 0)
    rows = MG.merge(feat_fused, keep, s1, valid, entry['grid'])
    rec.update(question_id=SAMPLE.get('question_id'), scene_id=SAMPLE.get('scene_id'), method=NAME, N=n)

    stats = os.environ.get('OURS_STATS_FILE')
    if stats:
        os.makedirs(os.path.dirname(stats) or '.', exist_ok=True)
        with open(stats, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
    pooled = rows.to(feat_fused.dtype)
    batch_offset = torch.tensor([keep.numel()], dtype=torch.int32, device=video_features.device).cumsum(0)
    return pooled, [vox_xyz[keep]], batch_offset
