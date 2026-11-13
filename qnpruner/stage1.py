"""Step 2 of QNPruner: Stage 1, query-noun anchors (kb = round(rho K) tokens).

Noun groups   voxel u -> argmax_j r_j(u), kept iff max_j r_j(u) >= tau (scene median over the valid voxels)
Noun quotas   equal shares of kn over the nouns (largest remainder, ties -> earlier noun); kn = round(kb / 2) when the
              question has attribute phrases, else kn = kb
Exploration   per noun j: 3D components of its group (single linkage <= 0.35 m); protect the ceil(q/2) most relevant
              members, then one pick (most relevant member) per not-yet-covered component of size >= 2, the component
              chosen by peak relevance x appearance novelty (1 - max cos of mean f2d to the covered components); then
              relevance order
Phrases       the rest kb - |S1| over the attribute phrases equally, each by the fused column
              (1 - w) z(r_head) + w z(r_phrase), w = s_head / (s_head + s_phrase), s = mean pairwise 3D distance of
              the top-16 voxels of that map
Fill          any shortfall by max_j r_j (most relevant valid voxels not yet taken)
All ties are broken toward the lower voxel index (stable sorts).
"""
import math

import torch
import torch.nn.functional as F

LINK = 0.35          # m, 3D component link (26-neighbourhood of 0.2 m voxels)
SPREAD_TOP = 16


def ranked(x):
    """Indices by descending value; equal values keep the lower index first."""
    return torch.argsort(-x, stable=True)


def apportion(total, weights):
    """Integers summing to ``total`` proportional to ``weights`` (largest remainder; ties -> earlier entry)."""
    n = len(weights)
    if n == 0:
        return []
    w = torch.as_tensor(weights, dtype=torch.float64)
    w = w / w.sum() if float(w.sum()) > 0 else torch.full((n,), 1.0 / n, dtype=torch.float64)
    raw = w * total
    floor = torch.floor(raw).long()
    out = [int(f) for f in floor]
    frac = (raw - floor).tolist()
    for i in sorted(range(n), key=lambda i: (-frac[i], i))[:total - int(floor.sum())]:
        out[i] += 1
    return out


def relevance_fill(best, valid, taken, need):
    have, out = set(taken), []
    for t in ranked(best.masked_fill(~valid, float('-inf'))).tolist():
        if len(out) >= need:
            break
        if valid[t] and t not in have:
            out.append(t)
    return out


def components(x, link=LINK):
    """Connected-component labels of points x (n, 3) under single linkage d <= link."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    if x.shape[0] <= 1:
        return torch.zeros(x.shape[0], dtype=torch.long)
    _, lab = connected_components(csr_matrix((torch.cdist(x, x) <= link).numpy()), directed=False)
    return torch.as_tensor(lab, dtype=torch.long)


def noun_groups(score, valid):
    """(best (N,), tau, group id (N,) in {-1, 0..J-1})."""
    best, arg = score.masked_fill(~valid[:, None], float('-inf')).max(dim=1)
    tau = float(torch.quantile(best[valid].float(), 0.5))
    gid = torch.where(valid & (best >= tau), arg, torch.full_like(arg, -1))
    return best, tau, gid


def explore_order(members, r, q, xyz, z):
    """Selection order (voxel ids) of one noun group: protected top ceil(q/2), then component exploration, then
    relevance. members (n,) voxel ids, r (n,) relevance, xyz (N, 3) / z (N, D) unit f2d of all voxels."""
    x, zm = xyz[members], z[members]
    lab = components(x)
    order = ranked(r).tolist()
    comps = [c for c in range(int(lab.max()) + 1) if int((lab == c).sum()) >= 2] if lab.numel() else []
    h = math.ceil(q / 2)
    sel = order[:h]
    mean_z = {c: F.normalize(zm[lab == c].mean(0), dim=0) for c in comps}
    covered = set(int(lab[i]) for i in sel)
    while len(sel) < q:
        cand = [c for c in comps if c not in covered]
        if not cand:
            break
        cov_z = [mean_z[c] for c in covered if c in mean_z]

        def key(c):
            novelty = 1 - max((float(mean_z[c] @ y) for y in cov_z), default=0.0)
            return float(r[lab == c].max()) * novelty
        c = max(cand, key=key)
        mem = torch.nonzero(lab == c).squeeze(1)
        sel.append(int(mem[torch.argmax(r[mem])]))
        covered.add(c)
    chosen = set(sel)
    sel += [i for i in order if i not in chosen]
    return [int(members[i]) for i in sel]


def noun_anchors(score, valid, kb, xyz, z):
    """kb noun anchors: equal quotas, per-noun exploration order, relevance fill. Returns (picks, best, gid)."""
    best, tau, gid = noun_groups(score, valid)
    quotas = apportion(kb, [1.0] * score.shape[1])
    picks, taken = [], set()
    for j in range(score.shape[1]):
        m = torch.nonzero(gid == j).squeeze(1)
        if quotas[j] <= 0 or m.numel() == 0:
            continue
        order = explore_order(m, score[m, j], quotas[j], xyz, z)
        pick = [u for u in dict.fromkeys(order) if u not in taken][:quotas[j]]
        taken.update(pick)
        picks += pick
    picks = list(dict.fromkeys(picks))
    return picks + relevance_fill(best, valid, picks, kb - len(picks)), best, gid


def _zscore(r, valid):
    v = r[valid]
    return (r - v.mean()) / v.std().clamp(min=1e-6)


def _spread(r, valid, xyz, k=SPREAD_TOP):
    top = torch.argsort(r.masked_fill(~valid, float('-inf')), descending=True)[:k]
    xs = xyz[top]
    return float(torch.cdist(xs, xs).sum() / (k * (k - 1)))


def stage1(score, phrase_score, phrase_heads, nouns, valid, kb, xyz, z):
    """Stage-1 anchors S1 (|S1| = kb). score (N, J) noun relevance, phrase_score (N, P) phrase relevance,
    phrase_heads[p] = lower-case head noun of phrase p, nouns = lower-case question nouns (column order)."""
    cols = []
    for p, head in enumerate(phrase_heads):
        if head not in nouns:
            continue
        rh, rp = score[:, nouns.index(head)], phrase_score[:, p]
        sh, sp = _spread(rh, valid, xyz), _spread(rp, valid, xyz)
        w = sh / max(sh + sp, 1e-9)
        cols.append((1 - w) * _zscore(rh, valid) + w * _zscore(rp, valid))
    if not cols:
        return noun_anchors(score, valid, kb, xyz, z)[0]
    picks, best, _ = noun_anchors(score, valid, int(round(0.5 * kb)), xyz, z)
    shares = apportion(kb - len(picks), [1.0] * len(cols))
    taken = set(picks)
    for col, share in zip(cols, shares):
        got = 0
        for u in ranked(col.masked_fill(~valid, float('-inf'))).tolist():
            if got >= share:
                break
            if valid[u] and u not in taken:
                picks.append(int(u))
                taken.add(int(u))
                got += 1
    return picks + relevance_fill(best, valid, picks, kb - len(picks))
