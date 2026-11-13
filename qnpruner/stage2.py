"""Step 3 of QNPruner: Stage 2, coverage completion to exactly K (kv = K - kb tokens), plus the no-noun path.

Local FL    m = round(0.25 kv) picks by a facility-location greedy on the appearance similarity
            Sz(u, i) = (cos(f2d_u, f2d_i) + 1) / 2; targets and candidates = valid voxels within 0.6 m of a Stage-1
            anchor (Stage-1 anchors excluded as candidates)
Global FL   the rest by  argmax_i  sum_u a_u max(0, S(u, i) - cur(u)) + 0.5 r_hat(i) / K  over all valid voxels,
            S = 0.5 (cos(P(f_fused)_u, P(f_fused)_i) + 1) / 2 + 0.5 Sz(u, i), a_u = 0.5 / |U|, r_hat = (max_j r_j + 1) / 2
No noun     max-min diversity on P(f_fused) for all K (first token = closest to the mean feature)
Greedy ties -> lower voxel index (torch.argmax / argmin return the first extremum).
"""
import torch
import torch.nn.functional as F

LOCAL_FRAC, LOCAL_RADIUS = 0.25, 0.6
REL_BONUS = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def coverage_similarity(v, feat_2d, vidx, dev=DEVICE):
    """(|U|, N) S = 0.5 (cos P(f_fused) + 1) / 2 + 0.5 (cos f2d + 1) / 2 of the valid voxels U to every voxel."""
    vi = vidx.to(dev)
    zf = F.normalize(v.float().to(dev), dim=1)
    Sf = (zf[vi] @ zf.T + 1) / 2
    z = F.normalize(feat_2d.float().to(dev), dim=1)
    Sz = (z[vi] @ z.T + 1) / 2
    return 0.5 * Sf + 0.5 * Sz


def local_fl(xyz, feat_2d, vidx, valid, init, m, dev):
    """Up to m local appearance-FL picks around the Stage-1 anchors ``init``."""
    if m <= 0 or not init:
        return []
    x = xyz.float().to(dev)
    vi = vidx.to(dev)
    it = torch.tensor(init, device=dev)
    near = torch.cdist(x[vi], x[it]).min(1).values <= LOCAL_RADIUS
    cand = torch.zeros(valid.numel(), dtype=torch.bool, device=dev)
    cand[vi[near]] = True
    cand[it] = False
    cand &= valid.to(dev)
    if int(near.sum()) == 0 or int(cand.sum()) == 0:
        return []
    z = F.normalize(feat_2d.float().to(dev), dim=1)
    Sz = (z[vi[near]] @ z.T + 1) / 2
    cur = Sz[:, it].max(1).values
    picks = []
    for _ in range(min(m, int(cand.sum()))):
        g = torch.clamp(Sz - cur[:, None], min=0).sum(0).masked_fill(~cand, float('-inf'))
        i = int(torch.argmax(g))
        picks.append(i)
        cand[i] = False
        cur = torch.maximum(cur, Sz[:, i])
    return picks


def global_fl(S, vidx, a_u, rel_gain, valid, k, init):
    """Greedy on sum_u a_u max_{i in T} S(u, i) + sum_{i in T} rel_gain_i from T = init until |T| = k."""
    dev = S.device
    a_u, rel_gain = a_u.to(dev), rel_gain.to(dev)
    cur = torch.zeros(vidx.numel(), device=dev)
    blocked = (~valid).clone().to(dev)
    order = []
    for i in init:
        order.append(int(i))
        blocked[i] = True
        cur = torch.maximum(cur, S[:, i])
    while len(order) < k:
        g = (a_u[:, None] * torch.clamp(S - cur[:, None], min=0)).sum(0) + rel_gain
        g = g.masked_fill(blocked, float('-inf'))
        i = int(torch.argmax(g))
        order.append(i)
        blocked[i] = True
        cur = torch.maximum(cur, S[:, i])
    return order


def stage2(entry, valid, r_hat, init, k_t):
    """Stage-1 anchors ``init`` -> K-token selection order (init first)."""
    init = [int(i) for i in init]
    vidx = torch.nonzero(valid).squeeze(1)
    S = coverage_similarity(entry['v'], entry['feat_2d'], vidx)
    a_u = torch.full((vidx.numel(),), (1 - REL_BONUS) / max(1, vidx.numel()))
    rel_gain = REL_BONUS * r_hat / max(1, k_t)
    loc = local_fl(entry['vox_xyz'], entry['feat_2d'], vidx, valid, init,
                   int(round(LOCAL_FRAC * (k_t - len(init)))), S.device)
    return global_fl(S, vidx, a_u, rel_gain, valid, k_t, init + loc)


def maxmin(fn, valid, need):
    """Max-min diversity of unit features fn (N, D): start at the voxel closest to the mean, then repeatedly the voxel
    whose maximum cosine to the chosen set is smallest."""
    if need <= 0:
        return []
    mean = F.normalize(fn[valid].mean(0, keepdim=True), dim=1)[0]
    first = int(torch.argsort(-(fn @ mean).masked_fill(~valid, float('-inf')), stable=True)[0])
    picks = [first]
    d = (fn @ fn[torch.tensor(picks)].T).max(dim=1).values
    blocked = ~valid.clone()
    blocked[torch.tensor(picks)] = True
    while len(picks) < need and not bool(blocked.all()):
        j = int(torch.argmin(d.masked_fill(blocked, float('inf'))))
        picks.append(j)
        blocked[j] = True
        d = torch.maximum(d, fn @ fn[j])
    return picks
