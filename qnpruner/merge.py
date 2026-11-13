"""Step 4 of QNPruner: NL2, noun-local merging into the Stage-1 anchors (token IDs unchanged, exactly K rows).

Every unselected valid voxel u is assigned to a(u) = argmax over ALL K selected tokens of cos(f_fused_u, f_fused_i)
(pre-projector fused features). It is merged only if a(u) is a Stage-1 anchor and hop(a(u), u) <= 2 on the
26-neighbour graph of the valid voxels (BFS); otherwise it is dropped. A merged anchor's row becomes the uniform mean
of its own row and its sources; Stage-2 tokens are left unchanged.
"""
import math

import torch
import torch.nn.functional as F

MAX_HOP = 2
UNREACHED = 32767
OFFSETS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]  # 26


def neighbour_table(grid, valid):
    """(N, 26) index of the valid 26-neighbours on the voxel grid, -1 = none."""
    g = grid.long().cpu()
    n = g.shape[0]
    ok = valid.cpu().bool()
    g = g - g.min(0).values + 1
    dims = g.max(0).values + 2

    def enc(c):
        return (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
    keys, order = torch.sort(enc(g))
    cols = []
    for off in OFFSETS:
        k2 = enc(g + torch.tensor(off))
        pos = torch.searchsorted(keys, k2).clamp(max=n - 1)
        hit = keys[pos] == k2
        j = torch.where(hit, order[pos], torch.full_like(pos, -1))
        good = hit & ok & ok[j.clamp(min=0)]
        cols.append(torch.where(good, j, torch.full_like(j, -1)))
    return torch.stack(cols, 1)


def hop_distance(anchors, nbr, hmax=MAX_HOP):
    """(K, N) int16 BFS hop distance from each anchor (UNREACHED beyond hmax)."""
    anchors = anchors.cpu().long()
    k, n = anchors.numel(), nbr.shape[0]
    ok = nbr >= 0
    nbc = nbr.clamp(min=0)
    H = torch.full((k, n), UNREACHED, dtype=torch.int16)
    reach = torch.zeros(k, n, dtype=torch.bool)
    reach[torch.arange(k), anchors] = True
    H[reach] = 0
    t = 0
    while t < hmax:
        t += 1
        new = (reach[:, nbc] & ok[None]).any(2) & ~reach
        if not bool(new.any()):
            break
        H[new] = t
        reach |= new
    return H


def merge(fused, keep, stage1, valid, grid):
    """fused (N, C) pre-projector rows, keep (K,) selected voxel ids (sorted), stage1 = Stage-1 anchor ids.
    Returns the (K, C) float32 merged rows in ``keep`` order."""
    dev = fused.device
    keep = keep.to(dev).long()
    n, k = fused.shape[0], keep.numel()
    is_sel = torch.zeros(n, dtype=torch.bool, device=dev)
    is_sel[keep] = True
    src = torch.nonzero(valid.to(dev) & ~is_sel).squeeze(1)
    G = fused.float()
    # assignment: global cosine argmax over all K tokens
    a = (F.normalize(G[src], dim=1) @ F.normalize(G[keep], dim=1).T).argmax(1)
    A = torch.zeros(src.numel(), k, dtype=torch.bool, device=dev)
    A[torch.arange(src.numel(), device=dev), a] = True
    # eligibility: within MAX_HOP hops of the assigned token, and the token is a Stage-1 anchor
    H = hop_distance(keep, neighbour_table(grid, valid), MAX_HOP)
    hu = H[a.cpu(), src.cpu()].long()
    A &= ((hu <= MAX_HOP) & (hu < UNREACHED)).to(dev)[:, None]
    s1 = set(int(x) for x in stage1)
    A &= torch.tensor([int(i) in s1 for i in keep.cpu().tolist()], dtype=torch.bool, device=dev)[None, :]
    # uniform mean of the anchor row and its sources
    W = A.float()
    wsum = W.sum(0)
    mean = (W.T @ G[src]) / wsum.clamp(min=1e-12)[:, None]
    own = G[keep]
    upd = wsum > 0
    alpha = torch.where(upd, 1.0 / (1.0 + wsum), torch.ones_like(wsum))
    return alpha[:, None] * own + (1 - alpha[:, None]) * torch.where(upd[:, None], mean, own)
