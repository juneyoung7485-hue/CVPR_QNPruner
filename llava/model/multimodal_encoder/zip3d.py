"""3DZip's Stage 1 (DPP greedy MAP) and Stage 2 (spatially-cut-off cosine merge).

Lifted verbatim out of the ``3dzip`` branch of ``video_encoder.py`` so that
method packages can reuse them without copying; the branch itself calls these
same functions, so its output is unchanged (bit-identical to the original
3DZip repo, see ``tests/test_zip3d_equivalence.py``).
"""
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add


def cosine_kernel(vox_feat):
    """3DZip's DPP kernel: cosine similarity of the voxel features."""
    feat_norm = F.normalize(vox_feat.float(), dim=-1)
    return feat_norm @ feat_norm.T


def dpp_greedy_map(L, k):
    """Greedy MAP inference for a DPP with kernel ``L`` (Cholesky-based).

    Returns the ``k`` selected indices in pick order, as a LongTensor.
    Past the kernel's rank the gains are all zero and ``clamp_`` resets the
    ``-inf`` of already-picked items, so indices can repeat -- 3DZip's
    behaviour, kept as is.
    """
    n = L.shape[0]
    d = L.diag().clone()
    C = torch.zeros(k, n, device=L.device, dtype=torch.float32)
    selected = []
    for t in range(k):
        j = d.argmax().item()
        selected.append(j)
        if t < k - 1:
            c_j = L[j].clone()
            if t > 0:
                c_j -= C[:t, j] @ C[:t]
            C[t] = c_j / torch.sqrt(d[j].clamp(min=1e-8))
            d -= C[t] ** 2
            d.clamp_(min=0)
        d[j] = -float('inf')
    return torch.tensor(selected, device=L.device)


def merge_into_anchors(vox_feat, vox_xyz, sel_idx, k, voxel_size, merge_cutoff):
    """Fold every non-anchor voxel into its cosine-nearest anchor, if that
    anchor is within ``merge_cutoff`` voxel grids (or metres when
    ``voxel_size <= 0``); uniform average.

    Returns (merged_feat (k, C), anchor_xyz (k, 3), merge_trace), where
    ``merge_trace`` is (remain_idx, nearest_anchor, close_mask) or None.
    """
    n_vox = vox_feat.shape[0]
    anchor_feat = vox_feat[sel_idx]
    anchor_xyz = vox_xyz[sel_idx]
    mask = torch.ones(n_vox, dtype=torch.bool, device=vox_feat.device)
    mask[sel_idx] = False
    remain_idx = torch.where(mask)[0]

    merge_trace = None
    if remain_idx.numel() > 0:
        remain_feat = vox_feat[remain_idx]
        remain_xyz = vox_xyz[remain_idx]
        remain_norm = F.normalize(remain_feat.float(), dim=-1)
        anchor_norm = F.normalize(anchor_feat.float(), dim=-1)
        cos_dist = 1.0 - (remain_norm @ anchor_norm.T)
        _, nearest_anchor = cos_dist.min(dim=1)

        # Spatial cutoff: only merge if within MERGE_CUTOFF voxel grids
        if voxel_size > 0:
            anchor_grid = torch.floor(anchor_xyz / voxel_size)
            remain_grid = torch.floor(remain_xyz / voxel_size)
            xyz_grid_dist = torch.norm(remain_grid.float() - anchor_grid[nearest_anchor].float(), dim=-1)
            close_mask = xyz_grid_dist <= merge_cutoff
        else:
            xyz_dist = torch.norm(remain_xyz.float() - anchor_xyz[nearest_anchor].float(), dim=-1)
            close_mask = xyz_dist <= merge_cutoff

        merge_trace = (remain_idx, nearest_anchor, close_mask)

        if close_mask.any():
            close_anchor = nearest_anchor[close_mask]
            close_feat = remain_feat[close_mask]
            ones = torch.ones(close_anchor.shape[0], dtype=vox_feat.dtype, device=vox_feat.device)
            merge_sum = scatter_add(close_feat, close_anchor, dim=0, dim_size=k)
            merge_cnt = scatter_add(ones, close_anchor, dim=0, dim_size=k)
            total_cnt = 1.0 + merge_cnt
            merged_feat = (anchor_feat + merge_sum) / total_cnt.unsqueeze(-1)
        else:
            merged_feat = anchor_feat
    else:
        merged_feat = anchor_feat
    return merged_feat, anchor_xyz, merge_trace
