"""The forward pass' voxel rule, and mesh-vertex lookup on that grid.

The grid here must be *the* grid the video tower used, not a similar one.
``llava/model/multimodal_encoder/unproject.py::voxelization`` computes::

    xyz = torch.round(xyz / voxel_size).long()

and only then subtracts a per-scene minimum before hashing. The subtraction is
a relabelling of cell ids, not a change of grid, so the cell a point belongs to
is fully described by ``round(xyz / voxel_size)`` -- an absolute grid with no
scene-dependent offset. That is what :func:`xyz_to_voxel` reproduces, and why
no min-bound normalization appears anywhere in this file.

``np.round`` and ``torch.round`` both round half to even, so the two agree bit
for bit on the tie cases as well.
"""
from __future__ import annotations

import numpy as np


VOXEL_RULES = ('round', 'floor')


def xyz_to_voxel(xyz, voxel_size, rule='round'):
    """Map metric XYZ to integer voxel coordinates on the forward-pass grid.

    ``rule='round'`` is the one the model runs; ``'floor'`` exists so a run can
    be re-checked against the other convention rather than argued about.
    """
    if voxel_size <= 0:
        raise ValueError(f'voxel_size must be positive, got {voxel_size}')
    if rule not in VOXEL_RULES:
        raise ValueError(f'Unknown voxel rule {rule!r}; valid: {VOXEL_RULES}')
    scaled = np.asarray(xyz, dtype=np.float64) / float(voxel_size)
    grid = np.round(scaled) if rule == 'round' else np.floor(scaled)
    return grid.astype(np.int64)


def voxel_center(voxel_coords, voxel_size, rule='round'):
    """Metric centre of each voxel cell, for drawing or distance checks."""
    coords = np.asarray(voxel_coords, dtype=np.float64)
    offset = 0.0 if rule == 'round' else 0.5
    return (coords + offset) * float(voxel_size)


def pack_voxel_keys(voxel_coords):
    """One int64 key per voxel, collision-free for |coord| < 2**20 (~210 km)."""
    coords = np.asarray(voxel_coords, dtype=np.int64).reshape(-1, 3)
    if np.abs(coords).max(initial=0) >= (1 << 20):
        raise ValueError('Voxel coordinate out of range for int64 packing')
    shifted = coords + (1 << 20)
    return (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]


class MeshVoxelIndex:
    """``voxel -> mesh vertex indices`` on the forward pass' grid.

    Stored as a sorted key table plus a CSR-style index array instead of a
    Python dict: a ScanNet mesh has ~10^5 vertices and a query set of a few
    thousand voxels, and ``np.searchsorted`` answers all of them at once.
    """

    def __init__(self, vertex_xyz, voxel_size, rule='round'):
        self.voxel_size = float(voxel_size)
        self.rule = rule
        self.vertex_voxel = xyz_to_voxel(vertex_xyz, voxel_size, rule)   # (N, 3)
        keys = pack_voxel_keys(self.vertex_voxel)
        order = np.argsort(keys, kind='stable')
        sorted_keys = keys[order]
        unique_keys, starts = np.unique(sorted_keys, return_index=True)
        self.unique_keys = unique_keys
        self.starts = np.append(starts, sorted_keys.shape[0])
        self.order = order

    @property
    def num_voxels(self) -> int:
        return int(self.unique_keys.shape[0])

    @property
    def unique_coords(self) -> np.ndarray:
        """The distinct voxel coordinates the mesh occupies, as (M, 3) int64."""
        return self.vertex_voxel[self.order[self.starts[:-1]]]

    def vertices_in(self, voxel_coords):
        """Vertices falling in any of ``voxel_coords``.

        Returns ``(vertex_indices, matched_mask)`` where ``matched_mask[i]`` is
        True when ``voxel_coords[i]`` contains at least one mesh vertex. An
        unmatched voxel is not an error by itself -- the tokens come from 20
        sampled RGB-D views and can land slightly off the reconstructed surface
        -- but a *large* unmatched fraction means the grid or the frame is
        wrong, which is what the caller reports on.
        """
        query = pack_voxel_keys(np.asarray(voxel_coords, dtype=np.int64).reshape(-1, 3))
        position = np.searchsorted(self.unique_keys, query)
        in_range = position < self.unique_keys.shape[0]
        matched = np.zeros(query.shape[0], dtype=bool)
        matched[in_range] = self.unique_keys[position[in_range]] == query[in_range]
        if not matched.any():
            return np.zeros(0, dtype=np.int64), matched
        hit = position[matched]
        starts = self.starts[hit]
        counts = self.starts[hit + 1] - starts
        # Concatenate every [start, start + count) slice without a Python loop.
        total = int(counts.sum())
        group_offset = np.cumsum(counts) - counts
        within = np.arange(total, dtype=np.int64) - np.repeat(group_offset, counts)
        flat = np.repeat(starts, counts) + within
        return self.order[flat], matched


def match_report(index: MeshVoxelIndex, voxel_coords, label='selected', mesh_bounds=None):
    """Counts a caller can print and store; no side effects.

    ``mesh_bounds`` adds an out-of-box count. It separates the two reasons a
    selected voxel can miss the mesh: a depth-backprojection outlier that landed
    outside the reconstructed room at all, versus a cell inside the room that
    simply holds no surface. Only the second is worth worrying about.
    """
    voxel_coords = np.asarray(voxel_coords, dtype=np.int64).reshape(-1, 3)
    vertices, matched = index.vertices_in(voxel_coords)
    total = int(voxel_coords.shape[0])
    n_matched = int(matched.sum())
    # 'grid cell' rather than 'voxel': for DTC and VisPruner many selected
    # tokens can share one display cell, so these counts are not the method's
    # unit counts and must not be confused with them.
    report = {
        f'num_{label}_grid_cells': total,
        f'num_{label}_grid_cells_matched': n_matched,
        f'num_{label}_grid_cells_unmatched': total - n_matched,
        f'{label}_unmatched_ratio': 0.0 if total == 0 else (total - n_matched) / total,
        f'num_mesh_vertices_in_{label}_cells': int(vertices.shape[0]),
    }
    if total:
        # An unmatched cell whose neighbour holds surface is a near miss: the
        # frame and the grid are right and the token just landed a cell off
        # (bf16 token XYZ, depth noise, a hole in the reconstruction). An
        # unmatched cell with nothing around it is the signal that something is
        # actually wrong, so the two are counted apart rather than lumped.
        near = np.zeros(total, dtype=bool)
        missing = ~matched
        if missing.any():
            probes = voxel_coords[missing]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        _, hit = index.vertices_in(probes + np.array([dx, dy, dz]))
                        near[missing] |= hit
        report[f'num_{label}_grid_cells_near_surface'] = int(near.sum())
        report[f'num_{label}_grid_cells_isolated'] = int((missing & ~near).sum())
        report[f'{label}_isolated_ratio'] = float((missing & ~near).sum()) / total
    if mesh_bounds is not None and total:
        low, high = (np.asarray(b, dtype=np.float64) for b in mesh_bounds)
        centers = voxel_center(voxel_coords, index.voxel_size, index.rule)
        outside = ~np.all((centers >= low - index.voxel_size) &
                          (centers <= high + index.voxel_size), axis=1)
        report[f'num_{label}_grid_cells_outside_mesh_bbox'] = int((outside & ~matched).sum())
    return report, vertices, matched


def snap_to_surface(index: MeshVoxelIndex, voxel_coords, max_cells=3.0):
    """Give every selected cell a mesh cell of its own, as close as possible.

    Display only. Two things keep K tokens from showing as K cells: a token in
    a cell with no mesh surface (depth noise, a window, a hole in the scan) has
    nothing to paint, and two tokens can share one display cell. Each such
    token is drawn on the nearest mesh cell no other token already occupies.
    Cells farther than ``max_cells`` grid steps from any free surface cell are
    left where they are -- moving them would draw a place the token never was.

    ``voxel_coords`` is one row per token, duplicates included. Returns
    ``(coords, snapped_mask, distances_in_cells)``.
    """
    coords = np.asarray(voxel_coords, dtype=np.int64).reshape(-1, 3).copy()
    snapped = np.zeros(coords.shape[0], dtype=bool)
    distance = np.zeros(coords.shape[0], dtype=np.float64)
    if coords.shape[0] == 0 or index.num_voxels == 0:
        return coords, snapped, distance
    _, matched = index.vertices_in(coords)
    # The first token in a matched cell keeps it; later ones need a new cell.
    first = np.zeros(coords.shape[0], dtype=bool)
    _, first_index = np.unique(coords, axis=0, return_index=True)
    first[first_index] = True
    settled = matched & first
    surface = index.unique_coords
    taken = set(map(tuple, coords[settled].tolist()))
    free = np.array([c not in taken for c in map(tuple, surface.tolist())], dtype=bool)
    for i in np.where(~settled)[0]:
        if not free.any():
            break
        d = np.linalg.norm((surface - coords[i]).astype(np.float64), axis=1)
        d[~free] = np.inf
        j = int(np.argmin(d))
        if d[j] > max_cells:
            continue
        coords[i] = surface[j]
        free[j] = False
        snapped[i] = True
        distance[i] = d[j]
    return coords, snapped, distance
