"""
woodpile_helpers.py
===================
Voxelized woodpile photonic crystal with controlled rod-segment defects
(S. Aeby, G. J. Aubry, N. Muller, F. Scheffold, Adv. Optical Mater. 9, 2001699 (2021)).

Main entry point: create_woodpile_dist(...) -> eps, rods, defects, ff, info
Companion notebook: create_woodpile_dist.ipynb

The membership test (circular test in the unwarped space z' = z / s, global z-scale
s = aspect_ratio) is the one of create_permittivity_grid_penlike from
LSU Project/20251001_LSU_Localization_Tests/20250903_create_h5_from_ends.ipynb (voxelize_rod).
Because woodpile rods are axis-aligned the test is separable, so the grid is built from 2-D
cross-section masks broadcast along the rod axis (voxelize_rod_axis / _layer_masks) and the
filling-fraction bisection never touches the 3-D grid (perfect_voxel_ff).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# AutomationModule lives in the root of the tidy3d project
sys.path.append(os.path.abspath(r'H:\codes\tidy3d'))
import AutomationModule as AM

__all__ = ['create_woodpile_dist', 'build_woodpile_rods', 'enumerate_segments', 'place_defects',
           'segment_centres', 'stratify_equal_mass', 'structure_factor', 'voxelize_woodpile', 'voxelize_rod', 'voxelize_rod_axis', 'perfect_voxel_ff', 'grid_coordinates', 'tables_to_dict', 'show_slice',
           'ROD_DTYPE', 'DEFECT_DTYPE']


ROD_DTYPE = np.dtype([
    ('x1', 'f8'), ('y1', 'f8'), ('z1', 'f8'), ('x2', 'f8'), ('y2', 'f8'), ('z2', 'f8'),
    ('layer', 'i4'), ('orientation', 'U1'), ('position', 'f8'), ('z', 'f8'),
    ('minor_radius', 'f8'), ('major_radius', 'f8'),
    ('seg_origin', 'f8'),        # segment boundaries of this rod sit at seg_origin + j*d
])

DEFECT_DTYPE = np.dtype([
    ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),                                   # segment centre
    ('x1', 'f8'), ('y1', 'f8'), ('z1', 'f8'), ('x2', 'f8'), ('y2', 'f8'), ('z2', 'f8'),  # segment endpoints
    ('layer', 'i4'), ('orientation', 'U1'), ('rod_index', 'i4'), ('segment_index', 'i4'),
    ('kappa', 'f8'), ('minor_radius', 'f8'), ('major_radius', 'f8'),
])


def _as_triple(v, cast=float):
    """Broadcast a scalar to (v, v, v); pass a length-3 sequence through."""
    if np.isscalar(v):
        return (cast(v),) * 3
    v = tuple(v)
    if len(v) != 3:
        raise ValueError("expected a scalar or a length-3 sequence")
    return tuple(cast(x) for x in v)


def grid_coordinates(box_size, grid_size):
    """Voxel-centre coordinates per axis: (arange(N) + 0.5) * dx - L/2 (box centred at origin)."""
    box = _as_triple(box_size, float)
    grid = _as_triple(grid_size, int)
    return [(np.arange(N, dtype=np.float64) + 0.5) * (L / N) - L / 2.0 for L, N in zip(box, grid)]


def voxelize_rod(eps, coords, p1, p2, b, s, value):
    """
    Write `value` into every voxel of `eps` whose centre lies inside the elliptical cylinder
    from p1 to p2 (world coordinates, ends cut flat).

    Same core as create_permittivity_grid_penlike: a RIGHT CIRCULAR cylinder of radius b is
    built in the unwarped space z' = z / s and the global z-scale s turns the cross-section
    into an ellipse with semi-axes b (in-plane) and a = s*b (along z).  Only the padded
    axis-aligned bounding box of the rod is visited.

    Returns the number of voxels written.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p1u = p1.copy(); p1u[2] /= s
    p2u = p2.copy(); p2u[2] /= s
    vu = p2u - p1u
    Lu = float(np.sqrt(np.dot(vu, vu)))
    if Lu <= 0.0 or b <= 0.0:
        return 0
    nu = vu / Lu

    pad = b * max(1.0, s)
    sl = []
    for ax in range(3):
        c = coords[ax]
        dx = float(c[1] - c[0]) if len(c) > 1 else 0.0
        lo = min(p1[ax], p2[ax]) - pad - dx
        hi = max(p1[ax], p2[ax]) + pad + dx
        i0 = max(int(np.searchsorted(c, lo, side='left')), 0)
        i1 = min(int(np.searchsorted(c, hi, side='right')), len(c))
        if i1 <= i0:
            return 0
        sl.append(slice(i0, i1))

    X, Y, Z = np.meshgrid(coords[0][sl[0]], coords[1][sl[1]], coords[2][sl[2]], indexing='ij')
    RX = X - p1u[0]
    RY = Y - p1u[1]
    RZ = Z / s - p1u[2]                    # unwarped z'
    t = RX * nu[0] + RY * nu[1] + RZ * nu[2]
    rX = RX - t * nu[0]
    rY = RY - t * nu[1]
    rZ = RZ - t * nu[2]
    mask = (t >= 0.0) & (t <= Lu) & (rX * rX + rY * rY + rZ * rZ <= b * b)
    n = int(np.count_nonzero(mask))
    if n:
        sub = eps[sl[0], sl[1], sl[2]]     # view -> in-place write
        sub[mask] = value
    return n


def _ellipse_mask_2d(c_perp, c_z, pos, z0, b, s):
    """
    2-D cross-section mask of an axis-aligned rod: voxel centres (perp, z) inside the ellipse of
    semi-axes b (in-plane) and s*b (along z) centred at (pos, z0).  Returns (slice_perp, slice_z, mask)
    restricted to the padded bounding box.  Same test as voxelize_rod in the unwarped space z' = z/s,
    which for an axis-aligned rod reduces to (perp - pos)^2 + (z/s - z0/s)^2 <= b^2.
    """
    pad = b * max(1.0, s)
    sl = []
    for c, centre in ((c_perp, pos), (c_z, z0)):
        dx = float(c[1] - c[0]) if len(c) > 1 else 0.0
        i0 = max(int(np.searchsorted(c, centre - pad - dx, side='left')), 0)
        i1 = min(int(np.searchsorted(c, centre + pad + dx, side='right')), len(c))
        sl.append(slice(i0, i1))
    if sl[0].stop <= sl[0].start or sl[1].stop <= sl[1].start:
        return sl[0], sl[1], None
    rP = c_perp[sl[0]] - pos
    rZ = c_z[sl[1]] / s - z0 / s
    mask = (rP * rP)[:, None] + (rZ * rZ)[None, :] <= b * b
    return sl[0], sl[1], mask


def voxelize_rod_axis(eps, coords, orientation, s0, s1, pos, z0, b, s, value):
    """
    Fast path of voxelize_rod for a rod parallel to x (orientation 'x') or y ('y') running from
    s0 to s1 along its axis at in-plane position `pos` and height z0.  The membership test is
    separable, so only a 2-D ellipse mask is built and broadcast along the rod axis (no 3-D meshgrid).
    Returns the number of voxels written.
    """
    if b <= 0.0 or s1 <= s0:
        return 0
    ax = 0 if orientation == 'x' else 1
    # inclusive clamp expressed exactly like the t-test of voxelize_rod: t = c - s0, 0 <= t <= L
    t = coords[ax] - s0                                    # monotone, so searchsorted applies
    sl_ax = slice(int(np.searchsorted(t, 0.0, side='left')), int(np.searchsorted(t, s1 - s0, side='right')))
    if sl_ax.stop <= sl_ax.start:
        return 0
    sl_p, sl_z, mask = _ellipse_mask_2d(coords[1 - ax], coords[2], pos, z0, b, s)
    if mask is None:
        return 0
    n_sec = int(np.count_nonzero(mask))
    if n_sec == 0:
        return 0
    if ax == 0:
        sub = eps[sl_ax, sl_p, sl_z]                       # (n_ax, n_perp, n_z) view
    else:
        sub = np.moveaxis(eps[sl_p, sl_ax, sl_z], 1, 0)    # view with the rod axis first
    sub[:, mask] = value                                   # broadcast along the axis, in place
    return n_sec * (sl_ax.stop - sl_ax.start)


def _layer_masks(rods, coords, s, skip=()):
    """
    Union cross-section masks of all full-length rods: Mx (Ny, Nz) for x-rods, My (Nx, Nz) for
    y-rods.  A voxel (i, j, k) of the perfect crystal is filled iff Mx[j, k] or My[i, k].
    Rods whose index is in `skip` (e.g. rods carrying defects) are left out.
    """
    Nx, Ny, Nz = (len(c) for c in coords)
    Mx = np.zeros((Ny, Nz), dtype=bool)
    My = np.zeros((Nx, Nz), dtype=bool)
    for i, rod in enumerate(rods):
        if i in skip:
            continue
        b = float(rod['minor_radius'])
        if b <= 0.0:
            continue
        if rod['orientation'] == 'x':
            sl_p, sl_z, m = _ellipse_mask_2d(coords[1], coords[2], rod['position'], rod['z'], b, s)
            if m is not None:
                Mx[sl_p, sl_z] |= m
        else:
            sl_p, sl_z, m = _ellipse_mask_2d(coords[0], coords[2], rod['position'], rod['z'], b, s)
            if m is not None:
                My[sl_p, sl_z] |= m
    return Mx, My


def _rods_span_box(rods, box_size, tol=1e-9):
    Lx, Ly, Lz = _as_triple(box_size, float)
    L_par = np.where(rods['orientation'] == 'x', Lx, Ly)
    lo = np.where(rods['orientation'] == 'x', rods['x1'], rods['y1'])
    hi = np.where(rods['orientation'] == 'x', rods['x2'], rods['y2'])
    return bool(np.all(np.abs(lo + L_par / 2) <= tol) and np.all(np.abs(hi - L_par / 2) <= tol))


def perfect_voxel_ff(rods, box_size, grid_size, aspect_ratio):
    """
    Voxel filling fraction of the defect-free woodpile WITHOUT building the 3-D grid.
    All rods span the full box along their axis, so per z-slab k the filled voxels are
    Nx*cx[k] + Ny*cy[k] - cx[k]*cy[k] with cx[k] = #j: Mx[j, k], cy[k] = #i: My[i, k]
    (inclusion-exclusion of the x-rod and y-rod unions).  Identical to mean(eps != background)
    of voxelize_woodpile for the same rods; O(Ny*Nz + Nx*Nz) instead of O(Nx*Ny*Nz).
    """
    if not _rods_span_box(rods, box_size):
        raise ValueError("perfect_voxel_ff requires rods spanning the full box along their axis")
    grid = _as_triple(grid_size, int)
    coords = grid_coordinates(box_size, grid)
    Mx, My = _layer_masks(rods, coords, float(aspect_ratio))
    Nx, Ny, Nz = grid
    cx = Mx.sum(axis=0, dtype=np.int64)
    cy = My.sum(axis=0, dtype=np.int64)
    filled = int(np.sum(Nx * cx + Ny * cy - cx * cy))
    return filled / float(Nx * Ny * Nz)


def build_woodpile_rods(box_size, d, dz, minor_radius, aspect_ratio=2.8, layer_offset=0.0,
                        segment_ref='below'):
    """
    Rod list of a woodpile lattice (no voxelization).

    Layer k (k = 0, 1, ...) is centred at z_k = -Lz/2 + h/2 + layer_offset + k*h with h = dz/4,
    and only layers whose centre lies inside the box are kept.  Even layers carry rods along x,
    odd layers rods along y; every second layer of the same orientation (k//2 odd) is shifted
    in-plane by d/2.  Rods sit at in-plane positions n*d + shift and every rod whose
    cross-section intersects the box is kept (|pos| < L/2 + b) so partial rods at the box
    boundary are represented.  Rods span the full box along their axis.

    seg_origin: rod segments of layer k are bounded by the crossings with the rods of the
    layer below (k-1) [segment_ref='below'] or above (k+1) ['above']; the other neighbouring
    layer then crosses every segment at its midpoint.  Layer 0 (or the top layer) uses its
    only neighbour.
    """
    Lx, Ly, Lz = _as_triple(box_size, float)
    b = float(minor_radius)
    s = float(aspect_ratio)
    a = s * b
    h = dz / 4.0

    def shift_of(k):
        return 0.0 if (k // 2) % 2 == 0 else d / 2.0

    z0 = -Lz / 2.0 + h / 2.0 + layer_offset
    n_layers = int(np.floor((Lz / 2.0 - z0) / h + 1e-9)) + 1 if z0 < Lz / 2.0 else 0
    z_layers = z0 + h * np.arange(n_layers)
    z_layers = z_layers[(z_layers >= -Lz / 2.0) & (z_layers < Lz / 2.0)]
    n_layers = len(z_layers)
    if n_layers == 0:
        raise ValueError("no layer centre falls inside the box; check Lz, dz and layer_offset")

    rods = []
    for k, z in enumerate(z_layers):
        orient = 'x' if k % 2 == 0 else 'y'
        shift = shift_of(k)
        if segment_ref == 'below':
            k_nb = k - 1 if k > 0 else k + 1
        elif segment_ref == 'above':
            k_nb = k + 1 if k < n_layers - 1 else k - 1
        else:
            raise ValueError("segment_ref must be 'below' or 'above'")
        seg_origin = shift_of(k_nb) if 0 <= k_nb < n_layers else shift_of(k + 1)
        L_perp = Ly if orient == 'x' else Lx
        L_par = Lx if orient == 'x' else Ly
        nmax = int(np.ceil((L_perp / 2.0 + b) / d)) + 1
        for n in range(-nmax, nmax + 1):
            pos = n * d + shift
            if abs(pos) >= L_perp / 2.0 + b:
                continue
            if orient == 'x':
                p1 = (-L_par / 2.0, pos, z); p2 = (L_par / 2.0, pos, z)
            else:
                p1 = (pos, -L_par / 2.0, z); p2 = (pos, L_par / 2.0, z)
            rods.append((*p1, *p2, k, orient, pos, z, b, a, seg_origin))
    return np.array(rods, dtype=ROD_DTYPE), z_layers


def enumerate_segments(rods, box_size, d, tol=1e-9, interior_only=True):
    """
    All complete rod segments (length d) inside the box: arrays rod, j, s0, s1.
    interior_only: skip rods whose axis lies on or outside the box boundary (partial rods).
    """
    Lx, Ly, Lz = _as_triple(box_size, float)
    out = []
    for i, r in enumerate(rods):
        L_par = Lx if r['orientation'] == 'x' else Ly
        L_perp = Ly if r['orientation'] == 'x' else Lx
        if interior_only and abs(r['position']) >= L_perp / 2.0 - tol:
            continue
        jmin = int(np.floor((-L_par / 2.0 - r['seg_origin']) / d)) - 1
        jmax = int(np.ceil((L_par / 2.0 - r['seg_origin']) / d)) + 1
        for j in range(jmin, jmax + 1):
            s0 = r['seg_origin'] + j * d
            s1 = s0 + d
            if s0 >= -L_par / 2.0 - tol and s1 <= L_par / 2.0 + tol:
                out.append((i, j, s0, s1))
    seg = np.array(out, dtype=[('rod', 'i4'), ('j', 'i4'), ('s0', 'f8'), ('s1', 'f8')])
    return seg


def _segments_conflict(cand, acc, rods, d, forbid_crossing=False, tol=1e-6):
    """
    Overlap rule between one candidate segment and the accepted ones (vectorized).
    Two defects "overlap" (Aeby et al.: "two defects cannot overlap") only when they are the SAME
    segment, which plain sampling without replacement already excludes.  Adjacent segments of one
    rod (they merge into one longer defect, cf. the 2-3 segment bulges of Figure 1d) and segments
    on neighbouring parallel rods (Figure 3: "when two defects are adjacent") are allowed.
    forbid_crossing=True additionally rejects a candidate whose segment crosses (touches) an
    accepted segment of an adjacent layer, where the elliptical rods physically overlap (a > h/2).
    """
    if len(acc) == 0 or not forbid_crossing:
        return False
    rc = rods[cand['rod']]
    ra = rods[acc['rod']]
    adj_layer = np.abs(ra['layer'] - rc['layer']) == 1
    crossing = adj_layer & (ra['position'] >= cand['s0'] - tol) & (ra['position'] <= cand['s1'] + tol)                & (rc['position'] >= acc['s0'] - tol) & (rc['position'] <= acc['s1'] + tol)
    return bool(np.any(crossing))


def segment_centres(rods, segments):
    """(N, 3) centres of the segments (rod, j, s0, s1) of enumerate_segments."""
    r = rods[segments['rod']]
    mid = 0.5 * (segments['s0'] + segments['s1'])
    is_x = r['orientation'] == 'x'
    return np.column_stack([np.where(is_x, mid, r['position']), np.where(is_x, r['position'], mid), r['z']])


def stratify_equal_mass(points, n_cells, rng, bounds=None):
    """
    Partition `points` (N, 3) into n_cells compact cells holding (almost) the same number of points:
    k-d tree that recursively cuts the cell along its longest side at the point quantile that gives
    each half a share of points proportional to the number of cells it will hold (ties on the cut plane
    are broken by the other two coordinates).  For an odd cell count the larger half is chosen at random.
    bounds = (lo, hi) of the root cell (default: bounding box of the points); the geometric cell bounds,
    not the point spread, decide the cut axis, so cells stay near-cubic on an anisotropic point lattice.
    Returns a list of n_cells index arrays into `points`; leaf sizes differ by at most a few points.
    """
    points = np.asarray(points, dtype=np.float64)
    N = len(points)
    if not 1 <= n_cells <= N:
        raise ValueError(f"n_cells = {n_cells} must be between 1 and the number of points ({N})")
    if bounds is None:
        bounds = (points.min(axis=0), points.max(axis=0))
    lo0 = np.asarray(bounds[0], dtype=np.float64)
    hi0 = np.asarray(bounds[1], dtype=np.float64)
    leaves = []
    stack = [(np.arange(N), n_cells, lo0, hi0)]
    while stack:
        idx, m, lo, hi = stack.pop()
        if m == 1:
            leaves.append(idx)
            continue
        ax = int(np.argmax(hi - lo))
        o1, o2 = [k for k in range(3) if k != ax]
        P = points[idx]
        order = np.lexsort((P[:, o2], P[:, o1], P[:, ax]))           # primary key: cut axis
        m_lo = m // 2 + (int(rng.integers(2)) if m % 2 else 0)
        n_lo = int(round(len(idx) * m_lo / m))
        n_lo = min(max(n_lo, m_lo), len(idx) - (m - m_lo))           # every leaf keeps >= 1 point
        cut = 0.5 * (P[order[n_lo - 1], ax] + P[order[n_lo], ax])
        hi_lo = hi.copy(); hi_lo[ax] = cut
        lo_hi = lo.copy(); lo_hi[ax] = cut
        stack.append((idx[order[:n_lo]], m_lo, lo, hi_lo))
        stack.append((idx[order[n_lo:]], m - m_lo, lo_hi, hi))
    return leaves


def _place_defects_hyperuniform(rods, segments, n_defects, d, rng, forbid_crossing=False, box_size=None):
    """
    Hyperuniform choice of n_defects distinct segments: the candidate segment centres are split into
    n_defects compact cells of equal candidate count (stratify_equal_mass) and one segment is drawn
    uniformly inside every cell (stratified / "one point per cell" sampling, a uniformly randomized
    lattice on an equal-mass partition).  The number of defects in any window then fluctuates only
    through the cells cut by the window boundary, so the number variance grows like the window
    surface and S(k) ~ k^2 as k -> 0 (class-I hyperuniform), instead of S(k) = 1 for random placement.
    With forbid_crossing=True the cells are visited in random order and the first non-crossing
    candidate of the cell (in random order) is taken; if every candidate of the cell crosses an accepted
    defect (only at high candidate fractions, n_defects/len(segments) >~ 0.15), the unused non-crossing
    candidate closest to the cell centroid is taken instead (small displacement, typically <~ the mean spacing).
    """
    if n_defects > len(segments):
        raise ValueError(f"n_defects = {n_defects} exceeds the {len(segments)} candidate segments; "
                         f"lower n_defects/defect_density")
    pts = segment_centres(rods, segments)
    bounds = None
    if box_size is not None:
        box = np.asarray(_as_triple(box_size, float))
        bounds = (np.minimum(-box / 2.0, pts.min(axis=0)), np.maximum(box / 2.0, pts.max(axis=0)))
    cells = stratify_equal_mass(pts, n_defects, rng, bounds=bounds)
    if not forbid_crossing:
        pick = np.array([cell[rng.integers(len(cell))] for cell in cells], dtype=np.int64)
        return segments[np.sort(pick)]
    accepted = segments[:0]
    used = np.zeros(len(segments), dtype=bool)
    for c in rng.permutation(len(cells)):
        cell = cells[c]
        for idx in cell[rng.permutation(len(cell))]:
            if not used[idx] and not _segments_conflict(segments[idx], accepted, rods, d, forbid_crossing=True):
                break
        else:                              # whole cell blocked (or taken by earlier fallbacks): nearest free candidate to its centroid
            dist2 = np.sum((pts - pts[cell].mean(axis=0)) ** 2, axis=1)
            for idx in np.argsort(dist2, kind='stable'):
                if not used[idx] and not _segments_conflict(segments[idx], accepted, rods, d, forbid_crossing=True):
                    break
            else:
                raise ValueError(f"could only place {len(accepted)} of {n_defects} non-crossing defects "
                                 f"({len(segments)} candidate segments); lower n_defects/defect_density "
                                 f"or set forbid_crossing=False")
        used[idx] = True
        accepted = np.append(accepted, segments[idx:idx + 1])
    return np.sort(accepted, order=['rod', 'j'])


def place_defects(rods, segments, n_defects, d, rng, forbid_crossing=False, distribution='random',
                  box_size=None):
    """
    Choice of n_defects distinct segments (sampling without replacement).
    distribution='random': uniformly random choice (Poisson-like, S(k) = 1 at small k).
    distribution='hyperuniform': one segment per cell of an equal-mass partition of the candidates
    (see _place_defects_hyperuniform); box_size, if given, sets the root cell of the partition.
    With forbid_crossing=True, segments crossing an already accepted defect of an adjacent
    layer are rejected (random: rejection sampling in random order; hyperuniform: per cell, with a
    fallback to the nearest free candidate).
    """
    if n_defects <= 0:
        return segments[:0]
    if distribution == 'hyperuniform':
        return _place_defects_hyperuniform(rods, segments, n_defects, d, rng, forbid_crossing, box_size)
    if distribution != 'random':
        raise ValueError("distribution must be 'random' or 'hyperuniform'")
    order = rng.permutation(len(segments))
    if not forbid_crossing:
        if n_defects > len(segments):
            raise ValueError(f"n_defects = {n_defects} exceeds the {len(segments)} candidate segments; "
                             f"lower n_defects/defect_density")
        return segments[np.sort(order[:n_defects])]
    accepted = segments[:0]
    for idx in order:
        if len(accepted) >= n_defects:
            break
        if not _segments_conflict(segments[idx], accepted, rods, d, forbid_crossing=True):
            accepted = np.append(accepted, segments[idx:idx + 1])
    if len(accepted) < n_defects:
        raise ValueError(f"could only place {len(accepted)} of {n_defects} non-crossing defects "
                         f"({len(segments)} candidate segments); lower n_defects/defect_density "
                         f"or set forbid_crossing=False")
    return accepted


def _rod_pieces(rod, defects_on_rod, box_size):
    """Split one rod into (s_start, s_end, kind) pieces: kind 'rod' or the index into defects_on_rod."""
    Lx, Ly, Lz = _as_triple(box_size, float)
    L_par = Lx if rod['orientation'] == 'x' else Ly
    pieces = []
    cursor = -L_par / 2.0
    order = np.argsort(defects_on_rod['s0']) if len(defects_on_rod) else []
    for i in order:
        s0, s1 = defects_on_rod['s0'][i], defects_on_rod['s1'][i]
        if s0 > cursor:
            pieces.append((cursor, s0, 'rod'))
        pieces.append((s0, s1, int(i)))
        cursor = s1
    if L_par / 2.0 > cursor:
        pieces.append((cursor, L_par / 2.0, 'rod'))
    return pieces


def _endpoints(rod, s0, s1):
    if rod['orientation'] == 'x':
        return (s0, rod['position'], rod['z']), (s1, rod['position'], rod['z'])
    return (rod['position'], s0, rod['z']), (rod['position'], s1, rod['z'])


def voxelize_woodpile(rods, box_size, grid_size, permittivity, background_permittivity,
                      aspect_ratio, defect_segments=None, kappa=0.0, progress_every=None):
    """
    Voxelize the rod list (plus optional defect segments) into a float32 permittivity grid.

    Rods are axis-aligned, so the membership test is separable: defect-free full-length rods are
    stamped through their union cross-section masks (one (Ny, Nz) mask for all x-rods, one (Nx, Nz)
    mask for all y-rods) broadcast along the rod axis; rods carrying defects are split into pieces
    and each piece is stamped with voxelize_rod_axis.  Voxel-for-voxel identical to calling
    voxelize_rod on every piece, but never builds a 3-D meshgrid.
    """
    grid = _as_triple(grid_size, int)
    coords = grid_coordinates(box_size, grid)
    eps = np.full(grid, background_permittivity, dtype=np.float32)
    s = float(aspect_ratio)
    scale = np.sqrt(1.0 + kappa)          # both semi-axes scale so that the AREA scales by (1 + kappa)
    if defect_segments is None:
        defect_segments = np.zeros(0, dtype=[('rod', 'i4'), ('j', 'i4'), ('s0', 'f8'), ('s1', 'f8')])
    with_defects = set(int(r) for r in np.unique(defect_segments['rod']))

    # (1) all defect-free rods at once through the union masks
    if _rods_span_box(rods, box_size):
        Mx, My = _layer_masks(rods, coords, s, skip=with_defects)
        eps[:, Mx] = permittivity
        np.moveaxis(eps, 1, 0)[:, My] = permittivity
        plain = ()
    else:                                  # generic rod list: stamp rod by rod
        plain = (i for i in range(len(rods)) if i not in with_defects)

    # (2) rods carrying defects (and any non-spanning rods) piece by piece
    todo = sorted(with_defects) + list(plain)
    for n, i in enumerate(todo):
        if progress_every and n % progress_every == 0:
            print(f"[voxelize] rod {n} / {len(todo)} (piecewise)")
        rod = rods[i]
        b = float(rod['minor_radius'])
        on_rod = defect_segments[defect_segments['rod'] == i]
        for s0, s1, kind in _rod_pieces(rod, on_rod, box_size):
            bb = b if kind == 'rod' else b * scale
            if bb > 0.0:
                voxelize_rod_axis(eps, coords, rod['orientation'], s0, s1, rod['position'], rod['z'], bb, s, permittivity)
    return eps, coords


def create_woodpile_dist(
    box_size,
    grid_size,
    d,
    dz=None,
    permittivity=1.53 ** 2,
    background_permittivity=1.0,
    minor_radius=None,
    filling_fraction=None,
    aspect_ratio=2.8,
    n_defects=0,
    defect_density=None,
    kappa=0.0,
    seed=None,
    layer_offset=0.0,
    progress_every=None,
    verbose=False,
    segment_ref='below',
    forbid_crossing=False,
    defect_distribution='random',
    ff_tolerance=1e-3,
    ff_max_iter=25,
    save_rods=False,
    add_eps_dist=True,
    dir_save="./Structures",
):
    """
    Voxelized permittivity of a woodpile photonic crystal with controlled rod-segment defects
    (Aeby, Aubry, Muller, Scheffold, Adv. Optical Mater. 9, 2001699 (2021)).

    Geometry conventions
    --------------------
    * Box of size (Lx, Ly, Lz) centred at the origin, voxel centres at (i + 0.5)*dx - L/2.
    * Layers of parallel rods are stacked along z with spacing h = dz/4 (dz = stacking period,
      default sqrt(2)*d -> FCC).  Layer k is centred at z = -Lz/2 + h/2 + layer_offset + k*h;
      even layers run along x, odd layers along y, and every second layer of one orientation
      is shifted in-plane by d/2.  In-plane rod pitch is d.  Only layers whose centre is inside
      the box are generated; rods span the full box along their axis and every rod whose
      cross-section intersects the box is kept.
    * Rod cross-section: ellipse with short semi-axis b = minor_radius in-plane and long
      semi-axis a = aspect_ratio*b along z (DLW voxel shape).  Adjacent layers overlap when a > h/2.
      Give either minor_radius or filling_fraction; in the latter case b is found by bisection on
      the VOXEL filling fraction of the defect-free crystal at the requested resolution.
    * A rod segment is the piece of rod of length d between two consecutive crossings with the
      rods of one neighbouring layer (segment_ref='below': the layer underneath), i.e. the
      boundaries sit at n*d or n*d + d/2 depending on the layer; the other neighbouring layer
      crosses the segment at its midpoint.

    Defects
    -------
    A defect replaces one complete segment by a segment whose cross-section AREA is
    (1 + kappa) times the regular one; the aspect ratio is kept so both semi-axes scale by
    sqrt(1 + kappa).  kappa > 0: thicker segment, kappa < 0: thinner, kappa = -1: segment
    removed.  The defect segment is written INSTEAD of the regular rod piece (the rod is split
    into pieces before voxelization), so for kappa < 0 no trace of the regular rod is left inside
    the segment while the crossing rods of the neighbouring layers stay intact.
    Defects are drawn uniformly at random without replacement from all complete segments in the
    box ("two defects cannot overlap" = no segment is chosen twice).  Adjacent segments of one rod
    may both be defects and then form one longer defect (the paper's Figure 1d shows bulges of
    2-3 segments); the defects table still lists one row per segment.  forbid_crossing=True
    additionally rejects defects on crossing (touching) segments of adjacent layers, where the
    overlapping rods would merge.  n_defects = round(defect_density * Lx*Ly*Lz) when
    defect_density is given.

    defect_distribution selects how the defect segments are spread over the box:
    * 'random' (default): the uniformly random choice above (Poisson-like, S(k) -> 1 as k -> 0).
      Same random stream as before, so a given seed reproduces earlier structures.
    * 'hyperuniform': the candidate segments are split into n_defects compact cells holding the same
      number of candidates (k-d tree cut along the longest cell side, stratify_equal_mass) and one
      segment is drawn uniformly at random inside every cell.  Every region of the box then holds
      the expected number of defects up to its boundary cells: the number variance in a window
      grows like its surface instead of its volume and S(k) ~ k^2 as k -> 0 (class-I hyperuniform),
      while the defects stay disordered on the scale of the mean spacing (V/n_defects)^(1/3).
      Exactly n_defects defects are placed (any n_defects <= number of candidates with forbid_crossing=False).
      Check with structure_factor(np.column_stack([defects['x'], defects['y'], defects['z']]), box_size,
      reference=segment_centres(rods, enumerate_segments(rods, box_size, d))); without `reference` the
      edge deficit of the candidate set (no rods on the box faces) adds a k-independent offset to S.

    add_eps_dist=False drops the voxel grid from the output: eps is returned as None and the
    'epsilon' dataset is left out of the HDF5 file (rods/defects tables and params are still
    written).  The grid is still voxelized internally to measure ff.

    Returns
    -------
    eps      : (Nx, Ny, Nz) float32 permittivity grid (background first, rods overwrite),
               or None when add_eps_dist=False.
    rods     : structured array, one entry per rod: endpoints x1..z2, layer, orientation ('x'/'y'),
               in-plane position, z, minor_radius, major_radius, seg_origin.
    defects  : structured array, one entry per defect: segment centre x,y,z, endpoints, layer,
               orientation, rod_index, segment_index, kappa, minor_radius, major_radius.
    ff       : voxel filling fraction of rod material, mean(eps != background_permittivity).
    info     : dict with the analytic no-overlap estimate ff_analytic = pi*a*b/(d*h), the
               defect-free voxel ff, the bisection residual, layer positions, voxel sizes, ...
               ff_defect_estimate = ff_perfect + n_defects*kappa*pi*a*b*d/V ignores the overlap of
               enlarged segments with the crossing rods, so it overestimates for large kappa > 0.

    Notes
    -----
    * Defect segments in the outermost layers are clipped by the z faces of the box (a_defect can
      exceed the distance to the face), so their effective volume is below (1 + kappa)*V_segment;
      interior layers are exact.
    * The voxel ff at dx = 0.05 um is ~3 % above the continuum value; it converges from above
      (0.3588 -> 0.3500 at dx = 0.025 for the paper parameters).
    """
    box = _as_triple(box_size, float)
    grid = _as_triple(grid_size, int)
    Lx, Ly, Lz = box
    d = float(d)
    dz = float(np.sqrt(2.0) * d) if dz is None else float(dz)
    h = dz / 4.0
    s = float(aspect_ratio)
    rng = np.random.default_rng(seed)

    if (minor_radius is None) == (filling_fraction is None):
        raise ValueError("give exactly one of minor_radius or filling_fraction")
    if defect_distribution not in ('random', 'hyperuniform'):
        raise ValueError("defect_distribution must be 'random' or 'hyperuniform'")

    def perfect_ff(b):
        # exact voxel ff of the defect-free crystal from the 2-D cross-section masks (no 3-D grid)
        rods_b, _ = build_woodpile_rods(box, d, dz, b, s, layer_offset, segment_ref)
        return perfect_voxel_ff(rods_b, box, grid, s)

    ff_residual = None
    if filling_fraction is not None:
        target = float(filling_fraction)
        lo, hi = 0.0, d / 2.0
        f_hi = perfect_ff(hi)
        if f_hi < target:
            raise ValueError(f"target filling fraction {target} not reachable with b <= d/2 (ff={f_hi:.4f})")
        best = (hi, f_hi)
        for it in range(ff_max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = perfect_ff(mid)
            if verbose:
                print(f"[ff bisection] it {it:2d}: b = {mid:.5f}  ff = {f_mid:.5f}  (target {target})")
            if abs(f_mid - target) < abs(best[1] - target):
                best = (mid, f_mid)
            if abs(f_mid - target) < ff_tolerance or (hi - lo) < 1e-5 * d:
                break                     # voxel ff is a step function of b: stop once the bracket collapses
            if f_mid < target:
                lo = mid
            else:
                hi = mid
        b, f_best = best
        ff_residual = f_best - target
        if verbose:
            print(f"[ff bisection] best b = {b:.5f}, ff = {f_best:.5f}, residual = {ff_residual:+.5f}")
    else:
        b = float(minor_radius)
    a = s * b

    rods, z_layers = build_woodpile_rods(box, d, dz, b, s, layer_offset, segment_ref)
    n_outside = int(np.sum(np.abs(rods['position']) > np.where(rods['orientation'] == 'x', Ly, Lx) / 2.0))
    if n_outside and verbose:
        print(f"[warn] {n_outside}/{len(rods)} rod axes fall outside the box (partial rods at the boundary).")

    if defect_density is not None:
        n_defects = int(round(float(defect_density) * Lx * Ly * Lz))
    n_defects = int(n_defects)
    # actual (post-rounding) defect density; 0.0 for a defect-free woodpile
    defect_density = n_defects / (Lx * Ly * Lz)
    segments = enumerate_segments(rods, box, d)
    chosen = (place_defects(rods, segments, n_defects, d, rng, forbid_crossing,
                            distribution=defect_distribution, box_size=box)
              if n_defects > 0 else segments[:0])

    if verbose:
        print(f"[woodpile] {len(z_layers)} layers (h = {h:.4f}), {len(rods)} rods, "
              f"{len(segments)} complete segments, {len(chosen)} defects (kappa = {kappa}, "
              f"{defect_distribution})")

    eps, coords = voxelize_woodpile(rods, box, grid, permittivity, background_permittivity, s,
                                    defect_segments=chosen, kappa=kappa, progress_every=progress_every)

    scale = float(np.sqrt(1.0 + kappa))
    defects = np.zeros(len(chosen), dtype=DEFECT_DTYPE)
    for m, seg in enumerate(chosen):
        rod = rods[seg['rod']]
        p1, p2 = _endpoints(rod, seg['s0'], seg['s1'])
        defects[m] = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2]),
                      *p1, *p2, rod['layer'], rod['orientation'], seg['rod'], seg['j'],
                      kappa, b * scale, a * scale)

    ff = float(np.mean(eps != np.float32(background_permittivity)))
    ff_analytic = np.pi * a * b / (d * h)
    ff_perfect = perfect_ff(b) if n_defects > 0 else ff
    seg_volume = np.pi * a * b * d
    info = dict(
        minor_radius=b, major_radius=a, aspect_ratio=s, d=d, dz=dz, layer_spacing=h,
        n_layers=len(z_layers), z_layers=z_layers, n_rods=len(rods), n_segments=len(segments),
        n_defects=len(chosen), kappa=kappa, forbid_crossing=forbid_crossing,
        defect_distribution=defect_distribution,
        ff=ff, ff_perfect=ff_perfect, ff_analytic=ff_analytic,
        ff_defect_estimate=ff_perfect + len(chosen) * kappa * seg_volume / (Lx * Ly * Lz),
        ff_target=filling_fraction, ff_residual=ff_residual,
        voxel_size=tuple(L / N for L, N in zip(box, grid)), coords=coords,
        rods_outside_box=n_outside,
    )

    if save_rods:
        dir = dir_save
        os.makedirs(dir, exist_ok=True)
        seed_str = "none" if seed is None else str(seed)
        tag = f"woodpile_d{d:.2f}_kappa{info['kappa']:+.2f}_rho{defect_density:.3f}_seed{seed_str}"
        if defect_distribution != 'random' and len(chosen) > 0:
            tag += f"_{defect_distribution}"     # random files keep their old names
        # AM.create_hdf5_from_dict({"epsilon": eps}, rf"{dir}/n_{np.sqrt(permittivity):.2f}_ff_{ff:.4f}.h5")
        AM.create_hdf5_from_dict(
            {**({"epsilon": eps} if add_eps_dist else {}), **tables_to_dict(rods, defects),
             "params": {"box_size": np.array(box_size), "grid_size": np.array(grid_size), "d": d, "dz": dz,
                        "minor_radius": info['minor_radius'], "major_radius": info['major_radius'],
                        "aspect_ratio": aspect_ratio, "permittivity": permittivity, "background_permittivity": background_permittivity,
                        "kappa": info['kappa'], "defect_density": defect_density, "seed": -1 if seed is None else int(seed), "ff": ff,
                        "defect_distribution": defect_distribution,
                        "ff_analytic": info['ff_analytic']}},
            rf"{dir}/n_{np.sqrt(permittivity):.2f}_ff_{ff:.4f}_{tag}_tables.h5")
    if verbose:
        print(f"[woodpile] b = {b:.4f}, a = {a:.4f}  ->  ff(voxel) = {ff:.4f}, "
              f"ff(defect-free) = {ff_perfect:.4f}, ff(analytic, no overlap) = {ff_analytic:.4f}")
    if not add_eps_dist:
        eps = None
    return eps, rods, defects, ff, info


def structure_factor(points, box_size, k_max=None, n_bins=40, reference=None, chunk=20000):
    """
    Angularly averaged structure factor S(k) = |sum_j exp(-i k.r_j)|^2 / N of a point pattern in the
    box (centred at the origin), evaluated EXACTLY (direct sum, no binning) on the reciprocal grid of
    the box k = 2 pi (mx/Lx, my/Ly, mz/Lz), k != 0.  On that grid the transform of a uniform box window
    vanishes, so there is no forward-scattering peak: S -> 1 for uncorrelated (random) points and
    S -> 0 as k -> 0 for hyperuniform ones.  Default k_max = 2 pi / a_mean (a_mean = (V/N)^(1/3), the
    mean spacing); the cost is ~ N * (number of k vectors) and grows like k_max^3.
    reference: optional (M, 3) points the pattern was drawn from (e.g. all candidate segment centres,
    segment_centres(rods, enumerate_segments(rods, box, d))).  Their mean-density transform, scaled by
    N/M, is subtracted from rho(k) before squaring, which removes the deterministic edge/window term of
    a candidate set that does not fill the box uniformly (no complete segments at the in-plane faces).
    Then S -> 1 - N/M for a random choice without replacement and S -> 0 as k -> 0 if hyperuniform.
    The sum factorises per axis, so it is done as one complex matrix product per kz plane, over
    chunks of `chunk` points (memory ~ chunk * (nx + ny + nz) * 16 bytes).
    Returns k (bin centres), S (bin means, nan for empty bins), counts (k vectors per bin).
    Use e.g. structure_factor(np.column_stack([defects['x'], defects['y'], defects['z']]), box).
    """
    pts = np.asarray(points, dtype=np.float64)
    N = len(pts)
    box = np.asarray(_as_triple(box_size, float))
    if k_max is None:
        k_max = 2.0 * np.pi / (np.prod(box) / N) ** (1.0 / 3.0)
    dk = 2.0 * np.pi / box
    mx = np.arange(-int(k_max / dk[0]), int(k_max / dk[0]) + 1)
    my = np.arange(-int(k_max / dk[1]), int(k_max / dk[1]) + 1)
    mz = np.arange(0, int(k_max / dk[2]) + 1)                  # S(k) = S(-k): half space kz >= 0
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        pts_all = np.concatenate([pts, ref])
        wt = np.concatenate([np.ones(N), np.full(len(ref), -N / len(ref))])   # rho(k) - (N/M) rho_ref(k)
    else:
        pts_all, wt = pts, np.ones(N)
    kx, ky, kz = mx * dk[0], my * dk[1], mz * dk[2]
    rho = np.zeros((len(mz), len(mx), len(my)), dtype=np.complex128)   # sum_j w_j e^{-i k.r_j}
    for j0 in range(0, len(pts_all), chunk):                           # chunks bound the memory
        p = pts_all[j0:j0 + chunk]
        ex = np.exp(-1j * np.outer(p[:, 0], kx)) * wt[j0:j0 + chunk, None]
        ey = np.exp(-1j * np.outer(p[:, 1], ky))
        ez = np.exp(-1j * np.outer(p[:, 2], kz))
        for m in range(len(mz)):
            rho[m] += (ex * ez[:, m:m + 1]).T @ ey
    k_edges = np.linspace(0.0, k_max, n_bins + 1)
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    KXY2 = kx[:, None] ** 2 + ky[None, :] ** 2
    for m in range(len(mz)):
        K = np.sqrt(KXY2 + kz[m] ** 2)
        sel = (K > 0) & (K <= k_max)
        if not np.any(sel):
            continue
        S2 = (rho[m].real ** 2 + rho[m].imag ** 2) / N
        w = 2.0 if mz[m] > 0 else 1.0                               # kz > 0 plane stands for +-kz
        ib = np.clip(np.digitize(K[sel], k_edges) - 1, 0, n_bins - 1)
        counts += w * np.bincount(ib, minlength=n_bins)
        sums += w * np.bincount(ib, weights=S2[sel], minlength=n_bins)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = np.where(counts > 0, sums / counts, np.nan)
    return 0.5 * (k_edges[1:] + k_edges[:-1]), S, counts


def tables_to_dict(rods, defects):
    """Plain-array dict of the rods/defects tables for AM.create_hdf5_from_dict (orientation: 0 = x, 1 = y)."""
    def conv(tab):
        out = {}
        for name in tab.dtype.names:
            col = tab[name]
            out[name] = (col == 'y').astype(np.int8) if col.dtype.kind == 'U' else np.asarray(col)
        return out
    return {'rods': conv(rods), 'defects': conv(defects)}


def show_slice(eps, box_size, axis, value, ax=None, title=None, cmap='gray_r', defects=None):
    """
    Plot the eps slice closest to `value` along `axis` (0 = x, 1 = y, 2 = z) with physical extent.
    If `defects` (structured array) is given, defect segments lying in or next to the slice are
    outlined in red (along-axis extent = segment, transverse extent = +- semi-axis).
    """
    if ax is None:
        _, ax = plt.subplots()
    box = _as_triple(box_size, float)
    coords = grid_coordinates(box, eps.shape)
    i = int(np.argmin(np.abs(coords[axis] - value)))
    other = [k for k in range(3) if k != axis]
    img = np.take(eps, i, axis=axis).T
    ext = [-box[other[0]] / 2, box[other[0]] / 2, -box[other[1]] / 2, box[other[1]] / 2]
    ax.imshow(img, origin='lower', extent=ext, cmap=cmap, aspect='equal', interpolation='nearest')
    names = 'xyz'
    ax.set_xlabel(f"{names[other[0]]} [um]"); ax.set_ylabel(f"{names[other[1]]} [um]")
    ax.set_title(title or f"{names[axis]} = {coords[axis][i]:.3f} um")
    if defects is not None and len(defects):
        seg_len = np.max(np.abs(defects['x2'] - defects['x1']) + np.abs(defects['y2'] - defects['y1']))
        near = np.abs(defects[names[axis]] - coords[axis][i]) <= max(defects['major_radius'].max(), seg_len / 2)
        for df in defects[near]:
            lo = [min(df[f"{names[k]}1"], df[f"{names[k]}2"]) for k in other]
            hi = [max(df[f"{names[k]}1"], df[f"{names[k]}2"]) for k in other]
            w = [hi[q] - lo[q] for q in range(2)]
            for q, k in enumerate(other):
                if w[q] == 0:      # transverse extent: +- semi-axis
                    half = df['major_radius'] if k == 2 else df['minor_radius']
                    lo[q] -= half; w[q] = 2 * half
            ax.add_patch(mpatches.Rectangle((lo[0], lo[1]), w[0], w[1], fill=False, ec='red', lw=1.0))
    return ax
