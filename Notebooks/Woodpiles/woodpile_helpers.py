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

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# AutomationModule lives in the root of the tidy3d project
sys.path.append(os.path.abspath(r'H:\codes\tidy3d'))
import AutomationModule as AM

__all__ = ['create_woodpile_dist', 'build_woodpile_rods', 'enumerate_segments', 'place_defects',
           'check_periodic_box', 'primary_rods', 'periodic_stamp_segments', 'write_woodpile_ctl',
           'write_woodpile_primitive_ctl', 'woodpile_kpoint_cartesian', 'CTL_DEFAULTS',
           'WOODPILE_KPOINTS_FCC', 'WOODPILE_KPATH', 'read_mpb_freqs',
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


def check_periodic_box(box_size, d, dz, n_layers=None, tol=1e-6):
    """
    Cell counts (Nx, Ny, Nz) of a box that is a periodic supercell of the woodpile: Lx = Nx d,
    Ly = Ny d, Lz = Nz dz (and 4 Nz layers, if n_layers is given).  Raises ValueError otherwise.
    """
    Lx, Ly, Lz = _as_triple(box_size, float)
    counts = [L / p for L, p in ((Lx, d), (Ly, d), (Lz, dz))]
    N = tuple(int(round(c)) for c in counts)
    if any(n < 1 or abs(c - n) > tol * max(1.0, c) for c, n in zip(counts, N)):
        raise ValueError(f"a periodic woodpile box needs Lx = Nx d, Ly = Ny d, Lz = Nz dz; got "
                         f"Lx/d = {counts[0]:.6f}, Ly/d = {counts[1]:.6f}, Lz/dz = {counts[2]:.6f}")
    if n_layers is not None and n_layers != 4 * N[2]:
        raise ValueError(f"a periodic box with Lz = {N[2]} dz needs {4 * N[2]} layers, got {n_layers} "
                         f"(keep 0 <= layer_offset < h/2)")
    return N


def primary_rods(rods, box_size, tol=1e-9):
    """
    Boolean mask of the rods that are NOT periodic images: build_woodpile_rods keeps a rod at both
    in-plane faces (pos = -L/2 and +L/2) when the box is a multiple of d; in a periodic supercell the
    one at +L/2 is the image of the one at -L/2.
    """
    Lx, Ly, Lz = _as_triple(box_size, float)
    L_perp = np.where(rods['orientation'] == 'x', Ly, Lx)
    return rods['position'] < L_perp / 2.0 - tol


def _wrap(x, L):
    """Minimal-image coordinate of x in a period L (result in [-L/2, L/2))."""
    return x - L * np.floor(x / L + 0.5)


def enumerate_segments(rods, box_size, d, tol=1e-9, interior_only=True, periodic=False):
    """
    All complete rod segments (length d) inside the box: arrays rod, j, s0, s1.
    interior_only: skip rods whose axis lies on or outside the box boundary (partial rods).
    periodic: the box is a periodic supercell (check_periodic_box).  Every primary rod (primary_rods,
    boundary rods included) contributes L_par / d segments that start inside the box; a segment that
    starts within d of the +L/2 face wraps around (s1 > L/2, its tail continues at -L/2).
    """
    Lx, Ly, Lz = _as_triple(box_size, float)
    keep = primary_rods(rods, box_size) if periodic else np.ones(len(rods), dtype=bool)
    out = []
    for i, r in enumerate(rods):
        L_par = Lx if r['orientation'] == 'x' else Ly
        L_perp = Ly if r['orientation'] == 'x' else Lx
        if not keep[i] or (not periodic and interior_only and abs(r['position']) >= L_perp / 2.0 - tol):
            continue
        jmin = int(np.floor((-L_par / 2.0 - r['seg_origin']) / d)) - 1
        jmax = int(np.ceil((L_par / 2.0 - r['seg_origin']) / d)) + 1
        for j in range(jmin, jmax + 1):
            s0 = r['seg_origin'] + j * d
            s1 = s0 + d
            inside = (s0 >= -L_par / 2.0 - tol and s0 < L_par / 2.0 - tol) if periodic else \
                     (s0 >= -L_par / 2.0 - tol and s1 <= L_par / 2.0 + tol)
            if inside:
                out.append((i, j, s0, s1))
    seg = np.array(out, dtype=[('rod', 'i4'), ('j', 'i4'), ('s0', 'f8'), ('s1', 'f8')])
    return seg


def periodic_stamp_segments(rods, segments, box_size, tol=1e-9):
    """
    Pieces to stamp for segments of a periodic supercell: a wrapping segment (s1 > L/2) is split into
    [s0, L/2] and [-L/2, s1 - L], and every piece is repeated on the periodic image of its rod (the
    copy at +L/2 of a boundary rod), so the voxel grid is periodic in-plane.  Same dtype as the input.
    """
    Lx, Ly, Lz = _as_triple(box_size, float)
    out = []
    for seg in segments:
        r = rods[seg['rod']]
        L_par = Lx if r['orientation'] == 'x' else Ly
        L_perp = Ly if r['orientation'] == 'x' else Lx
        pieces = [(seg['s0'], min(seg['s1'], L_par / 2.0))]
        if seg['s1'] > L_par / 2.0 + tol:
            pieces.append((-L_par / 2.0, seg['s1'] - L_par))
        images = np.flatnonzero((rods['layer'] == r['layer'])
                                & (np.abs(rods['position'] - (r['position'] + L_perp)) < 1e-6))
        for i in (int(seg['rod']), *(int(k) for k in images)):
            for s0, s1 in pieces:
                out.append((i, seg['j'], s0, s1))
    return np.array(out, dtype=segments.dtype)


def _with_z_images(rods, segments, box_size, reach):
    """
    Rods (and their defect segments) of a periodic supercell plus their images shifted by +-Lz whose
    cross-section (z extent `reach`) still reaches into the box, so the voxel grid is periodic along z
    when the rods overlap the z faces (a > h/2).  The images are appended after the original rods.
    """
    Lz = _as_triple(box_size, float)[2]
    imgs, src = [], []
    for shift in (-Lz, Lz):
        z = rods['z'] + shift
        for i in np.flatnonzero(np.abs(z) - reach < Lz / 2.0):
            r = rods[i].copy()
            r['z'] += shift; r['z1'] += shift; r['z2'] += shift
            imgs.append(r); src.append(i)
    if not imgs:
        return rods, segments
    all_rods = np.concatenate([rods, np.array(imgs, dtype=rods.dtype)])
    extra = []
    for k, i in enumerate(src):
        for seg in segments[segments['rod'] == i]:
            extra.append((len(rods) + k, seg['j'], seg['s0'], seg['s1']))
    all_segs = np.concatenate([segments, np.array(extra, dtype=segments.dtype)]) if extra else segments
    return all_rods, all_segs


def _segments_conflict(cand, acc, rods, d, forbid_crossing=False, tol=1e-6, periodic_box=None):
    """
    Overlap rule between one candidate segment and the accepted ones (vectorized).
    Two defects "overlap" (Aeby et al.: "two defects cannot overlap") only when they are the SAME
    segment, which plain sampling without replacement already excludes.  Adjacent segments of one
    rod (they merge into one longer defect, cf. the 2-3 segment bulges of Figure 1d) and segments
    on neighbouring parallel rods (Figure 3: "when two defects are adjacent") are allowed.
    forbid_crossing=True additionally rejects a candidate whose segment crosses (touches) an
    accepted segment of an adjacent layer, where the elliptical rods physically overlap (a > h/2).
    periodic_box: box of a periodic supercell; distances along the rod axes are then minimal-image
    and the top and bottom layers are adjacent.
    """
    if len(acc) == 0 or not forbid_crossing:
        return False
    rc = rods[cand['rod']]
    ra = rods[acc['rod']]
    if periodic_box is None:
        adj_layer = np.abs(ra['layer'] - rc['layer']) == 1
        crossing = adj_layer & (ra['position'] >= cand['s0'] - tol) & (ra['position'] <= cand['s1'] + tol) \
                   & (rc['position'] >= acc['s0'] - tol) & (rc['position'] <= acc['s1'] + tol)
        return bool(np.any(crossing))
    Lx, Ly, Lz = _as_triple(periodic_box, float)
    n_layers = int(rods['layer'].max()) + 1
    dl = np.abs(ra['layer'] - rc['layer'])
    adj_layer = np.minimum(dl, n_layers - dl) == 1
    Lc = Lx if rc['orientation'] == 'x' else Ly                  # period along the candidate axis
    La = np.where(ra['orientation'] == 'x', Lx, Ly)               # periods along the accepted axes
    crossing = adj_layer \
        & (np.abs(_wrap(ra['position'] - 0.5 * (cand['s0'] + cand['s1']), Lc)) <= 0.5 * (cand['s1'] - cand['s0']) + tol) \
        & (np.abs(_wrap(rc['position'] - 0.5 * (acc['s0'] + acc['s1']), La)) <= 0.5 * (acc['s1'] - acc['s0']) + tol)
    return bool(np.any(crossing))


def segment_centres(rods, segments, box_size=None):
    """
    (N, 3) centres of the segments (rod, j, s0, s1) of enumerate_segments.  With box_size the centre
    of a wrapping segment of a periodic supercell is mapped back into the box (no-op otherwise).
    """
    r = rods[segments['rod']]
    mid = 0.5 * (segments['s0'] + segments['s1'])
    is_x = r['orientation'] == 'x'
    if box_size is not None:
        Lx, Ly, Lz = _as_triple(box_size, float)
        mid = _wrap(mid, np.where(is_x, Lx, Ly))
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


def _place_defects_hyperuniform(rods, segments, n_defects, d, rng, forbid_crossing=False, box_size=None,
                                periodic=False):
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
    periodic=True (periodic supercell, needs box_size): wrapping segments are binned by their wrapped centre
    and the crossing test is minimal-image.
    """
    if n_defects > len(segments):
        raise ValueError(f"n_defects = {n_defects} exceeds the {len(segments)} candidate segments; "
                         f"lower n_defects/defect_density")
    pts = segment_centres(rods, segments, box_size if periodic else None)
    pbox = box_size if periodic else None
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
            if not used[idx] and not _segments_conflict(segments[idx], accepted, rods, d, forbid_crossing=True,
                                                        periodic_box=pbox):
                break
        else:                              # whole cell blocked (or taken by earlier fallbacks): nearest free candidate to its centroid
            dist2 = np.sum((pts - pts[cell].mean(axis=0)) ** 2, axis=1)
            for idx in np.argsort(dist2, kind='stable'):
                if not used[idx] and not _segments_conflict(segments[idx], accepted, rods, d, forbid_crossing=True,
                                                            periodic_box=pbox):
                    break
            else:
                raise ValueError(f"could only place {len(accepted)} of {n_defects} non-crossing defects "
                                 f"({len(segments)} candidate segments); lower n_defects/defect_density "
                                 f"or set forbid_crossing=False")
        used[idx] = True
        accepted = np.append(accepted, segments[idx:idx + 1])
    return np.sort(accepted, order=['rod', 'j'])


def place_defects(rods, segments, n_defects, d, rng, forbid_crossing=False, distribution='random',
                  box_size=None, periodic=False):
    """
    Choice of n_defects distinct segments (sampling without replacement).
    distribution='random': uniformly random choice (Poisson-like, S(k) = 1 at small k).
    distribution='hyperuniform': one segment per cell of an equal-mass partition of the candidates
    (see _place_defects_hyperuniform); box_size, if given, sets the root cell of the partition.
    With forbid_crossing=True, segments crossing an already accepted defect of an adjacent
    layer are rejected (random: rejection sampling in random order; hyperuniform: per cell, with a
    fallback to the nearest free candidate).
    periodic=True: segments of a periodic supercell (enumerate_segments(..., periodic=True), needs
    box_size); the crossing test then uses minimal-image distances.
    """
    if n_defects <= 0:
        return segments[:0]
    if distribution == 'hyperuniform':
        return _place_defects_hyperuniform(rods, segments, n_defects, d, rng, forbid_crossing, box_size, periodic)
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
        if not _segments_conflict(segments[idx], accepted, rods, d, forbid_crossing=True,
                                  periodic_box=box_size if periodic else None):
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
    _stamp_woodpile(eps, coords, rods, box_size, permittivity, aspect_ratio,
                    defect_segments=defect_segments, kappa=kappa, progress_every=progress_every)
    return eps, coords


def _stamp_woodpile(eps, coords, rods, box_size, value, aspect_ratio, defect_segments=None, kappa=0.0,
                    progress_every=None):
    """
    Write `value` into every voxel of `eps` (shape = lengths of `coords`) inside a rod or defect
    piece.  coords may be any sub-range of the full grid coordinates (e.g. a slab of z planes): the
    membership test is pointwise on voxel centres, so the result is identical on the overlap.
    """
    s = float(aspect_ratio)
    scale = np.sqrt(1.0 + kappa)          # both semi-axes scale so that the AREA scales by (1 + kappa)
    if defect_segments is None:
        defect_segments = np.zeros(0, dtype=[('rod', 'i4'), ('j', 'i4'), ('s0', 'f8'), ('s1', 'f8')])
    with_defects = set(int(r) for r in np.unique(defect_segments['rod']))

    # (1) all defect-free rods at once through the union masks
    if _rods_span_box(rods, box_size):
        Mx, My = _layer_masks(rods, coords, s, skip=with_defects)
        eps[:, Mx] = value
        np.moveaxis(eps, 1, 0)[:, My] = value
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
                voxelize_rod_axis(eps, coords, rod['orientation'], s0, s1, rod['position'], rod['z'], bb, s, value)


def woodpile_voxel_ff(rods, box_size, grid_size, aspect_ratio, defect_segments=None, kappa=0.0,
                      max_chunk_voxels=2 ** 26):
    """
    Voxel filling fraction mean(eps != background) of voxelize_woodpile WITHOUT building the
    float grid.  Defect-free spanning rods: perfect_voxel_ff (2-D masks only).  Otherwise the
    occupied voxels are counted slab by slab along z in a boolean array of <= max_chunk_voxels
    (same membership tests, so the count is voxel-for-voxel identical; memory ~ Nx*Ny*nz_chunk bytes).
    """
    grid = _as_triple(grid_size, int)
    Nx, Ny, Nz = grid
    no_defects = defect_segments is None or len(defect_segments) == 0
    if no_defects and _rods_span_box(rods, box_size):
        return perfect_voxel_ff(rods, box_size, grid, aspect_ratio)
    coords = grid_coordinates(box_size, grid)
    nz_chunk = max(1, int(max_chunk_voxels) // (Nx * Ny))
    filled = 0
    for k0 in range(0, Nz, nz_chunk):
        sub = [coords[0], coords[1], coords[2][k0:k0 + nz_chunk]]
        occ = np.zeros((Nx, Ny, len(sub[2])), dtype=bool)
        _stamp_woodpile(occ, sub, rods, box_size, True, aspect_ratio, defect_segments=defect_segments, kappa=kappa)
        filled += int(np.count_nonzero(occ))
    return filled / float(Nx * Ny * Nz)


# ---------------------------------------------------------------------------------------------------
# MPB (.ctl) export
# ---------------------------------------------------------------------------------------------------
# Defaults of the supercell ctl (write_woodpile_ctl); see the MPB section of create_woodpile_dist.ipynb
# for the convergence tests behind them.  Lengths in the ctl are in units of the rod pitch d, so MPB
# frequencies are nu = d / lambda (the a/lambda of the FDTD notebooks, a = d).
CTL_DEFAULTS = dict(
    resolution=16,          # grid points per d (MPB `resolution`); 'voxels' geometry: the voxel grid
    mesh_size=3,            # sub-pixel averaging mesh (MPB `mesh-size`)
    num_bands=None,         # None: ceil(band_margin * gap_band) with gap_band = 4 Nx Ny Nz
    band_margin=1.25,       # bands computed above the gap band (fraction of gap_band)
    tolerance=1e-5,         # eigensolver tolerance
    block_size=None,        # MPB eigensolver-block-size (None: MPB default -11)
    k_points='gamma',       # 'gamma' or a list of (k1, k2, k3) in the supercell reciprocal basis
    geometry='auto',        # 'objects' | 'voxels' | 'auto' (objects unless elliptical rods carry defects)
    output_epsilon=True,    # write <name>-epsilon.h5 (check the geometry with h5topng / h5py)
)
_ELLIPSOID_LENGTH = 1e4     # length (units of d) of the ellipsoid standing in for an infinite elliptical rod


def _scm(x):
    """Scheme number literal."""
    return 'infinity' if np.isinf(x) else f"{float(x):.10g}"


def _rod_object(axis, pos, z0, s0, s1, b, a, c2l=False):
    """
    MPB geometric object of one rod piece from s0 to s1 along `axis` ('x' or 'y'; s0, s1 = -inf, inf for an
    infinite rod).  Circular rods (a == b) are cylinders.  An infinite elliptical rod (in-plane semi-axis b,
    a along z) is an ellipsoid of length _ELLIPSOID_LENGTH, whose cross-section inside the cell differs from
    the ellipse by < (L / 1e4)^2 / 2; MPB has no finite elliptical cylinder (libctl prisms work but take
    minutes per object to initialize), so finite elliptical pieces raise.  c2l wraps every vector in
    (c->l ...) for a non-orthogonal lattice; radius, height and size are cartesian lengths.
    """
    def vec(v):
        txt = ' '.join(_scm(c) for c in v)
        return f"(c->l {txt})" if c2l else f"(vector3 {txt})"
    ax = (1, 0, 0) if axis == 'x' else (0, 1, 0)
    infinite = np.isinf(s0) or np.isinf(s1)
    centre, height = (0.0, np.inf) if infinite else (0.5 * (s0 + s1), s1 - s0)
    c = (centre, pos, z0) if axis == 'x' else (pos, centre, z0)
    if np.isclose(a, b):
        return (f"(make cylinder (material diel) (center {vec(c)}) (axis {vec(ax)}) "
                f"(radius {_scm(b)}) (height {_scm(height)}))")
    if not infinite:
        raise ValueError("finite elliptical rod pieces cannot be MPB objects; use geometry='voxels'")
    perp = (0, 1, 0) if axis == 'x' else (1, 0, 0)
    return (f"(make ellipsoid (material diel) (center {vec(c)}) (e1 {vec(ax)}) (e2 {vec(perp)}) "
            f"(e3 {vec((0, 0, 1))}) (size {_scm(_ELLIPSOID_LENGTH)} {_scm(2 * b)} {_scm(2 * a)}))")


def _ctl_header(lines, title, permittivity, background_permittivity, d):
    lines += [f"; {title}",
              "; written by woodpile_helpers.py (create_woodpile_dist, generate_ctl=True)",
              f"; lengths in units of the rod pitch d = {d:g} um  ->  MPB frequencies are nu = d/lambda",
              "",
              f"(set! default-material (make dielectric (epsilon {_scm(background_permittivity)})))",
              f"(define diel (make dielectric (epsilon {_scm(permittivity)})))",
              ""]


def _ctl_run(lines, opt):
    res = opt['resolution']
    res = f"(vector3 {' '.join(_scm(r) for r in res)})" if np.ndim(res) else str(int(res))
    lines += [f"(set-param! resolution {res})",
              f"(set-param! mesh-size {int(opt['mesh_size'])})",
              f"(set-param! num-bands {int(opt['num_bands'])})",
              f"(set! tolerance {opt['tolerance']:g})"]
    if opt.get('block_size') is not None:
        lines.append(f"(set! eigensolver-block-size {int(opt['block_size'])})")
    if not opt.get('output_epsilon', True):
        lines.append('(set! output-epsilon (lambda () (print "skipping output-epsilon\\n")))')
    lines += ["", "(run)", ""]


def write_woodpile_ctl(path, rods, defect_segments, box_size, d, dz, aspect_ratio, permittivity,
                       background_permittivity, kappa=0.0, title=None, eps=None, **options):
    """
    MPB control file of the box as ONE periodic supercell (Lx = Nx d, Ly = Ny d, Lz = Nz dz, see
    check_periodic_box).  Lengths are in units of d (frequencies nu = d/lambda).  The complete gap of the
    perfect woodpile lies above band gap_band = 4 Nx Ny Nz (2 rods per primitive cell, 2 primitive cells per
    d x d x dz cell); num_bands defaults to ceil(band_margin * gap_band).  Options: CTL_DEFAULTS.

    geometry='objects': exact MPB geometric objects with sub-pixel averaging.  rods: rod table of
    build_woodpile_rods / create_woodpile_dist; only primary_rods are written (the copy at +L/2 of a boundary
    rod is MPB's periodic image).  defect_segments: (rod, j, s0, s1) segments carrying the defect (cross-section
    AREA x (1 + kappa)), e.g. the `chosen` segments of create_woodpile_dist; wrapping segments (s1 > L/2) are
    split with periodic_stamp_segments.  Rods without defects are single infinite objects, rods with defects
    are written piece by piece (_rod_pieces).  MPB's ensure-periodicity (default true) adds the images of
    pieces and rods that cross the cell faces, so overlaps across the z faces (a > h/2) are periodic too.
    Circular rods are cylinders; elliptical rods (aspect_ratio != 1) only work without defects (ellipsoids).
    geometry='voxels': the periodic voxel grid `eps` (create_woodpile_dist with periodic=True) is written to
    <path stem>_eps.h5 (dataset 'epsilon') and read as MPB's epsilon-input-file on the same grid; needed for
    elliptical rods with defects.  geometry='auto': 'voxels' only in that case.
    Returns (path, options used).
    """
    opt = {**CTL_DEFAULTS, **options}
    box = _as_triple(box_size, float)
    Lx, Ly, Lz = box
    n_layers = int(rods['layer'].max()) + 1
    Nx, Ny, Nz = check_periodic_box(box, d, dz, n_layers)
    gap_band = 4 * Nx * Ny * Nz
    if opt['num_bands'] is None:
        opt['num_bands'] = max(int(np.ceil(opt['band_margin'] * gap_band)), gap_band + 8)
    s = float(aspect_ratio)
    n_def = 0 if defect_segments is None else len(defect_segments)
    if opt['geometry'] == 'auto':
        opt['geometry'] = 'voxels' if (s != 1.0 and n_def > 0) else 'objects'
    if opt['geometry'] not in ('objects', 'voxels'):
        raise ValueError("geometry must be 'objects', 'voxels' or 'auto'")
    keep = primary_rods(rods, box)

    lines = []
    _ctl_header(lines, title or f"woodpile supercell {Nx} x {Ny} x {Nz} (d x d x dz cells)",
                permittivity, background_permittivity, d)
    lines += [f"; cells {Nx} x {Ny} x {Nz}, {n_layers} layers, {int(keep.sum())} rods, "
              f"{n_def} defect segments (kappa = {kappa:+g}), geometry: {opt['geometry']}",
              f"; complete gap of the perfect crystal above band {gap_band}",
              f"(set! geometry-lattice (make lattice (size {_scm(Lx / d)} {_scm(Ly / d)} {_scm(Lz / d)})))", ""]
    if opt['geometry'] == 'voxels':
        if eps is None:
            raise ValueError("geometry='voxels' needs the voxel grid eps")
        eps_file = os.path.splitext(path)[0] + "_eps.h5"
        os.makedirs(os.path.dirname(os.path.abspath(eps_file)), exist_ok=True)
        with h5py.File(eps_file, 'w') as fh:
            fh.create_dataset('epsilon', data=np.asarray(eps, dtype=np.float64))
        # MPB grid = voxel grid (MPB rounds resolution * size UP, so stay a hair below n / size)
        opt['resolution'] = tuple(n / (L / d) * (1.0 - 1e-9) for n, L in zip(eps.shape, box))
        opt['eps_file'] = eps_file
        lines += [f'(set! epsilon-input-file "{os.path.basename(eps_file)}")   ; voxel grid {eps.shape}',
                  "(set! geometry (list))", ""]
    else:
        scale = np.sqrt(1.0 + kappa)
        stamp = (periodic_stamp_segments(rods, defect_segments, box) if n_def
                 else np.zeros(0, dtype=[('rod', 'i4'), ('j', 'i4'), ('s0', 'f8'), ('s1', 'f8')]))
        lines += ["(set! geometry", " (list"]
        for i in np.flatnonzero(keep):
            rod = rods[i]
            b = float(rod['minor_radius'])
            on_rod = stamp[stamp['rod'] == i]
            pieces = _rod_pieces(rod, on_rod, box) if len(on_rod) else [(-np.inf, np.inf, 'rod')]
            for s0, s1, kind in pieces:
                f = 1.0 if kind == 'rod' else scale
                if f * b > 0.0:
                    lines.append("  " + _rod_object(rod['orientation'], rod['position'] / d, rod['z'] / d,
                                                     s0 / d, s1 / d, f * b / d, f * s * b / d))
        lines += [" ))", ""]
    kp = opt['k_points']
    if isinstance(kp, str):
        if kp != 'gamma':
            raise ValueError("k_points must be 'gamma' or a list of (k1, k2, k3)")
        kp = [(0.0, 0.0, 0.0)]
    lines.append("(set! k-points (list " + ' '.join(f"(vector3 {' '.join(_scm(c) for c in k)})" for k in kp) + "))")
    _ctl_run(lines, opt)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='\n') as fh:
        fh.write('\n'.join(lines))
    return path, dict(opt, gap_band=gap_band, cells=(Nx, Ny, Nz))


# high-symmetry points of the woodpile in the frame of its FCC cube (units 2 pi / a_c, cube axes
# e1 = (x + y)/sqrt2, e2 = (x - y)/sqrt2, e3 = z); exact for dz = sqrt(2) d.  The woodpile only has D2d
# symmetry, so points that are equivalent in FCC split by their component along the stacking axis z:
# X (z) vs X' (in-plane), W = (1, 1/2, 0) vs W' = (1, 0, 1/2).  For n = 3.3, ff = 0.40 the valence-band
# maximum sits at W' and the conduction-band minimum at L (dense 12^3 BZ scan); the plain FCC path
# Gamma-X-U-L-Gamma-X-W-K misses W' and overestimates the gap (15.8 % instead of 13.5 %).
WOODPILE_KPOINTS_FCC = {
    'Gamma': (0.0, 0.0, 0.0), 'X': (0.0, 0.0, 1.0), "X'": (1.0, 0.0, 0.0), 'L': (0.5, 0.5, 0.5),
    'U': (1.0, 0.25, 0.25), 'W': (1.0, 0.5, 0.0), "W'": (1.0, 0.0, 0.5), 'K': (0.75, 0.75, 0.0),
}
WOODPILE_KPATH = ('Gamma', 'X', 'U', 'L', 'Gamma', 'K', 'W', "X'", "W'", 'L')


def woodpile_kpoint_cartesian(v, d, dz):
    """
    Cartesian k (units 2 pi / d, woodpile frame: rods along x and y, stacking along z) of a point v given in
    the FCC cube frame (WOODPILE_KPOINTS_FCC).  In-plane: (x, y) = ((v1 + v2)/2, (v1 - v2)/2); along z:
    v3 d / dz, so X = (0, 0, d/dz) is the zone boundary 2 pi/dz of the stacking direction for any dz.
    """
    v = np.asarray(v, dtype=float)
    return np.array([0.5 * (v[0] + v[1]), 0.5 * (v[0] - v[1]), v[2] * d / dz])


def write_woodpile_primitive_ctl(path, d, dz, minor_radius, aspect_ratio, permittivity,
                                 background_permittivity, resolution=32, mesh_size=3, num_bands=8,
                                 k_path=WOODPILE_KPATH, k_interp=8, tolerance=1e-7, output_epsilon=True,
                                 title=None):
    """
    MPB band-structure file of the PERFECT woodpile in its primitive cell: body-centred tetragonal lattice
    a1 = (d, 0, 0), a2 = (0, d, 0), a3 = (d/2, d/2, dz/2) (FCC for dz = sqrt(2) d) with 2 rods: an x-rod at
    z = -h/2 and a y-rod at z = +h/2 (layers 0 and 1 of build_woodpile_rods; layers 2 and 3 are their
    images under a3); cylinders, or long ellipsoids for elliptical rods.  The complete gap lies between bands 2 and 3.  k_path: names of WOODPILE_KPOINTS_FCC
    (converted with woodpile_kpoint_cartesian and MPB's cartesian->reciprocal), interpolated with k_interp
    points per leg.  Lengths in units of d (nu = d/lambda).  Returns path.
    """
    h = dz / 4.0
    b = float(minor_radius)
    a = float(aspect_ratio) * b
    c = dz / (2.0 * d)
    lines = []
    _ctl_header(lines, title or "woodpile primitive cell (body-centred tetragonal, 2 rods)",
                permittivity, background_permittivity, d)
    lines += [f"; b = {b:g} um, a = {a:g} um, dz/d = {dz / d:.6g}; complete gap between bands 2 and 3",
              "(set! geometry-lattice (make lattice (basis1 1 0 0) (basis2 0 1 0) "
              f"(basis3 0.5 0.5 {_scm(c)}) (basis-size 1 1 {_scm(np.sqrt(0.5 + c * c))})))",
              "(define (c->l . args) (cartesian->lattice (apply vector3 args)))",
              "(define (c->r . args) (cartesian->reciprocal (apply vector3 args)))",
              "", "(set! geometry", " (list"]
    for axis, z0 in (('x', -h / 2.0), ('y', h / 2.0)):
        lines.append("  " + _rod_object(axis, 0.0, z0 / d, -np.inf, np.inf, b / d, a / d, c2l=True))
    lines += [" ))", ""]
    for name, v in WOODPILE_KPOINTS_FCC.items():
        kc = woodpile_kpoint_cartesian(v, d, dz)
        sym = name.replace("'", "p")
        lines.append(f"(define {sym} (c->r {' '.join(_scm(x) for x in kc)}))   ; {name}")
    path_syms = ' '.join(n.replace("'", "p") for n in k_path)
    lines += [f"(define-param k-interp {int(k_interp)})",
              f"(set! k-points (interpolate k-interp (list {path_syms})))   ; {' - '.join(k_path)}"]
    _ctl_run(lines, dict(resolution=resolution, mesh_size=mesh_size, num_bands=num_bands,
                         tolerance=tolerance, output_epsilon=output_epsilon))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='\n') as fh:
        fh.write('\n'.join(lines))
    return path


def read_mpb_freqs(out_file, prefix='freqs:'):
    """
    k-points (reciprocal basis, (n_k, 3)), |k| / 2 pi and frequencies (n_k, n_bands) from the `freqs:`
    lines of an MPB output (stdout of `mpb file.ctl > file.out`).  prefix='zevenfreqs:' etc. for run-zeven.
    """
    rows = [ln.strip().split(',') for ln in open(out_file)
            if ln.startswith(prefix + ',') and 'band 1' not in ln]
    if not rows:
        raise ValueError(f"no '{prefix}' lines in {out_file}")
    k = np.array([[float(x) for x in r[2:5]] for r in rows])
    kmag = np.array([float(r[5]) for r in rows])
    freqs = np.array([[float(x) for x in r[6:]] for r in rows])
    return k, kmag, freqs


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
    generate_ctl=False,
    ctl_options=None,
    periodic=None,
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
    written).  The float grid is never built: ff is the same voxel count, taken from the 2-D
    cross-section masks (no defects) or from boolean z-slabs (woodpile_voxel_ff).

    MPB export
    ----------
    generate_ctl=True writes MPB control files to dir_save (same stem as the HDF5 file, lengths in units of
    d, so MPB frequencies are nu = d/lambda):
    * <stem>.ctl: the box as one periodic supercell (write_woodpile_ctl), exact MPB geometric objects with
      the defects; by default Gamma only, num-bands = 1.25 x the gap band 4 Nx Ny Nz.
    * <stem>_primitive.ctl (defect-free structures only): band structure of the perfect crystal in its
      2-rod primitive cell along WOODPILE_KPATH (write_woodpile_primitive_ctl).
    The box must be a supercell: Lx = Nx d, Ly = Ny d, Lz = Nz dz (check_periodic_box).  ctl_options
    overrides CTL_DEFAULTS (resolution, mesh_size, num_bands, k_points, ...); the keys 'primitive_resolution',
    'primitive_num_bands' and 'k_interp' go to the primitive-cell file.  The paths are in info['ctl_files'].
    periodic (default: generate_ctl) treats the box as a periodic supercell when PLACING defects: every
    primary rod (primary_rods; boundary rods included) carries Lpar/d candidate segments, segments may wrap
    around the box, the crossing test is minimal-image, and the voxel grid stamps wrapped pieces on both
    faces and on the z images of rods overlapping the z faces (a > h/2), so eps, ff and the filling-fraction
    bisection describe the bulk crystal, exactly what the ctl describes.
    Without it the box faces get no defects, which a periodic MPB supercell would see as a defect-free plane.
    Periodic files with defects get a `_periodic` tag; the realization differs from periodic=False.

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
      interior layers are exact (periodic=False; with periodic=True they wrap around instead).
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
    periodic = bool(generate_ctl) if periodic is None else bool(periodic)
    segments_none = np.zeros(0, dtype=[('rod', 'i4'), ('j', 'i4'), ('s0', 'f8'), ('s1', 'f8')])

    def perfect_ff(b):
        # exact voxel ff of the defect-free crystal from the 2-D cross-section masks (no 3-D grid)
        rods_b, _ = build_woodpile_rods(box, d, dz, b, s, layer_offset, segment_ref)
        if periodic:                      # bulk crystal: rods overlapping the z faces wrap around
            rods_b, _ = _with_z_images(rods_b, segments_none, box, s * b)
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
    cells = check_periodic_box(box, d, dz, len(z_layers)) if periodic else None
    n_outside = int(np.sum(np.abs(rods['position']) > np.where(rods['orientation'] == 'x', Ly, Lx) / 2.0))
    if n_outside and verbose:
        print(f"[warn] {n_outside}/{len(rods)} rod axes fall outside the box (partial rods at the boundary).")

    if defect_density is not None:
        n_defects = int(round(float(defect_density) * Lx * Ly * Lz))
    n_defects = int(n_defects)
    # actual (post-rounding) defect density; 0.0 for a defect-free woodpile
    defect_density = n_defects / (Lx * Ly * Lz)
    segments = enumerate_segments(rods, box, d, periodic=periodic)
    chosen = (place_defects(rods, segments, n_defects, d, rng, forbid_crossing,
                            distribution=defect_distribution, box_size=box, periodic=periodic)
              if n_defects > 0 else segments[:0])
    # pieces actually stamped: wrapped segments split and copied onto the periodic images of boundary rods
    # and, along z, onto the images of the rods that overlap the z faces
    rods_st, stamp = rods, chosen
    if periodic:
        rods_st, stamp = _with_z_images(rods, periodic_stamp_segments(rods, chosen, box), box,
                                        a * max(1.0, np.sqrt(1.0 + kappa)))

    if verbose:
        print(f"[woodpile] {len(z_layers)} layers (h = {h:.4f}), {len(rods)} rods, "
              f"{len(segments)} complete segments, {len(chosen)} defects (kappa = {kappa}, "
              f"{defect_distribution})")

    if add_eps_dist:
        eps, coords = voxelize_woodpile(rods_st, box, grid, permittivity, background_permittivity, s,
                                        defect_segments=stamp, kappa=kappa, progress_every=progress_every)
        ff = float(np.mean(eps != np.float32(background_permittivity)))
    else:                                  # no float grid: count the occupied voxels in boolean z-slabs
        eps, coords = None, grid_coordinates(box, grid)
        ff = woodpile_voxel_ff(rods_st, box, grid, s, defect_segments=stamp, kappa=kappa)

    scale = float(np.sqrt(1.0 + kappa))
    defects = np.zeros(len(chosen), dtype=DEFECT_DTYPE)
    centres = segment_centres(rods, chosen, box)          # wrapping segments: centre mapped into the box
    for m, seg in enumerate(chosen):
        rod = rods[seg['rod']]
        p1, p2 = _endpoints(rod, seg['s0'], seg['s1'])    # unwrapped: p2 may lie beyond +L/2 (periodic)
        defects[m] = (*centres[m], *p1, *p2, rod['layer'], rod['orientation'], seg['rod'], seg['j'],
                      kappa, b * scale, a * scale)

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
        rods_outside_box=n_outside, periodic=periodic, cells=cells, ctl_files=[],
    )

    seed_str = "none" if seed is None else str(seed)
    tag = f"woodpile_d{d:.2f}_kappa{info['kappa']:+.2f}_rho{defect_density:.3f}_seed{seed_str}"
    if defect_distribution != 'random' and len(chosen) > 0:
        tag += f"_{defect_distribution}"     # random files keep their old names
    if periodic and len(chosen) > 0:
        tag += "_periodic"
    stem = rf"{dir_save}/n_{np.sqrt(permittivity):.2f}_ff_{ff:.4f}_{tag}"

    if generate_ctl:
        opt = dict(ctl_options or {})
        prim = dict(resolution=opt.pop('primitive_resolution', 32), num_bands=opt.pop('primitive_num_bands', 8),
                    k_interp=opt.pop('k_interp', 8))
        title = (f"woodpile n = {np.sqrt(permittivity):.3f}, ff = {ff:.4f}, b = {b:.5f} um, a/b = {s:g}, "
                 f"{len(chosen)} defects (kappa = {kappa:+g}, {defect_distribution})")
        geom = opt.get('geometry', CTL_DEFAULTS['geometry'])
        if geom == 'auto':
            geom = opt['geometry'] = 'voxels' if (s != 1.0 and len(chosen) > 0) else 'objects'
        eps_ctl = eps
        if geom == 'voxels' and eps_ctl is None:   # add_eps_dist=False: build the grid for MPB only
            eps_ctl, _ = voxelize_woodpile(rods_st, box, grid, permittivity, background_permittivity, s,
                                           defect_segments=stamp, kappa=kappa)
        path, used = write_woodpile_ctl(f"{stem}.ctl", rods, chosen, box, d, dz, s, permittivity,
                                        background_permittivity, kappa=kappa, title=title, eps=eps_ctl, **opt)
        info['ctl_files'].append(path)
        info['ctl_options'] = used
        if len(chosen) == 0:
            info['ctl_files'].append(write_woodpile_primitive_ctl(
                f"{stem}_primitive.ctl", d, dz, b, s, permittivity, background_permittivity,
                mesh_size=used['mesh_size'], **prim))
        if verbose:
            print(f"[ctl] supercell {used['cells']} -> gap above band {used['gap_band']}, "
                  f"num-bands {used['num_bands']}, resolution {used['resolution']}/d")
            for p_ in info['ctl_files']:
                print(f"[ctl] wrote {p_}")

    if save_rods:
        dir = dir_save
        os.makedirs(dir, exist_ok=True)
        # AM.create_hdf5_from_dict({"epsilon": eps}, rf"{dir}/n_{np.sqrt(permittivity):.2f}_ff_{ff:.4f}.h5")
        AM.create_hdf5_from_dict(
            {**({"epsilon": eps} if add_eps_dist else {}), **tables_to_dict(rods, defects),
             "params": {"box_size": np.array(box_size), "grid_size": np.array(grid_size), "d": d, "dz": dz,
                        "minor_radius": info['minor_radius'], "major_radius": info['major_radius'],
                        "aspect_ratio": aspect_ratio, "permittivity": permittivity, "background_permittivity": background_permittivity,
                        "kappa": info['kappa'], "defect_density": defect_density, "seed": -1 if seed is None else int(seed), "ff": ff,
                        "defect_distribution": defect_distribution, "periodic": int(periodic),
                        "ff_analytic": info['ff_analytic']}},
            rf"{stem}_tables.h5")
    if verbose:
        print(f"[woodpile] b = {b:.4f}, a = {a:.4f}  ->  ff(voxel) = {ff:.4f}, "
              f"ff(defect-free) = {ff_perfect:.4f}, ff(analytic, no overlap) = {ff_analytic:.4f}")
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
