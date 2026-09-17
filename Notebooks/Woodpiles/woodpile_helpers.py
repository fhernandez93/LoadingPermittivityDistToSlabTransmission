"""
woodpile_helpers.py
===================
Voxelized woodpile photonic crystal with controlled rod-segment defects
(S. Aeby, G. J. Aubry, N. Muller, F. Scheffold, Adv. Optical Mater. 9, 2001699 (2021)).

Main entry point: create_woodpile_dist(...) -> eps, rods, defects, ff, info
Companion notebook: create_woodpile_dist.ipynb

The voxelization core (padded AABB per rod, circular membership test in the unwarped space
z' = z / s, global z-scale s = aspect_ratio) mirrors create_permittivity_grid_penlike from
LSU Project/20251001_LSU_Localization_Tests/20250903_create_h5_from_ends.ipynb.
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
           'voxelize_woodpile', 'voxelize_rod', 'grid_coordinates', 'tables_to_dict', 'show_slice',
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


def _segments_conflict(cand, acc, rods, d, tol=1e-6):
    """
    Non-overlap rule between one candidate segment and the accepted ones (vectorized).
    Two segments conflict when they
      * lie on the same rod and are the same or adjacent segments (share an endpoint),
      * lie on neighbouring rods (distance d) of the same layer and overlap along the axis,
      * lie in adjacent layers and cross (touch) each other.
    Segments two or more layers apart never conflict.
    """
    if len(acc) == 0:
        return False
    rc = rods[cand['rod']]
    ra = rods[acc['rod']]
    same_rod = acc['rod'] == cand['rod']
    same_layer = ra['layer'] == rc['layer']
    adj_layer = np.abs(ra['layer'] - rc['layer']) == 1

    c1 = same_rod & (np.abs(acc['j'] - cand['j']) <= 1)
    neighbour_rod = same_layer & (~same_rod) & (np.abs(ra['position'] - rc['position']) <= d * (1 + tol))
    c2 = neighbour_rod & (cand['s0'] < acc['s1'] - tol) & (acc['s0'] < cand['s1'] - tol)
    c3 = adj_layer & (ra['position'] >= cand['s0'] - tol) & (ra['position'] <= cand['s1'] + tol) \
         & (rc['position'] >= acc['s0'] - tol) & (rc['position'] <= acc['s1'] + tol)
    return bool(np.any(c1 | c2 | c3))


def place_defects(rods, segments, n_defects, d, rng):
    """Uniformly random choice of n_defects segments without replacement, honouring the non-overlap rule."""
    if n_defects <= 0:
        return segments[:0]
    order = rng.permutation(len(segments))
    accepted = segments[:0]
    for idx in order:
        if len(accepted) >= n_defects:
            break
        c = segments[idx]
        if not _segments_conflict(c, accepted, rods, d):
            accepted = np.append(accepted, segments[idx:idx + 1])
    if len(accepted) < n_defects:
        raise ValueError(f"could only place {len(accepted)} of {n_defects} non-overlapping defects "
                         f"({len(segments)} candidate segments); lower n_defects/defect_density")
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
    """Voxelize the rod list (plus optional defect segments) into a float32 permittivity grid."""
    grid = _as_triple(grid_size, int)
    coords = grid_coordinates(box_size, grid)
    eps = np.full(grid, background_permittivity, dtype=np.float32)
    s = float(aspect_ratio)
    scale = np.sqrt(1.0 + kappa)          # both semi-axes scale so that the AREA scales by (1 + kappa)
    if defect_segments is None:
        defect_segments = np.zeros(0, dtype=[('rod', 'i4'), ('j', 'i4'), ('s0', 'f8'), ('s1', 'f8')])
    for i, rod in enumerate(rods):
        if progress_every and i % progress_every == 0:
            print(f"[voxelize] rod {i} / {len(rods)}")
        b = float(rod['minor_radius'])
        on_rod = defect_segments[defect_segments['rod'] == i]
        for s0, s1, kind in _rod_pieces(rod, on_rod, box_size):
            p1, p2 = _endpoints(rod, s0, s1)
            bb = b if kind == 'rod' else b * scale
            if bb > 0.0:
                voxelize_rod(eps, coords, p1, p2, bb, s, permittivity)
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
    ff_tolerance=1e-3,
    ff_max_iter=25,
    save_rods=False,
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
    box, subject to the non-overlap rule: no two defects on the same or adjacent segments of a
    rod, on overlapping segments of neighbouring rods of the same layer, or on crossing
    (touching) segments of adjacent layers.  n_defects = round(defect_density * Lx*Ly*Lz) when
    defect_density is given.

    Returns
    -------
    eps      : (Nx, Ny, Nz) float32 permittivity grid (background first, rods overwrite).
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

    def perfect_ff(b):
        rods_b, _ = build_woodpile_rods(box, d, dz, b, s, layer_offset, segment_ref)
        eps_b, _ = voxelize_woodpile(rods_b, box, grid, permittivity, background_permittivity, s)
        return float(np.mean(eps_b != np.float32(background_permittivity)))

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
    chosen = place_defects(rods, segments, n_defects, d, rng) if n_defects > 0 else segments[:0]

    if verbose:
        print(f"[woodpile] {len(z_layers)} layers (h = {h:.4f}), {len(rods)} rods, "
              f"{len(segments)} complete segments, {len(chosen)} defects (kappa = {kappa})")

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
        n_defects=len(chosen), kappa=kappa,
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
        AM.create_hdf5_from_dict({"epsilon": eps}, rf"{dir}/n_{np.sqrt(permittivity):.2f}_ff_{ff:.4f}.h5")
        AM.create_hdf5_from_dict(
            {**tables_to_dict(rods, defects),
             "params": {"box_size": np.array(box_size), "grid_size": np.array(grid_size), "d": d, "dz": dz,
                        "minor_radius": info['minor_radius'], "major_radius": info['major_radius'],
                        "aspect_ratio": aspect_ratio, "permittivity": permittivity, "background_permittivity": background_permittivity,
                        "kappa": info['kappa'], "defect_density": defect_density, "seed": -1 if seed is None else int(seed), "ff": ff,
                        "ff_analytic": info['ff_analytic']}},
            rf"{dir}/n_{np.sqrt(permittivity):.2f}_ff_{ff:.4f}_{tag}_tables.h5")
    if verbose:
        print(f"[woodpile] b = {b:.4f}, a = {a:.4f}  ->  ff(voxel) = {ff:.4f}, "
              f"ff(defect-free) = {ff_perfect:.4f}, ff(analytic, no overlap) = {ff_analytic:.4f}")
    return eps, rods, defects, ff, info


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
