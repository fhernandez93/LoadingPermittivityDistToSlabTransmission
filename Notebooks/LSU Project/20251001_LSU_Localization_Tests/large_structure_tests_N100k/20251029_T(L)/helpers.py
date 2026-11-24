"""Helpers for the N100k T(L) runs: the LSU rod network built from tidy3d cylinders.

The voxelized slab (Structures/N100000/n_2.90_ff_0.2248_periodic.h5, a 1000^3 CustomMedium)
is too large for the tidy3d server ("Unable to allocate 3.73 GiB for an array with shape
(1000, 1000, 1000)"). Here the slab is built directly from the rod end points instead:
one td.Cylinder per rod, rotated and translated with td.Transformed, all in one GeometryGroup.

The geometry reproduces the voxelizer (20250903_create_h5_from_ends.ipynb):
  * the same unclipped rod file, rod radius and periodic box,
  * flat-ended cylinders (no spheres at the nodes),
  * periodic images of every rod that crosses a face of the box.

The simulation reproduces AM.loadAndRunStructure(direction="z", source="planewave",
monitors=["flux"], cut_cell=False, boundaries="periodic") so the task files stay readable by
AM.loadFromFile and comparable with the h5 runs:
  * x/y periodic with the box period, z absorbers,
  * slab of thickness box*cut centred at z=0 inside a fixed Lz = box + 2*spacing,
  * a cut keeps the bottom `cut` fraction of the cube along z (h5 planes 0..N*cut).
"""

from pathlib import Path

import numpy as np
import tidy3d as td
from tidy3d import web


# --------------------------------------------------------------------------- geometry

def load_rods(path):
    """Rod end points, shape (N, 6): x1 y1 z1 x2 y2 z2 in um."""
    rods = np.loadtxt(path, dtype=float, ndmin=2)
    if rods.shape[1] != 6:
        raise ValueError(f"{path}: expected 6 columns (two end points), got {rods.shape[1]}")
    return rods


def add_periodic_images(rods, box, radius):
    """Add the copy of every rod that, shifted by a lattice vector of the cubic box
    [-box/2, box/2]^3, comes within `radius` of the box. Same rule as add_images in the
    voxelizer notebook, so the result is periodic in x, y and z."""
    p1, p2 = rods[:, :3], rods[:, 3:]
    lo, hi = np.minimum(p1, p2) - radius, np.maximum(p1, p2) + radius
    h, extra = box / 2, []
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                if (sx, sy, sz) == (0, 0, 0):
                    continue
                s = np.array([sx, sy, sz]) * box
                ok = np.all((lo + s <= h) & (hi + s >= -h), axis=1)
                if ok.any():
                    extra.append(rods[ok] + np.tile(s, 2))
    return np.vstack([rods] + extra)


def cut_slab(rods, box, cut, radius):
    """Rods of the bottom `cut` fraction of the cube along z, shifted so the slab
    [-t/2, t/2] (t = box*cut) is centred at z=0, as loadAndRunStructure does with the h5
    (it keeps planes 0..N*cut and centres them). Returns (rods, t).

    A rod is kept when its bounding box, padded by `radius`, overlaps the slab volume
    (|x|, |y| <= box/2, |z| <= t/2). Parts that stick out along z are removed by the
    background caps of slab_structures; parts outside x/y lie outside the domain."""
    if not 0 < cut <= 1:
        raise ValueError(f"cut must be in (0, 1], got {cut}")
    t = box * cut
    shifted = rods + np.tile([0.0, 0.0, box / 2 - t / 2], 2)
    p1, p2 = shifted[:, :3], shifted[:, 3:]
    lo, hi = np.minimum(p1, p2) - radius, np.maximum(p1, p2) + radius
    half = np.array([box / 2, box / 2, t / 2])
    keep = np.all((lo <= half) & (hi >= -half), axis=1)
    return shifted[keep], t


def rod_transforms(rods):
    """4x4 transforms taking a z-axis cylinder centred at the origin onto each rod.
    Returns (transforms (N, 4, 4), lengths (N,))."""
    p1, p2 = rods[:, :3], rods[:, 3:]
    axis = p2 - p1
    lengths = np.linalg.norm(axis, axis=1)
    if np.any(lengths <= 0):
        raise ValueError(f"{np.count_nonzero(lengths <= 0)} rods have zero length")
    d = axis / lengths[:, None]

    # Rodrigues: rotation taking z_hat onto d. v = z x d, c = z.d, R = I + K + K^2 / (1 + c).
    v = np.stack([-d[:, 1], d[:, 0], np.zeros(len(d))], axis=1)
    c = d[:, 2]
    K = np.zeros((len(d), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    anti = c < -1 + 1e-12                               # d = -z: 1 + c -> 0, use a pi turn about x
    R = np.eye(3) + K + (K @ K) / np.where(anti, 1.0, 1 + c)[:, None, None]
    R[anti] = np.diag([1.0, -1.0, -1.0])

    T = np.zeros((len(d), 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = (p1 + p2) / 2
    T[:, 3, 3] = 1.0
    return T, lengths


def rod_geometries(rods, radius):
    """One td.Transformed(td.Cylinder) per rod."""
    transforms, lengths = rod_transforms(rods)
    return [
        td.Transformed(
            geometry=td.Cylinder(center=(0, 0, 0), radius=radius, length=float(L), axis=2),
            transform=T,
        )
        for T, L in zip(transforms, lengths)
    ]


def slab_structures(rods, radius, permittivity, t_slab, Lz, bg_permittivity=1.0):
    """[rods, cap below, cap above]. The caps are background-medium boxes covering |z| > t/2
    over the whole domain and absorbers; listed after the rods they override them, which clips
    the slab to |z| <= t/2 exactly like the voxel h5. They are the background, so removing
    every structure (reference run) leaves the same empty domain."""
    rods_structure = td.Structure(
        geometry=td.GeometryGroup(geometries=rod_geometries(rods, radius)),
        medium=td.Medium(permittivity=permittivity),
        name="lsu_rods",
    )
    bg = td.Medium(permittivity=bg_permittivity)
    z_far = 2 * Lz                                      # well past Lz/2 and the absorber layers
    caps = [
        td.Structure(geometry=td.Box.from_bounds((-td.inf, -td.inf, -z_far), (td.inf, td.inf, -t_slab / 2)),
                     medium=bg, name="cap_below"),
        td.Structure(geometry=td.Box.from_bounds((-td.inf, -td.inf, t_slab / 2), (td.inf, td.inf, z_far)),
                     medium=bg, name="cap_above"),
    ]
    return [rods_structure] + caps


# --------------------------------------------------------------------------- simulation

def build_simulation(slab_rods, t_slab, *, radius, box, lambda_range, runtime_ps, min_steps_per_lambda,
                     permittivity, n_freqs=400, shutoff=1e-9, absorbers=130, bg_permittivity=1.0,
                     width=0.4, spacing_factor=1.0, pol_angle=0.0, subpixel=True):
    """z-incidence plane-wave transmission simulation of the slab (rods and thickness from
    cut_slab), laid out as AM.loadAndRunStructure with direction="z", source="planewave",
    monitors=["flux"], cut_cell=False, periodic x/y.

    lambda_range = [lambda_max, lambda_min] in um (same order as the notebook's `lambdas`).
    Lz = box + 2*spacing (spacing = spacing_factor*lambda_max) does not depend on the cut,
    so one empty-domain reference serves every cut."""
    lambda_range = np.asarray(lambda_range, dtype=float)
    freq_range = td.C_0 / lambda_range                  # [f_min, f_max]
    freq0 = freq_range.sum() / 2
    fwidth = width * (freq_range[1] - freq_range[0])
    monitor_freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)

    spacing = spacing_factor * lambda_range[0]
    Lx = Ly = box
    Lz = box + 2 * spacing
    dl_min = (lambda_range[1] / min_steps_per_lambda) / np.sqrt(permittivity)
    structures = slab_structures(slab_rods, radius, permittivity, t_slab, Lz, bg_permittivity)

    source = td.PlaneWave(
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        size=(td.inf, td.inf, 0),
        center=(0, 0, -Lz / 2 + 1),
        direction="+",
        pol_angle=pol_angle,
        name="planewave",
    )
    flux_exit = td.FluxMonitor(center=(0, 0, (Lz - spacing) / 2), size=(td.inf, td.inf, 0),
                               freqs=monitor_freqs, name="flux1")
    flux_entry = td.FluxMonitor(center=(0, 0, -(Lz - spacing) / 2), size=(td.inf, td.inf, 0),
                                freqs=monitor_freqs, name="flux2")

    absorber = td.Boundary(plus=td.Absorber(num_layers=absorbers), minus=td.Absorber(num_layers=absorbers))
    return td.Simulation(
        center=(0, 0, 0),
        size=(Lx, Ly, Lz),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_lambda, wavelength=td.C_0 / freq0,
                                   dl_min=dl_min, max_scale=1.2),
        sources=[source],
        monitors=[flux_exit, flux_entry],
        run_time=runtime_ps,
        shutoff=shutoff,
        boundary_spec=td.BoundarySpec(x=td.Boundary.periodic(), y=td.Boundary.periodic(), z=absorber),
        normalize_index=None,
        structures=structures,
        subpixel=subpixel,
        medium=td.Medium(permittivity=bg_permittivity),
    )


def plot_layout(sim, freq=None):
    """eps at the grid cell centres on x=0 (side view, with source and flux planes) and z=0
    (top view). Uses sim.epsilon instead of sim.plot_eps: plotting the rotated cylinders with
    plot_eps goes through trimesh, which needs networkx (not in the tidy3d env)."""
    import matplotlib.pyplot as plt
    freq = freq if freq is not None else sim.sources[0].source_time.freq0
    Lx, Ly, Lz = sim.size
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), tight_layout=True)
    for ax, plane, (h, v) in zip(axes, [(0, Ly, Lz), (Lx, Ly, 0)], [("y", "z"), ("x", "y")]):
        eps = sim.epsilon(box=td.Box(center=(0, 0, 0), size=plane), coord_key="centers", freq=freq)
        eps = np.real(eps).squeeze(drop=True)
        im = ax.pcolormesh(eps[h], eps[v], eps.transpose(v, h).values, cmap="gist_yarg", shading="auto")
        fig.colorbar(im, ax=ax, label="eps")
        ax.set_aspect("equal")
        ax.set_xlabel(f"{h} (um)")
        ax.set_ylabel(f"{v} (um)")
    axes[0].set_title("x = 0")
    axes[1].set_title("z = 0")
    axes[0].axhline(sim.sources[0].center[2], color="r", lw=1)          # source
    for m in sim.monitors:
        axes[0].axhline(m.center[2], color="b", ls="--", lw=1)            # flux planes
    plt.show()


# --------------------------------------------------------------------------- tasks

def submit(sim, *, project_name, sim_name, run_file, ref_file, run=False):
    """run=True: upload, start and monitor; the task id goes to run_file as "\\n<task id>"
    (the empty-domain reference is run once first and stored as "\\n<ref id>" in ref_file).
    run=False: upload as a draft, print the cost estimate and delete the draft."""
    run_file, ref_file = Path(run_file), Path(ref_file)
    if not run:
        task_id = web.upload(sim, folder_name=project_name, task_name=sim_name, verbose=True)
        try:
            cost = web.estimate_cost(task_id, verbose=False)   # upload already logged the estimate
            print(f"{sim_name}: estimated cost {cost} FlexCredits (draft {task_id}, not started)")
        finally:
            web.delete(task_id)
        return None

    if not ref_file.exists():
        sim0 = sim.copy(update={"structures": []})
        id0 = web.upload(sim0, folder_name=project_name, task_name="LSU_reference", verbose=True)
        web.start(task_id=id0)
        ref_file.write_text("\n" + id0)
        web.monitor(id0)

    task_id = web.upload(sim, folder_name=project_name, task_name=sim_name, verbose=True)
    web.start(task_id=task_id)
    run_file.write_text("\n" + task_id)
    web.monitor(task_id)
    return task_id
