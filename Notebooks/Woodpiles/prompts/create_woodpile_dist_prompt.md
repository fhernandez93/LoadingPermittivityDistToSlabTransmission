# Prompt: build `create_woodpile_dist.ipynb` (woodpile photonic crystal with controlled defects)

## Goal

Fill the empty notebook `H:\Codes\tidy3d\Notebooks\Woodpiles\create_woodpile_dist.ipynb` with a function that
generates a voxelized permittivity distribution of a **woodpile photonic crystal with controlled rod-segment
defects**, following the geometry of

> S. Aeby, G. J. Aubry, N. Muller, F. Scheffold, *Scattering From Controlled Defects in Woodpile Photonic
> Crystals*, Adv. Optical Mater. 9, 2001699 (2021).
> PDF: `H:\Codes\tidy3d\Notebooks\Woodpiles\Papers\Advanced Optical Materials - 2021 - Aeby - Scattering From Controlled Defects in Woodpile Photonic Crystals.pdf`

Read the paper first (pages 1-2 and the Experimental Section on page 6 carry the geometry). The output of the
notebook is meant to be saved as an `.h5` file and loaded into Tidy3D through `AutomationModule`, exactly like the
existing random-slab workflow.

## Reference implementation to mirror

`H:\Codes\tidy3d\Notebooks\LSU Project\20251001_LSU_Localization_Tests\20250903_create_h5_from_ends.ipynb`

That notebook defines `create_permittivity_grid_penlike(rod_endpoints, grid_size, minor_radius, aspect_ratio,
permittivity, box_size, ...)`. It takes a list of rod endpoints `(x1,y1,z1,x2,y2,z2)`, and for each rod:

1. computes a padded axis-aligned bounding box in voxel indices (`idx_range_for_world`),
2. builds a right circular cylinder in an "unwarped" space `z' = z / s` and applies a global z-scale `s`
   (`aspect_ratio`) so that the cross-section becomes an ellipse with major axis along z,
3. writes `permittivity` into the voxels inside the rod (background is 1),
4. returns `(grid, b_array, ff)` where `ff` is the voxel filling fraction.

Grid convention to keep: `grid_coords = (arange(G) + 0.5) * dx - box/2` (voxel centres, box centred at origin),
`float32` array, background permittivity written first, rods overwrite. Saving convention to keep:

```python
AM.create_hdf5_from_dict({"epsilon": grid}, f"{dir}/n_{np.sqrt(PERM):.2f}_ff_{ff:.4f}.h5")
```

with `sys.path.append(os.path.abspath(r'H:\codes\tidy3d')); import AutomationModule as AM`.

Reuse the voxelization core (AABB + unwarped-circle test + z-warp). Do **not** rewrite it as a brute-force full-grid
mask per rod; the target grids are large (650^3 in the reference notebook).

## Woodpile geometry (from the paper)

* Layers of parallel rods stacked along **z**. In-plane rod pitch `d` (paper: 1.2 um). Each layer is rotated by
  90 deg with respect to the previous one, and every second layer of the same orientation is shifted by `d/2`.
  The structure therefore repeats every **4 layers** with period `d_z`, and `d_z / d = sqrt(2)` gives an FCC lattice.
  Layer spacing is `d_z / 4`.
* Rods have an **elliptical cross-section** with the long axis along z (DLW voxel shape). Paper values:
  semi-axes `a = 0.42 um` (long, along z) and `b = 0.15 um` (short, in-plane), aspect ratio `a/b = 2.8`,
  equivalent radius `r = sqrt(a b) = 0.25 um`. Adjacent layers overlap in z.
* Paper sample: `70 x 70 x 8.5 um^3`, 20 layers = 5 FCC unit cells along z, polymer filling fraction ~0.35,
  `n_rod = 1.53` (IP-Dip), background air.
* A **rod segment** is the piece of rod between two consecutive crossings with the rods of the neighbouring
  layers, i.e. of length `d`.

## Defects (from the paper)

* A defect is one rod segment whose cross-sectional **area** is changed by a factor `kappa`:
  `A_defect = (1 + kappa) * A_rod`. Positive `kappa` = thicker segment, negative `kappa` = thinner.
  `kappa = -1` (-100 %) is a missing segment; `kappa = +3.1` (+310 %) fills the interstitial space between
  parallel rods. Keep the aspect ratio fixed, so both semi-axes scale by `sqrt(1 + kappa)`.
* Defects are placed **uniformly at random** over the sample volume with number density `rho` (paper range
  0 to 0.24 um^-3), under the constraint that **two defects never overlap** (never share a segment; also do not put
  two defects on segments that touch each other in-plane or across adjacent layers, so the "non-overlap" rule of
  the paper holds).
* Paper filling fraction bookkeeping: `phi = (1 - rho V_0) phi_0 + rho V_{0,d}` with `V_0` the unit-cell volume.

## Function to write

```python
def create_woodpile_dist(
    box_size,                 # float or (Lx, Ly, Lz), physical units (um); box centred at origin
    grid_size,                # int or (Nx, Ny, Nz) voxels
    d,                        # in-plane rod pitch
    dz=None,                  # stacking period (4 layers); default sqrt(2)*d (FCC)
    permittivity=1.53**2,     # rod permittivity
    background_permittivity=1.0,
    minor_radius=None,        # short semi-axis b (in-plane); give this OR filling_fraction
    filling_fraction=None,    # target voxel ff; solve for minor_radius at fixed aspect_ratio
    aspect_ratio=2.8,         # a/b, long axis along z; 1.0 = circular rods
    n_defects=0,              # integer count; alternatively defect_density (um^-3) -> n = round(rho * V)
    defect_density=None,
    kappa=0.0,                # relative change of cross-section area of a defect segment
    seed=None,                # numpy Generator seed for defect placement (and for reproducibility)
    layer_offset=0.0,         # optional z offset of the first layer
    progress_every=None,
    verbose=False,
):
    ...
    return eps, rods, defects, ff
```

Required return values:

* `eps`: `(Nx, Ny, Nz)` float32 permittivity grid.
* `rods`: structured array or dict with, for every rod (full rod, spanning the box): endpoints `(x1,y1,z1,x2,y2,z2)`,
  layer index, orientation (`'x'` or `'y'`), in-plane position, `minor_radius`, `major_radius`.
* `defects`: same style, one entry per defect: segment centre `(x,y,z)`, segment endpoints, layer index,
  orientation, `kappa`, defect `minor_radius`/`major_radius`, and the index of the parent rod.
* `ff`: voxel filling fraction of the rod material (`eps != background`), computed the same way as the reference
  notebook, plus (as a separate quantity, e.g. in a returned dict or printed) the analytic no-overlap estimate
  `pi a b / (d * d_z / 4)` so the two can be compared.

Implementation notes:

* Build the rod list from the lattice rules first (as endpoints, like the `.dat` input of the reference notebook),
  then voxelize. Rods must span the full box in their direction; clip nothing.
* The `filling_fraction` branch should solve for `minor_radius` by bisection on the **voxel** ff at the requested
  resolution (overlap between layers makes the analytic formula overestimate ff), and report the residual.
* Defects: pick segments at random without replacement, enforce the non-overlap rule, and voxelize the segment with
  the scaled semi-axes. For `kappa < 0` the defect voxels must be reset to background before the thinner segment
  is written (otherwise the original rod remains). For `kappa = -1` the segment is removed entirely. Segment
  endpoints are the crossing points with the neighbouring layers' rods (positions `n*d` or `n*d + d/2`), so the
  rod is partitioned into segments of length `d` consistently with the paper's Figure 1.
* Handle rods that fall partly outside the box gracefully (same warning style as the reference notebook).
* Keep everything in numpy; no tidy3d import needed for generation. Use
  `C:\Users\HernandF\AppData\Local\Programs\Python\Python312\python.exe` if you execute the notebook headless
  (it is the only env with the full stack; see `AutomationModule`).

## Notebook layout

1. Imports + `sys.path` for `AutomationModule`.
2. The function, with a docstring that states the geometry conventions and the meaning of `kappa`.
3. A parameter cell reproducing the paper's crystal
   (`d=1.2, dz=sqrt(2)*1.2, minor_radius=0.15, aspect_ratio=2.8, n=1.53, box=(L, L, 5*dz)`), at a modest
   resolution for a quick run, plus a second cell with positive and negative defects.
4. Sanity plots: xz and yz cross-sections through a rod axis, one xy slice per layer orientation, a histogram of
   defect positions, and the printed ff (voxel vs analytic).
5. A commented-out save cell using `AM.create_hdf5_from_dict` with the `n_{n:.2f}_ff_{ff:.4f}.h5` naming, plus a
   second file with the `rods` and `defects` tables so the defect positions are recoverable.

## Verification (required)

After the notebook runs, **spawn a separate agent to verify the code independently** (do not verify it yourself
only). Give the agent the notebook path, the paper path and this prompt, and ask it to run its own checks with the
Python312 interpreter and report numbers, not opinions. At minimum it must check:

1. **Stacking rules**: from `rods`, confirm layers alternate orientation x/y, that layers `i` and `i+2` are shifted by
   exactly `d/2`, and that the z-period is `dz` (4 layers). Cross-check against a slice of `eps`.
2. **Cross-section**: on an xz slice through a y-oriented rod, measure the extent along z and along x and confirm
   `2a` and `2b` within one voxel; confirm the ellipse's long axis is along z.
3. **Filling fraction**: compare the returned voxel `ff` to the analytic no-overlap value for the paper parameters
   and confirm the voxel value is **lower** (layer overlap) and lands near the paper's ~0.35; run at two resolutions
   and confirm convergence. With `filling_fraction=` given, confirm the achieved ff matches within tolerance.
4. **Defects**: `len(defects) == n_defects`; for `kappa > 0` the defect segments have larger measured cross-section
   than regular rods by a factor `(1 + kappa)` in area; for `kappa < 0` smaller; for `kappa = -1` the segment is
   gone (background) while the neighbouring segments of the same rod are intact. Confirm no two defects share or
   touch segments. Confirm `seed` gives reproducible placement.
5. **Grid convention**: voxel centres symmetric about the origin, `eps.shape == grid_size`, dtype float32,
   background voxels equal `background_permittivity` (test with a value other than 1).
6. **Edge behaviour**: rods do not leak across the box boundary (no wrap-around), and a box that is not a multiple
   of `d` or `dz` still produces a well-formed lattice.

Report every check as PASS/FAIL with the measured numbers. Fix anything that fails, rerun the verification, and
summarize what changed.
