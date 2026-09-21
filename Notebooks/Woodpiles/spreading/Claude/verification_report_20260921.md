# Verification report — `20260918_Beam_Spreading_Experiment.ipynb`

Date: 2026-09-21. Interpreter: `AppData\Local\Programs\Python\Python312\python.exe`, tidy3d 2.9.1.
No cloud task was uploaded, started or monitored; every check below is local. Cost numbers in item 6 are
copied from the executed notebook outputs (`run = False` branch).

Method: the sources of code cells 1, 3, 5 and 7 were `exec`-ed into one namespace (skipping `%matplotlib inline`,
loading `H:\Codes\tidy3d\.env` explicitly), then `build_simulation()` was called on

* **defect**: `../Structures/deffects_positive/n_2.50_ff_0.4229_woodpile_d1.20_kappa+1.20_rho0.100_seed12345_tables.h5`
* **crystal**: `../Structures/crystal/n_2.50_ff_0.3996_woodpile_d1.20_kappa+0.00_rho0.000_seednone_tables.h5`

Scratch scripts: `...\scratchpad\verify\verify_main.py`, `waist_test.py`, `dump_nb.py`.

## Summary

| # | Check | Result |
|---|---|---|
| 1a | cylinder count = rods + defects | PASS (332 = 210 + 122; 210 = 210 + 0) |
| 1b | tidy3d geometry vs stored `epsilon` mask on the h5 voxel grid | PASS (0 mismatched voxels of 2.40 M / 2.26 M rod voxels, 0.000 %) |
| 1c | all 122 defect centres inside geometry and eps-high in the h5 | PASS (122/122, 122/122) |
| 1d | defect / rod cross-section area ratio = 1 + κ | PASS (2.2048 vs 2.200; radius ratio 1.4849 vs 1.4832) |
| 2 | waist on the entrance face, sign of `waist_distance` | PASS (−1 µm → converging, w ≈ 1.5 µm on the face; +1 µm → ≈1.85–1.9 µm) |
| 3 | six `Absorber(num_layers=130)`, source/box/monitors inside `sim.size` | PASS |
| 4 | d(ν) ≤ 1/run_time, monitor positions/sizes/names | PASS (45.432 GHz ≤ 45.455 GHz) |
| 5 | `sim0` identical to `sim` except `structures=[]` | PASS (all 16 comparisons True) |
| 6 | cost / storage per structure | 3.20 / 3.23 / 3.21 / 3.22 FlexCredits; 0.435 GB per task |
| — | data-folder path depth `Path.cwd().parents[2]` | PASS (resolves to `H:\Codes\tidy3d`) |

Nothing is wrong in the notebook code. Six observations (none blocking) are listed at the end; the ones that
matter physically are (i) the free-space reference spot at the exit face is already ≈ the box half-width for
λ ≳ 3 µm, (ii) the monitor planes sit 46 nm *inside* the outermost rod layers, and (iii) `shutoff = 1e-20`
means `sim0` will almost certainly run the full 22 ps, so the "same or lower cost, ends earlier by shutoff"
remark is optimistic — budget ≈ 2 × 3.2 FC per structure, ≈ 26 FC for the four files.

---

## 1. Geometry

### 1a. Cylinder count

| file | `len(geometry.geometries)` | rods | defects | expected |
|---|---|---|---|---|
| defect κ=+1.2 | 332 | 210 | 122 | 332 |
| crystal | 210 | 210 | 0 | 210 |

`sim.structures` has exactly one `td.Structure`; its medium permittivity equals `params["permittivity"]` = 6.25.

### 1b. Geometry vs stored voxel array

`sim.epsilon(...)` samples on the simulation grid (156 × 156 × 110 cells inside the box), not on the h5 grid,
so the comparison was done directly with `GeometryGroup.inside(x, y, z)` on the h5 voxel centres
(`-L/2 + (i+0.5)·L/N`), **subsampled every 2nd voxel in each axis** (200 × 200 × 142 = 5.68 M points, ≈ 50 s and
≈ 33 s). Binary masks: tidy3d `inside` vs stored `epsilon > 3.5`.

| file | rod voxels (stored) | mismatched | td-only | stored-only | mismatch / rod voxels | ff stored | ff tidy3d | `params["ff"]` |
|---|---|---|---|---|---|---|---|---|
| defect | 2 396 007 | **0** | 0 | 0 | **0.000 %** | 0.4218 | 0.4218 | 0.4229 |
| crystal | 2 264 000 | **0** | 0 | 0 | **0.000 %** | 0.3986 | 0.3986 | 0.3996 |

The masks are bit-identical: the voxeliser that produced `epsilon` evidently used the same
centre-inside-cylinder rule on the same voxel centres. (The 0.001 difference to `params["ff"]` is the
subsampling.) Cross-check on the simulation grid with `sim.epsilon(box, coord_key='centers', freq=f0)`:
min 1.000, max 6.250, sub-pixel-averaged filling fraction ⟨(ε−1)/(ε_rod−1)⟩ = 0.4227 (defect) and 0.3992
(crystal), matching `params["ff"]` = 0.4229 / 0.3996 to 3 decimals.

### 1c. Defect centres

All 122 table centres `(x, y, z)` satisfy `geometry.inside == True`, and the stored `epsilon` at the
containing voxel is > 3.5 for 122/122. Max |centre − midpoint(x1..x2, y1..y2, z1..z2)| = 0.0 µm, so the
end-points and the centre columns are consistent.

### 1d. Defect cross-section vs parent rod

Defect #0: rod 170, orientation 0 (along x), centre (3.6, −3.6, 2.7577), `td.Cylinder(center=(3.6, −3.6,
2.7577), radius=0.38240, length=1.2, axis=0)`. Parent rod 170: `td.Cylinder(center=(0, −3.6, 2.7577),
radius=0.25781, length=12.0, axis=0)`, probed at x = −5.0 (8.6 µm from the nearest defect on that rod).

Counting `inside` on a 0.01 µm grid over ±0.6 µm in the (y, z) plane perpendicular to the axis, **evaluated on
the individual `td.Cylinder` objects** (see note below):

| | measured area (µm²) | π r² from table |
|---|---|---|
| defect | 0.4597 | 0.4594 |
| parent rod | 0.2085 | 0.2088 |
| **ratio** | **2.2048** | expected 1 + κ = **2.200** |
| radius ratio | 1.4849 | table 0.38240 / 0.25781 = 1.4832 = √2.2 |

Also verified on the full `GeometryGroup`: all 4597/4597 sample points of the disc r ≤ r_def around the
defect axis are `inside`, i.e. the union really contains the thick segment.

Note on method: evaluating the *whole group* in that perpendicular plane over ±0.6 µm gives 0.886 µm² for the
defect, because the plane (y, z) at x = 3.6 also slices the rods of the two adjacent layers (z ± 0.424 µm)
that run along y and lie in that plane. That is a property of the probe, not of the geometry, hence the
per-cylinder measurement. All 122 defects share radius 0.38240 µm and all 210 rods 0.25781 µm.

## 2. Waist

`td.GaussianBeam.__fields__['waist_distance'].field_info.description` (2.9.1):

> Distance from the beam waist along the propagation direction. **A positive value means the waist is
> positioned behind the source**, considering the propagation direction. For example, for a beam propagating
> in the `+` direction, a positive value of `beam_distance` means the beam waist is positioned in the `-`
> direction (behind the source). A negative value means the beam waist is in the `+` direction (in front of
> the source).

Built source: `GaussianBeam`, center (0, 0, −5.2426), size (9, 9, 0), direction `+`, `waist_radius` 1.5,
`waist_distance` **−1.0**, `pol_angle` 0. Entrance face z = −4.2426 is +1.00 µm in front of the source plane.
f0 = 117.78 THz → λ0 = 2.5455 µm, z_R = π w0²/λ0 = 2.777 µm.

Independent check (not the notebook's `beam_params` route, whose `scalar_field` also has a float/complex
in-place cast bug in 2.9.1): take the injected field `GaussianBeamProfile(...).field_data.Ex` on the source
plane (24 × 24 µm, 1024² points) and propagate it in free space by ±1 µm with an angular-spectrum
(exp(+i k_z Δz), the forward convention used by tidy3d's own `scalar_field`).

| `waist_distance` | w on source plane (2nd moment) | wavefront on source plane (phase(1 µm) − phase(0)) | w on entrance face, +1 µm (2nd moment / 1/e² crossing) | paraxial w(z) | w at −1 µm (behind source) |
|---|---|---|---|---|---|
| **−1.0 (used)** | 1.594 µm (analytic 1.594) | −0.144 rad → **converging** | **1.529 / 1.551 µm** | 1.500 | 1.819 |
| +1.0 | 1.594 µm | +0.144 rad → diverging | **1.945 / 1.920 µm** | 1.849 | 1.819 |

So with the sign the notebook uses the beam narrows from the source plane to the entrance face (waist in
front of the source, on the face); the opposite sign puts the waist 1 µm behind the source and the face sees
≈1.9 µm. The notebook's own cell 13 reports 1.5000 / 1.8486 µm; it uses tidy3d's `beam_params` w(z) and is
therefore a consistency check of the parametrisation rather than of the field, but its conclusion is the same.
The ≈3–5 % excess of the propagated widths over the paraxial formula is expected: w0 = 1.5 µm ≈ 0.6 λ0
(NA ≈ 0.5–0.75 across the band) is strongly non-paraxial and the injected field is the full vector beam.

## 3. Boundaries and layout

Both files: `boundary_spec.{x,y,z}.{plus,minus}` are all `td.Absorber` with `num_layers == 130` (12/12).

`sim.size` = (12, 12, 15.4853), `sim.center` = (0, 0, 0) → non-absorbing region x, y ∈ [−6, 6], z ∈ [−7.7426, 7.7426].

| object | centre | size | extent | inside `sim.size` |
|---|---|---|---|---|
| source | (0, 0, −5.2426) | (9, 9, 0) | x,y ∈ [−4.5, 4.5] | yes |
| crystal box (`params["box_size"]`) | (0, 0, 0) | (12, 12, 8.4853) | z ∈ [−4.2426, 4.2426] | yes |
| `monitorField_exit` | (0, 0, +4.2426) | (12, 12, 0) | | yes |
| `monitorField_entry` | (0, 0, −4.2426) | (12, 12, 0) | | yes |

Absorber layers are added outside: `sim.grid.boundaries.x[0]` = **−16.065 µm** vs −size_x/2 = −6.000 µm
(130 layers × 0.0774 µm = 10.065 µm each side); z[0] = −32.66 µm (defect) / −33.02 µm (crystal) vs −7.743 µm
(the grid coarsens to 0.19 µm in the air padding, so the z absorber is ≈25 µm thick). Total grid 415 × 415 ×
414 cells (155 cells across the 12 µm box), `num_cells` 7.13e7 (defect) / 7.08e7 (crystal).

## 4. Frequencies and monitors

| | value |
|---|---|
| `nfreqs` | 1415 |
| f range | 85.655 – 149.896 THz = λ 3.5 – 2.0 µm |
| **d(ν)** (uniform, min = max) | **45.432 GHz** |
| **1/run_time** (22 ps) | **45.455 GHz** → d(ν) ≤ 1/T holds; unaliased window 22.011 ps |
| monitor names | `['monitorField_exit', 'monitorField_entry']` |
| exit centre z | +4.242641 = +box_z/2 |
| entry centre z | −4.242641 = −box_z/2 |
| both sizes | (12.0, 12.0, 0.0) = (box_x, box_y, 0) |
| fields / interval_space / freqs | (Ex, Ey, Ez) / (2, 2, 1) / identical 1415-point array on both |
| source `GaussianPulse` | freq0 117.78 THz, fwidth 32.12 THz, offset 10 → f0 ± fwidth = [85.65, 149.90] THz = exactly the monitor band |

## 5. `sim0` vs `sim`

For both files, every comparison is True: `np.array_equal` of `grid.boundaries.x/y/z`; `sources[0].json()`;
each monitor's `.json()` (and same count); `run_time`; `boundary_spec.json()`; `size`; `center`; `shutoff`
(1e-20); `normalize_index` (None); `medium.json()` (air); `subpixel` (True); `sim0.structures == []`;
`sim0.grid_spec` is a `from_grid` custom-boundaries spec. Grid shapes 416 × 416 × 415 (defect) / 416 × 416 ×
412 (crystal) boundaries in both; `num_cells`, `num_time_steps` (149 535 / 149 145) and `dt` identical.

## 6. Cost and storage (from the executed cell 11 outputs)

| structure | est. cost (FlexCredits) | monitor storage per task | num_cells | time steps | cylinders |
|---|---|---|---|---|---|
| crystal κ=0, ff 0.3996 | 3.196 | 0.435 GB (0.217 + 0.217) | 7.078e7 | 149 145 | 210 |
| κ=+1.2, ρ=0.1, ff 0.4229 | 3.226 | 0.435 GB | 7.130e7 | 149 535 | 332 |
| κ=+1.6, ρ=0.1, ff 0.4300 | 3.210 | 0.435 GB | 7.113e7 | 149 107 | 332 |
| κ=+2.8, ρ=0.1, ff 0.4506 | 3.222 | 0.435 GB | 7.147e7 | 149 003 | 332 |

Sum of the four structure runs ≈ 12.85 FC; the four `sim0` references have the same mesh and `run_time` and
(see observation 3) will most likely cost the same, so ≈ 26 FC for the full loop. Storage check: 78 × 78 points
× 3 fields × 1415 freqs × 8 B = 0.207 GB per monitor, consistent with the 0.217 GB reported.

## Path depth

From cwd `H:\Codes\tidy3d\Notebooks\Woodpiles\spreading`, `Path.cwd().parents[2]` = `H:\Codes\tidy3d`, so the
bookkeeping folder is `H:\Codes\tidy3d/data/20260918_Beam_Spreading_Experiment_woodpiles/n_2.50` — correct
(mixed slash is harmless on Windows). `../Structures` and `../data` also resolve correctly (the `rglob` lists
the four `*_tables.h5`; the transmission file loads).

## Observations (nothing blocking)

1. **Reference spot vs box width.** With w0 = 1.5 µm the free-space beam at the exit-face distance (8.49 µm)
   has w ≈ 3.9 / 4.7 / 5.4 / 6.5 µm at λ = 2.0 / 2.5 / 2.9 / 3.5 µm, i.e. comparable to the box half-width
   (6 µm) over most of the band; the relative intensity at the box edge is 0.9 % / 4 % / 9 % / 18 % and for
   λ = 3.5 µm ≈ 6 % of the power lies outside |x| < 6 µm. The `sim0` exit spot and any σ² of the empty-box
   reference are therefore absorber-clipped for λ ≳ 3 µm, not only the spreading through the structure (the
   notebook's "pilot-box caveat" already says this qualitatively). For the transverse-size measurement itself
   a larger transverse box or a larger w0 (smaller NA) would be needed; nothing to change for a pilot.
2. **Monitor planes cut through the outermost rods.** Rod layers are at z = ±4.0305 (20 layers, spacing
   dz/4 = 0.4243) with radius 0.2578, so the outer rod surfaces are at ±4.2883 µm while `box_size`/2 =
   4.2426 µm: the `td.Cylinder`s protrude **0.046 µm** beyond the h5 box (the voxel array simply clips them),
   and both monitors sit 46 nm inside the last rod layer. Since E ∥ x is tangential to the top surface of the
   y-oriented exit rods this is continuous and harmless for |E|², but if a pure "in air" exit plane is wanted,
   move the exit monitor to z ≥ +4.29 µm (e.g. `z_face_out + 0.1`). The 1b mask comparison was restricted to
   the box, so this sliver is not counted there.
3. **`shutoff = 1e-20`** is far below the float32 noise floor of the solver (≈1e-7 relative), so it will
   effectively never trigger: both `sim` and `sim0` will run all ≈149 k steps. The cell-11 message
   "the reference sim0 ... ends earlier by shutoff" is therefore optimistic; expect the same ≈3.2 FC for each
   `sim0`. Same setting as the reference script, so this is a cost note, not a physics problem.
4. **Cell 13 is only semi-independent**: it builds I(x) from tidy3d's `beam_params` w(z) and then measures
   that I(x), so it verifies the parametrisation `w(z) = w0·√(1+((z+z0)/z_R)²)` rather than the injected
   field. The field-propagation check above confirms the sign independently, so nothing to change; the
   docstring quote in the markdown could be added for the record.
5. **Band edges at the 1/e amplitude of the pulse.** `fwidth = 0.5·(f_max − f_min)` puts the monitor band
   edges at f0 ± fwidth, where the Gaussian spectrum is at amplitude e^(−1/2) ≈ 0.61 (power 0.37). Normalising
   by `sim0` removes this, but the SNR at λ = 2.0 and 3.5 µm is a factor ≈3 lower than at the centre — same as
   the reference; fine.
6. Cosmetic: `id` shadows the builtin in cell 11; `verbose=True` for `web.upload` in the `run` branch is
   fine. Grid: rod diameter / dx = 6.7 cells (subpixel on) and (λ_min/n_rod)/dx = 10.3 steps — coarse but the
   notebook flags it explicitly.
