# Prompt: build the woodpile **beam-spreading** experiment (focused beam, exit-face field, native tidy3d cylinders)

## Goal

Replicate the LSU "beam spreading vs frequency" numerical experiment on the **woodpile photonic crystals** already
generated in this folder. This is a *different* experiment from the transmission run in
`20260917_Transmission_Experiment.ipynb`: instead of a plane wave + flux monitors, we launch a **focused Gaussian
beam** onto the entrance face and record the **complex E-field on the exit face** (frequency-domain
`FieldMonitor`, not just flux), so that the transverse intensity profile `I(x, y; ν)` and its width
`σ²(ν)`, `d(ν)` can be analysed afterwards.

**The structure must be built from native tidy3d geometry (`td.Cylinder` objects in a `td.GeometryGroup`),
exactly like the reference script — NOT from the voxelised `epsilon` array and NOT through
`AM.loadAndRunStructure`.** The `rods` and `defects` tables stored in each structure `.h5` give every cylinder's
endpoints and radius; use those.

Deliver two notebooks in `H:\Codes\tidy3d\Notebooks\Woodpiles\spreading`:

1. `20260918_Beam_Spreading_Experiment.ipynb` — builds the simulations (and, only when explicitly enabled, submits).
2. `20260918_Beam_Spreading_retrieval.ipynb` — downloads the field data and saves a compact `.h5` under `./data`.

## Reference implementation to mirror

`H:\Codes\tidy3d\Notebooks\LSU Project\20251001_LSU_Localization_Tests\20260601_Beam_Spreading_freq\20251001_numerical_experiment_using_td_cylinders.py`

Read it first. The pieces to carry over are:

* **Geometry as tidy3d cylinders.** The reference reads a `.txt` of rod end-points, calls
  `create_cylinder_from_ends(top, bottom, radius)` for each (a `td.Cylinder` rotated/translated with
  `td.Transformed`), collects them in `td.GeometryGroup(geometries=cyl_group)` and wraps that in one
  `td.Structure(medium=td.Medium(permittivity=n**2))`. Do the same here. Since woodpile rods are axis-aligned
  you may skip the rotation matrix and use `td.Cylinder(center, radius, length, axis=0|1)` directly — but keep the
  `create_cylinder_from_ends` helper (endpoint-based) so the code path is the same for rods and defect segments.
* A **finite-size source** centred on the entrance face (there: a 5×5 µm `PlaneWave` patch at
  `z = -t_slab/2 - 3`, `direction='+'`, `GaussianPulse(freq0=f0, fwidth=0.5*(f_max-f_min), offset=10)`).
  Here use a `td.GaussianBeam` so the beam is a true focused beam with a defined waist (see below).
* An **exit-face `FieldMonitor`** (`center=[0,0,+t_slab/2]`, `size=[Lx, Ly, 0]`, `fields=["Ex","Ey","Ez"]`,
  `freqs=monitor_freqs`, `interval_space=(3,3,3)` or coarser). `normalize_index=None`.
* **Absorbing boundaries on all six sides** (`td.Absorber(num_layers=...)` in x, y, z — no periodic transverse
  boundaries; the beam must be able to leave the box laterally).
* Air padding above and below the slab (there `Lz = 45` for a 32 µm slab). The two `cube` structures are
  `n_eff = 1.0` dummies: drop them.
* `td.GridSpec.auto(min_steps_per_wvl=..., wavelength=lambdas[0], dl_min=dl, max_scale=1.2)` with
  `dl = (λ_min / min_steps_per_lambda) / n` so the smallest wavelength *inside the rods* is resolved.
* The `spectral_sampling(T_ps, band_width, a, lambdas)` helper: choose `nfreqs` so the frequency grid is fine
  enough to IFFT the field to an unaliased time window ≥ `run_time` (`dν ≤ 1/T`). Keep the helper and print its
  report.
* The submission bookkeeping: `web.upload(sim, folder_name=project_name, task_name=sim_name)` → write the task id to
  `{Path.cwd().parents[2]}/data/{project_name}/n_{n:.2f}/{sim_name}.txt` → `web.start` / `web.monitor`; skip if the
  `.txt` already exists ("Exist!"). Keep the reference's `run = False` branch that only calls `web.estimate_cost`
  (upload + estimate + `web.delete`) and plots the layout.

`20260917_Transmission_Experiment.ipynb` is only a template for the *folder walk*, the `params` read
(`AM.read_hdf5_as_dict(file)["params"]`) and the empty-box reference (`sim0 = sim.copy(update={"structures": []})`,
uploaded once as `sim_name + "_0"`). Do not copy its `AM.loadAndRunStructure` call.

## Deliberate differences from the reference script (state these in the notebook's first markdown cell)

* **API key.** The reference never calls `web.configure`; it works because `AutomationModule` was imported
  elsewhere. Here the experiment notebook bypasses `AM.loadAndRunStructure`, so call
  `web.configure(os.environ["API_TIDY3D_KEY"])` explicitly after `load_dotenv()` (the retrieval notebook gets it
  through `AM.loadFromFile(key=...)`).
* **No `numpy-stl`.** `from stl import mesh` is only used for the rotation matrix of tilted cylinders. Woodpile rods
  are axis-aligned, so `create_cylinder_from_ends` here builds `td.Cylinder(axis=...)` from the end-points
  directly; assert the two end-points differ in exactly one coordinate and raise otherwise.
* **No rescaling, no clipping, no hard-coded radius.** The reference divides positions by 0.8 (file was at
  0.8-scale), drops cylinders whose centre lies outside `±t_slab_z/2 ± 0.8`, and uses `radius = 0.42`. None of that
  applies: tables are in µm at scale 1, all rods lie inside the box, radii come from the tables.
* **Source.** The reference uses a 5×5 µm `PlaneWave` patch; we use `td.GaussianBeam`. Keep a `source_kind`
  switch (`"gaussian"` default, `"planewave_patch"` reproducing the reference's patch at `z = -t_slab_z/2 - 1`
  with `size=(S, S, 0)`) so a like-for-like comparison with the LSU runs stays possible.
* **Absorbers.** Reference: 200 layers; transmission woodpile run: 130. Use 130 (parameter) and say so.
* **Empty-box reference `sim0`.** Not in the reference script (added from the transmission notebook) — needed to
  normalise the exit field and measure the free-space spot.
* **Task name.** The reference encodes the band, `LSU_{stem}_lambda_{λ_min}_{λ_max}`. Do the same
  (`f"{stem}_lambda_{lambdas[1]}_{lambdas[0]}"`), otherwise a rerun with a different band is skipped by the
  "Exist!" check.
* **No blanket `try/except`** around the file loop (the reference swallows errors and prints them); let
  exceptions propagate so a malformed table is not silently skipped.
* **Maxwell-Garnett `e_eff` / `n_eff` block and `cube1/cube2`**: dropped (they were `n_eff = 1` no-ops).

## Structures to use and how to read them

All under `H:\Codes\tidy3d\Notebooks\Woodpiles\Structures\`:

| folder | file | ff | κ | ρ (µm⁻³) |
|---|---|---|---|---|
| `crystal/` | `n_2.50_ff_0.3996_woodpile_d1.20_kappa+0.00_rho0.000_seednone_tables.h5` | 0.3996 | 0 | 0 |
| `deffects_positive/` | `n_2.50_ff_0.4229_woodpile_d1.20_kappa+1.20_rho0.100_seed12345_tables.h5` | 0.4229 | +1.2 | 0.10 |
| `deffects_positive/` | `n_2.50_ff_0.4300_woodpile_d1.20_kappa+1.60_rho0.100_seed12345_tables.h5` | 0.4300 | +1.6 | 0.10 |
| `deffects_positive/` | `n_2.50_ff_0.4506_woodpile_d1.20_kappa+2.80_rho0.100_seed12345_tables.h5` | 0.4506 | +2.8 | 0.10 |

Each file has four groups. Ignore `epsilon` entirely. Use:

* `params`: `permittivity = 6.25` (`n_rod = 2.5`), `background_permittivity = 1.0`, `d = 1.2 µm`,
  `dz = √2·d = 1.697 µm`, `box_size = (12, 12, 8.485) µm` (10×10 pitches, 5 FCC cells = 20 layers along z),
  `aspect_ratio = 1` (circular rods), `minor_radius = major_radius = 0.2578 µm`, `kappa`, `defect_density`, `seed`.
  Take everything from `params` — do not hard-code.
* `rods` (210 entries per file): `x1,y1,z1,x2,y2,z2` (full-length rod end-points spanning the box, ±6 µm),
  `orientation` **int8: 0 = rod along x, 1 = rod along y** (verified from the end-points), `position` (in-plane
  coordinate of the axis), `z` (layer height), `layer`, `minor_radius`, `major_radius`, `seg_origin`.
* `defects` (0 entries for the crystal, 122 for the ρ = 0.1 files): one entry per defect segment with
  `x1,y1,z1,x2,y2,z2` (segment end-points, length `d`), centre `x,y,z`, `rod_index` (row of the parent rod in
  `rods`), `segment_index`, `orientation`, `layer`, `kappa`, and the defect's own `minor_radius`/`major_radius`
  (= rod radius × `√(1+κ)`, e.g. 0.3824 µm for κ = +1.2).

**Building the geometry from the tables** (write a function `woodpile_geometry_from_tables(rods, defects, ...)`):

1. For every rod, add `create_cylinder_from_ends((x2,y2,z2), (x1,y1,z1), rod.minor_radius)`.
2. For every defect, add `create_cylinder_from_ends((x2,y2,z2), (x1,y1,z1), defect.minor_radius)`. Because all
   defects here are **positive** (κ > 0, thicker segments) the defect cylinder simply overlaps the thinner parent
   rod — a `GeometryGroup` is a union, so no rod splitting is needed. Write the function so that a **negative** κ
   (thinner or missing segment) is at least detected and refused with a clear error (that case would need the
   parent rod split into pieces, like `woodpile_helpers._rod_pieces`, and is out of scope now).
3. Cylinder length: use the end-point distance; for axis-aligned rods `td.Cylinder(axis=0 or 1)` with
   `center = midpoint`, `length = |p2−p1|` is exact and avoids `td.Transformed`. Use the end-points **exactly as
   stored** (`x1..z2`): do not clip, extend or rebuild them. The transverse sim size is `params["box_size"]` — the
   user will regenerate larger structures later; the notebook must then work unchanged, picking up the new box,
   rod list and defects from the tables.
4. Return `td.Structure(geometry=td.GeometryGroup(geometries=cyls), medium=td.Medium(permittivity=params["permittivity"]))`
   and the count of cylinders (print it: 210 rods + N defects).
5. Sanity plot before anything else: `sim.plot_eps(z=z_layer)` for one x-layer and one y-layer and
   `sim.plot_eps(x=0)`; overlay the defect centres from the table; confirm the stacking (alternating orientation,
   `d/2` shift every second layer of the same orientation) by eye and compare a slice against `epsilon` from the
   same file *only as a check*, never as input.

## Frequency window — from the transmission data already measured

`H:\Codes\tidy3d\Notebooks\Woodpiles\data\20260918_transmission_data_woodpiles.h5` (`lambda` (300,),
`transmission_data/<file>.txt/transmission_exit`) was recorded over λ ∈ [1.5, 3.5] µm. Read it in the experiment
notebook and use it to set the source band. Measured `T_exit < 1e-2` windows:

* κ=0 crystal: λ ∈ [2.57, 3.20] µm (min T = 2e-4 at 2.82 µm) — the stacking-direction gap.
* κ=+1.2: [2.86, 3.15] µm; κ=+1.6: [2.99, 3.14] µm; κ=+2.8: no λ with T < 1e-2 (min 0.036 at 3.15 µm) —
  the gap fills in progressively with defect scattering.

Choose `lambdas` so the band covers the gap **and** both pass-band flanks with margin, e.g.
`lambdas = [3.5, 2.0] µm` (ν = d/λ ≈ 0.34–0.60). Justify the choice in a markdown cell using the numbers above.
Express results in both λ and the reduced frequency `ν = d/λ`.

## Simulation definition (what the experiment notebook must do)

Build `td.Simulation` by hand, as the reference does:

* **Sim size.** Everything comes from `params`: `t_slab_x, t_slab_y, t_slab_z = params["box_size"]`,
  `n_rod = sqrt(params["permittivity"])`, `d = params["d"]`. `Lz = t_slab_z + 2 × padding` with padding ≥ λ_max
  on each side (the reference uses `Lz = 45` for a 32 µm slab); transverse sim size = `t_slab_x, t_slab_y` plus the
  absorber layers, which tidy3d adds outside. Nothing structure-specific may be typed by hand — the same loop must
  run on any future `*_tables.h5` (larger box, other κ/ρ/seed) without edits. Note in markdown that the current
  12×12 µm box is a pilot and that spreading will be clipped by the absorbers at `±6 µm`.
* **Boundaries.** `td.Absorber(num_layers=130)` on all six sides (same count as the transmission run).
* **Source.** `td.GaussianBeam(source_time=td.GaussianPulse(freq0=f0, fwidth=0.5*(f_max-f_min), offset=10),
  size=(S, S, 0), center=(0, 0, z_src), direction='+', pol_angle=0, waist_radius=w0, waist_distance=...)`, with the
  source plane `z_src` about 1 µm below the entrance face (`-t_slab_z/2 - 1`) and the waist **on the entrance
  face**. **Sign convention (tidy3d 2.9.x, verified in memory): a positive `waist_distance` puts the waist BEHIND the
  source plane**, so to put the waist 1 µm *ahead* (at the face) use `waist_distance = -(z_face - z_src) = -1`.
  Verify this numerically (see Verification 1). Source patch `S = min(6 w0, t_slab_x)`, inside the non-absorbing region.
  Waist: parameter `w0` (default 1.5 µm ≈ λ/2 at the gap — the only hand-set physics parameter besides `lambdas`,
  `runtime_ps`, `min_steps_per_lambda`, `absorbers`); state the NA ≈ λ/(π w0) and the free-space Rayleigh range in a
  markdown cell.
* **Monitors.** Exit-face `td.FieldMonitor` at `z = +t_slab_z/2`, `size=(t_slab_x, t_slab_y, 0)`, `Ex,Ey,Ez`, `freqs=monitor_freqs`,
  `interval_space=(2,2,1)`; an identical **entrance-face** monitor at `z = -t_slab_z/2` so the incident spot is
  measured in the same run. `normalize_index=None`.
* **Grid.** `td.GridSpec.auto(min_steps_per_wvl=18, wavelength=lambdas[0], dl_min=dl, max_scale=1.2)`,
  `dl = (λ_min/18)/n_rod`. `subpixel=True`, `shutoff=1e-20`, `medium=td.Medium(permittivity=1)`.
* **Frequencies.** `nfreqs = spectral_sampling(T_ps=runtime_ps*1e12, band_width=0.01, a=d, lambdas=lambdas)`;
  `monitor_freqs = np.linspace(f_min, f_max, nfreqs)`. `runtime_ps`: the woodpile is thin (8.5 µm); start from
  `runtime_ps = 22e-12` and check in the `run=False` branch that the estimated cost and monitor data size are
  acceptable (field monitor size ∝ `nfreqs × Nx × Ny × 3 × 2` monitors; cap around a few GB). If needed, reduce
  `nfreqs` and accept a shorter reconstructable time window — print the resulting window.
* **Reference.** `sim0 = sim.copy(update={"structures": []})`, uploaded once with `task_name = sim_name + "_0"`,
  exactly as the transmission notebook's `ref = True` block. Pin the empty-box grid to the structure grid with
  `grid_spec=td.GridSpec.from_grid(sim.grid)` so the two runs share the same mesh (memory note).
* `project_name = "20260918_Beam_Spreading_Experiment_woodpiles"`, `sim_name = Path(filename).stem`.
* Polarisation `pol_angle=0` (E along x). Note in markdown which orientation (`0` = x, `1` = y) the top layer
  (max-z `rods/z`) has, i.e. whether E is parallel or perpendicular to the last rods.

Only verify in the `run=False` branch: `sim.plot(y=0)`, `sim.plot(z=+t_slab_z/2)`, `sim.plot_eps(z=z_layer)`
to confirm cylinders, source plane, waist position, monitor planes and absorbers; print `sim.num_cells`,
`sim.num_time_steps`, the grid `dl` inside the rods, the number of cylinders, and `web.estimate_cost`.
**Do not run anything on tidy3d — this costs money!** Leave `run = False`; the user flips it.

## Retrieval notebook

Mirror `20260918_Transmission_retrieval.ipynb` and the LSU field-retrieval notebook
`20260601_Beam_Spreading_freq/20251007_Retireve_Field_Data.ipynb` (which does
`structure_1 = AM.loadFromFile(key, file_path, get_ref=False)`;
`fd = structure_1.sim_data.monitor_data["monitorField_exit"]`; stores `fd.Ex.values, ..., fd.Ex.x.values, fd.Ex.y.values`
and `fd.Ex.f.values` per structure in a dict, iterating with `natsorted(os.listdir(...))`, and skips keys already
present so the notebook can be re-run incrementally). Here `get_ref=True` so `sim_data0` is downloaded too.
Name the monitors `"monitorField_exit"` and `"monitorField_entry"` in the experiment notebook so this lookup is
the same as in the LSU notebooks. Walk `{Path.cwd().parents[2]}/data/{project_name}`, use
`AM.loadFromFile(key, file_path, get_ref=True)` to get `sim_data` and `sim_data0`. For each structure store into a
dict and save with `AM.create_hdf5_from_dict(data, "./data/20260918_beam_spreading_data_woodpiles.h5")`:

* `x`, `y` coordinates of the exit monitor, `freqs`, `lambda`, `nu = d/lambda`.
* `I_exit[ν, x, y] = |Ex|²+|Ey|²+|Ez|²` (float32) for the structure, `I_exit_ref` for the empty box, `I_entry` for the
  entrance monitor. Keep the **complex fields** too (`Ex, Ey, Ez` on the exit face, complex64) — they are needed for
  the time-domain reconstruction later — but only if the file stays below ~2 GB; otherwise keep intensities plus the
  complex field for a decimated (`::2`) grid and say so.
* the structure `params` and the `defects` table (positions), copied from the structure `.h5`.

## Practical notes

* Interpreter: `C:\Users\HernandF\AppData\Local\Programs\Python\Python312\python.exe` (only env with tidy3d +
  ipywidgets + plotly). `API_TIDY3D_KEY` comes from `.env` via `load_dotenv()`.
* `sys.path.append(os.path.abspath(r'H:\codes\tidy3d')); import AutomationModule as AM` (needed only for
  `read_hdf5_as_dict`, `create_hdf5_from_dict`, `loadFromFile`).
* The notebooks live one level deeper (`Woodpiles/spreading/`), so data paths use `Path.cwd().parents[2]` for the
  tidy3d-root `data/` folder and `../Structures`, `../data` for the woodpile folder.
* Put any Claude-authored explanatory documents in a `Claude/` subfolder, not next to the notebooks.

## Verification (required)

After the experiment notebook is written, **spawn a separate agent** to check, with the Python312 interpreter and the
`run=False` branch (no cloud runs), and report numbers:

1. **Geometry.** Count cylinders = `len(rods) + len(defects)`. Sample the built `sim.epsilon(...)` (or
   `sim.plot_eps`) on the crystal-box region at the voxel centres of the `.h5` and compare with the stored `epsilon`:
   the mismatch fraction must be at the sub-voxel level (< ~2 % of rod voxels). Check that every defect centre from
   the table sits in high-permittivity material and that its measured cross-section is larger than a plain rod's by
   `(1+κ)` in area.
2. **Waist.** Build a structure-free sim with the same source, short `run_time`, and an entrance-face monitor;
   measure the 1/e² intensity radius there (must equal `w0` within the grid resolution). This fixes the sign of
   `waist_distance` — report the measured radius for both signs.
3. All six boundaries are `Absorber` with `num_layers=130`; source patch, the crystal box and both monitor
   planes lie inside the non-absorbing region.
4. `monitor_freqs` spacing satisfies `dν ≤ 1/runtime` (print both); exit monitor at `z = +box_z/2`, entrance at
   `-box_z/2` (compare against `params["box_size"]`).
5. `sim0` has identical grid (`sim0.grid == sim.grid` coordinate arrays), source, monitors and `run_time` to `sim` —
   only `structures` differs (compare the fields explicitly; do not rely on tidy3d `__eq__`).
6. Estimated cost and field-monitor storage size per task are printed.
