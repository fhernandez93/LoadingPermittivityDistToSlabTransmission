# 3D Cavity Numerical Experiment — 2D SHU (Antenna Scan)

`20250708 Numerical Experiment Cavity 3D.ipynb`

FDTD (Tidy3D) reproduction of the microwave cavity experiment: a quasi‑2D
stealthy hyperuniform (SHU) slab of high‑index dielectric rods, driven by a
point dipole ("emitter" antenna) that is scanned across the sample, sandwiched
between two metal plates that form a 3D parallel‑plate cavity. For each emitter
position a matching **empty‑cavity reference** run is also launched so that
transmission/flux can be normalized in post‑processing.

---

## Units

Tidy3D is **scale‑invariant**: Maxwell's equations are solved in one generic
length unit and all frequencies are computed from `td.C_0` in that same unit.
In this notebook the length unit is interpreted physically as **centimeters
(cm)**, which places the experiment in the **microwave (GHz)** regime.

| Quantity | Unit in the notebook | Notes |
|---|---|---|
| Length (all `center`, `size`, `radius`, `Lx/Ly/Lz`, `a`, `air_gap`) | cm | interpreted as cm to match the microwave experiment |
| Time (`runtime_ps`, `t_stop`) | seconds internally; **ps** at the API (`runtime_ps = 10e-12`) | 10 ps physical run time |
| Frequency (`freq0`, `monitor_freqs`) | Hz (Tidy3D internal) | convert to GHz for physical interpretation |
| Permittivity (`medium`, ε = 45) | dimensionless | high‑index dielectric rods (microwave ceramic) |
| Reduced / normalized frequency `u = a/λ` | dimensionless | the physically meaningful spectral coordinate |

The sanity‑check cell

```python
a = 1.5873
lambdas = a/np.array([0.12, 0.5])   # wavelengths in cm
3e10/lambdas * 1e-9                  # -> [2.268, 9.450]  GHz
```

uses c = 3×10¹⁰ cm/s to confirm that the reduced‑frequency window
`u = a/λ ∈ [0.12, 0.5]` corresponds to **2.27 – 9.45 GHz** in the lab.

---

## The scaling factor `a`

`a = 1.5873` cm is the **characteristic length** of the structure (the SHU
length scale / rod‑spacing unit). Everything about the disordered sample is
expressed in **dimensionless units of `a`** and multiplied by `a` to obtain
physical cm:

- **Rod centers** are read from the structure file in units of `a`, then scaled:
  ```python
  x, y = x*a, y*a
  ```
- **Rod radius**: `radius = 0.189*a` (≈ 0.30 cm) — the SHU rods.
- **Sample box** `Lx, Ly` read from the file are also in units of `a`.
- **Spectral window** is defined in reduced units first
  (`u = [0.12, 0.5]`), then turned into wavelengths with `lambdas = a/u`, from
  which Tidy3D derives `freq_range = C_0/lambdas` inside
  `AM.loadAndRunStructure`.

This is the standard "work in `a/λ`, scale by `a`" convention: the FDTD result
is identical for any choice of `a`; `a` only fixes the mapping to physical cm /
GHz.

### The `slicing` parameter and slab size

`slices = [0.07]` selects the **fraction of the full SHU point set** used as the
active sample, i.e. it sets the slab length `L` along `y`. Rods are kept only
where `|y| ≤ 100/2 · a · slicing` and `|x| ≤ slab_size_x/2 · a`, which gives the
**physical slab size** (rods, ε = 45):

| Axis | Half‑extent | Full size | Set by |
|---|---|---|---|
| **x** (width) | `slab_size_x/2 · a = 27.5 · a ≈ 43.65 cm` | **≈ 87.3 cm** | `slab_size_x = 55` |
| **y** (length `L`) | `100/2 · a · slicing = 50 · a · 0.07 ≈ 5.56 cm` | **≈ 11.1 cm** | `slicing = 0.07` |
| **z** (thickness) | — | **0.5 cm** | rod `length = 0.5` |

So the disordered slab is roughly **87.3 cm (x) × 11.1 cm (y) × 0.5 cm (z)**,
sitting inside the `110 × 70 × 0.515 cm` cavity. `L ≈ 11.1 cm` is what appears
in the output folder name `..._L=11`.

- Flux monitors sit just outside the slab (in `y`) at
  `y = ±((Lx·slicing)/2 + 3) ≈ ±6.85 cm`.
- Changing `slicing` rescales `L` (y); changing `slab_size_x` rescales the width
  (x).

---

## Geometry / cavity setup

| Element | Value | Meaning |
|---|---|---|
| Domain `Lx, Ly, Lz` | `110, 70, 0.5 + air_gap` cm | `air_gap = 0.015` cm → `Lz ≈ 0.515` cm |
| x, y boundaries | `td.Absorber(num_layers=160)` | open (absorbing) sides |
| z boundaries | `td.PECBoundary()` (top & bottom) | metal plates → the **3D cavity** |
| Rods | ε = 45, `radius = 0.189·a`, `length = 0.5` cm | quasi‑2D dielectric SHU slab |
| Emitter | `td.PointDipole`, `Ez` polarization | Gaussian pulse at `freq0`, `fwidth = freqw` |
| Emitter antenna | PEC `Cylinder`, r = 0.03, L = 0.2 cm | mimics the physical probe antenna |
| Emitter scan | `x_positions = arange(-5.5, 6.5, 1)` | 12 emitter positions across the sample |

The parallel‑plate PEC boundaries in `z` are what make this a **cavity** rather
than a free‑space slab: the field is confined between two metal plates spaced
`Lz ≈ 0.5 cm` apart.

---

## Source / resolution settings (from `loadAndRunStructure`)

- `freq0 = mean(freq_range)`, `freqw = width·Δf` with `width = 0.4`.
- `min_steps_per_lambda = 28`, uniform grid
  `dl = (λ_min / 28) / √ε ≈ 0.017 cm (≈ 170 µm)`
  (`GridSpec.uniform(dl=structure_1.dl)`).
- `runtime_ps = 10e-12` (10 ps), field‑decay shutoff `1e‑20`.
- `freqs = 700` monitor frequency points across the band.

---

## Monitors

| Name | Type | Purpose |
|---|---|---|
| `freq_monitorFieldOut` | `FieldMonitor` (z‑plane at `Lz/2−0.2`) | frequency‑domain `Ex, Ey, Ez` field map |
| `flux1` / `flux2` | `FluxMonitor` at `y = ±((Lx·slicing)/2+3)` | transmitted flux on each side of the slab |
| `time_monitorFieldOut` | `FieldTimeMonitor` | (defined; not attached in the final run) |

---

## Runs, references and outputs

For every (emitter x‑position) the notebook uploads **two tasks**:

1. **Reference** `{sim_name}_ref` — same simulation with `structures=[]`
   (empty cavity), used to normalize transmission.
2. **Sample** `{sim_name}` — the full SHU + antenna cavity.

Both `task_id`s are written (reference first, sample second, newline‑separated)
to:

```
H:\phd stuff\tidy3d\data\{project_name}\{structure_stem}\{structure_stem}_L={L}\{sim_name}.txt
```

with

```python
project_name = "20250729 0.15mm Gap Cavity 3D"   # from air_gap*10
sim_name     = "chi_0.37_N_10000_posics_emmiter_{x:.4f} - Sample_{k}"
```

Existing `.txt` files are skipped, so re‑running the notebook resumes rather
than resubmitting.

---

## Key parameters to change

| Variable | Effect |
|---|---|
| `a` | physical scale (cm / GHz mapping); FDTD result unchanged |
| `lambdas` via `a/[0.12,0.5]` | spectral window in reduced units `a/λ` |
| `medium = td.Medium(permittivity=45)` | rod dielectric constant |
| `slices` | slab length `L` / fraction of SHU points used |
| `air_gap` | plate spacing (`Lz`) and `project_name` |
| `x_positions` | emitter scan positions |
| `min_steps_per_lambda`, `runtime_ps`, `freqs` | resolution / run length / spectral sampling |
| `run` | `True` uploads & starts jobs; `False` stops before submission |
