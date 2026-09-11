# Transmission through inverse (air-in-Si) TLPP diamond slabs

Tidy3D FDTD transmission spectra of **bidisperse TLPP diamond crystals** (tetrahedral
lobed patchy particles, Pine group / Akash Ghosh, NYU) realised as a **negative in
silicon**: every core and lobe of the particle configurations is an **air sphere
(ε = 1)** carved into a **Si background, n = 3.4 (ε = 11.56)**. The Si background fills
the whole simulation domain (there is no air cladding), so the spectra are the
transmission of the porous slab *embedded in bulk Si*, normalised to a homogeneous-Si
reference run.

## Contents of this folder

| file | what |
|---|---|
| `20260911_Spheres_Pine_Transmission.h5` | normalised transmission spectra, all structures × slab thicknesses (see *h5 layout*) |
| `mpb_geometry_run_XXX_….png` | one plot per structure: transmission (log) vs λ for each slab thickness |

## Input structures (`../Akash_data/*.csv`)

Five pressure-relaxed, orthorhombic, periodic diamond-lattice configurations of TLPPs.
Each particle = 1 core + 4 tetrahedrally arranged lobes (5 spheres); each CSV holds
**20 000 spheres = 4 000 particles** (`optical_particle_index` labels the particle,
`optical_material_label` ∈ {`host_core`, `host_lobe`, `defect_core`, `defect_lobe`}).

| file-name token | meaning |
|---|---|
| `box_10x10x10` | 10×10×10 tetragonal diamond cells (4 sites/cell). Box x,y ∥ ⟨110⟩, z ∥ [001], so **L_z = √2·L_x** and the cubic lattice constant is **a = L_z/10 ≈ 2.77–2.78 µm** |
| `rdef` | lobe radius (µm) of the *defect* (smaller) TLPPs: 0.45 or 0.48. Host lobes are 0.500 µm |
| `xdef` | fraction of lattice sites replaced by defect TLPPs: 0.10, 0.20, 0.30 |

Sphere radii (µm): host core 0.575, host lobe 0.500, defect core 1.15·rdef
(0.5175 / 0.552), defect lobe = rdef. Core–lobe centre distance ≈ 0.36 µm, so the five
spheres of a particle overlap heavily (nominal Σ4πr³/3 / V_box ≈ 1.04–1.06; the
Monte-Carlo air volume fraction is ≈ 0.57, i.e. **Si filling fraction ≈ 0.43** for run 000).

| structure | L_x = L_y (µm) | L_z (µm) | a (µm) |
|---|---|---|---|
| `mpb_geometry_run_000_box_10x10x10_rdef_0.45_xdef_0.10` | 19.681 | 27.833 | 2.783 |
| `mpb_geometry_run_001_box_10x10x10_rdef_0.45_xdef_0.20` | 19.706 | 27.868 | 2.787 |
| `mpb_geometry_run_004_box_10x10x10_rdef_0.48_xdef_0.10` | 19.676 | 27.826 | 2.783 |
| `mpb_geometry_run_006_box_10x10x10_rdef_0.48_xdef_0.20` | 19.647 | 27.785 | 2.778 |
| `mpb_geometry_run_007_box_10x10x10_rdef_0.48_xdef_0.30` | 19.578 | 27.688 | 2.769 |

## Simulation setup (Tidy3D, per structure and per slab thickness)

All lengths in **µm**.

- **Materials.** Domain medium `td.Medium(permittivity=3.4**2)` (Si). Every CSV sphere
  becomes a `td.Sphere` with `td.Medium(permittivity=1)` (air). All labels (host/defect,
  core/lobe) get the same air medium — the bidispersity enters only through geometry.
- **Slab thickness cuts.** From the full box (L_z) a slab of thickness
  `t = cut · L_z`, `cut ∈ {0.3, 0.5, 0.7, 0.8, 1.0}`, is kept: a sphere is included only
  if it lies **entirely** inside |z| < t/2 (`|z| + r < t/2`), otherwise it is dropped
  (no clipping, no partial spheres). The resulting `t` (e.g. 8.3500, 13.9167, …,
  27.8334 µm for run 000) is the thickness key used in the h5 file and in the task names
  (`…_size_z_<t>`).
- **Transverse periodicity.** Periodic BCs in x and y with period L_x, L_y; spheres are
  replicated with image copies at ±L_x, ±L_y so those crossing the box edge are wrapped
  correctly.
- **Domain.** Size (L_x, L_y, L_z + 12), centred at the origin. z boundaries:
  `td.Absorber(num_layers=200)` on both sides. The domain, source and monitor
  positions are set by the *full* box L_z for every cut, so thinner slabs simply have
  more bulk Si on either side.
- **Source.** `td.PlaneWave`, direction `+z`, default polarisation (E ∥ x), at
  z = −L_z/2 − 3. `GaussianPulse(freq0 = (f_min+f_max)/2, fwidth = 0.3·(f_max−f_min),
  offset = 10)` with f_min = c/7.5 µm, f_max = c/3.5 µm.
- **Monitors.** Two `td.FluxMonitor` planes spanning the full transverse domain:
  `entry` at z = −L_z/2 − 2 (between source and slab) and `exit` at z = +L_z/2 + 2
  (behind the slab). Both record 250 frequencies,
  `np.linspace(c/7.5 µm, c/3.5 µm, 250)` — **uniform in frequency**, λ from 7.5 → 3.5 µm.
- **Reference.** One extra task `Reference`: the same simulation with `structures=[]`,
  i.e. homogeneous Si. Used to normalise every spectrum (the reference was uploaded
  once with the first geometry; the domain size differs by < 0.2 µm across structures,
  which is immaterial for a homogeneous-medium reference).

## h5 layout — `20260911_Spheres_Pine_Transmission.h5`

Written with `AutomationModule.create_hdf5_from_dict`; no attributes are stored.

```
/lambdas                                              float64[250]   wavelength axis, µm
/transmission_data/<structure_name>/<thickness>/entry float32[250]
/transmission_data/<structure_name>/<thickness>/exit  float32[250]
```

| path | values |
|---|---|
| `lambdas` | wavelength (µm) shared by every spectrum: `td.C_0 / np.linspace(c/7.5, c/3.5, 250)`, i.e. **uniform in frequency, descending in λ** — `lambdas[0] = 7.5`, `lambdas[249] = 3.5` |
| `transmission_data/<structure_name>` | the five CSV stems listed above, e.g. `mpb_geometry_run_000_box_10x10x10_rdef_0.45_xdef_0.10` |
| `…/<structure_name>/<thickness>` | slab thickness **t in µm as a 4-decimal string**, e.g. `"8.3500"`, `"13.9167"`, `"19.4834"`, `"22.2667"`, `"27.8334"` (= 0.3, 0.5, 0.7, 0.8, 1.0 × L_z; values differ slightly per structure). h5py lists them lexicographically, so `"8.3500"` comes last — sort numerically (`natsorted` or `key=float`) |
| `…/<thickness>/exit` | **normalised transmission** T(λ) = flux(exit, slab) / flux(exit, reference), 250 points on the `lambdas` axis |
| `…/<thickness>/entry` | normalised flux through the entry plane, flux(entry, slab) / flux(entry, reference). Net Poynting flux in +z, so this is 1 − R (incident minus reflected); ≈ `exit` where the slab is transparent |

```python
import h5py
with h5py.File("20260911_Spheres_Pine_Transmission.h5") as f:
    lam = f["lambdas"][()]                       # µm, 7.5 -> 3.5
    for name, grp in f["transmission_data"].items():
        for t in sorted(grp, key=float):         # thickness keys as strings
            T = grp[t]["exit"][()]
```

## Reading the plots

`mpb_geometry_run_XXX_….png`: `exit` transmission (log scale) vs λ (µm) for each slab
thickness. The gap deepens roughly exponentially with t (≈ 5e-3 at 13.9 µm → ≈ 3e-6 at
27.8 µm for run 000), consistent with a photonic band gap of the inverse diamond lattice;
the pass-band level (0.3–0.5) reflects impedance mismatch / scattering rather than
absorption (the media are lossless).
