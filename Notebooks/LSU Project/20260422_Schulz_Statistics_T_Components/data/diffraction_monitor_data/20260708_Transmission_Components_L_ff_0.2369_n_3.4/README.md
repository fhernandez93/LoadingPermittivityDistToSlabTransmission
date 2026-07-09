# Transmission **components** vs. thickness — Schulz hole statistics

This folder contains the **polarization- and angle-resolved transmission components**
(`T_ballistic`, `T_co`, `T_cross`, `T_total`) of the 3D LSU disordered structures, for a
sweep of slab thicknesses `L`. Here the transmitted field
is decomposed into ballistic, co-polarized, and cross-polarized parts instead of a single
total transmission.

## What was simulated
In short:

- **Structure**: 3D LSU dielectric (`n = 3.4`), elliptical rods (`AR = 2.5`) perforated
  with elliptical holes (`AR = 2.5`); unperforated `ff = 0.237`, hole fraction ≈ 0.22.
- **Disorder**: hole radii drawn from a **Schulz (gamma) distribution** of shape `z`.
  **Only `z = 5.0` (broad / strongly polydisperse) is present in this folder** (`z_5.0/`).
- **Thickness sweep**: `L = 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15` (lattice units).
- **Averaging**: 5 independent disorder realizations per thickness, averaged
  (`..._average_L_*`).

The post-processing: a **diffraction
monitor** behind the slab gives the complex order amplitudes `amps`, from which the
components below are extracted.

## How the components are computed

With `amps` (sample) and `amps_ref` (empty reference) from the diffraction monitor, and
`Pinc = |amps_ref(0,0, p)|²`:

| Component | Definition | Meaning |
|---|---|---|
| `T_ballistic` | `|amps(0,0,p) / amps_ref(0,0,p)|²` | Coherent, un-scattered specular beam (`∝ exp(-L/ℓ)`). |
| `T_co`        | `Σ_orders |amps(p)|² / Pinc`       | Co-polarized transmission, all diffraction orders. |
| `T_cross`     | `Σ_orders |amps(s)|² / Pinc`       | Cross-polarized (depolarized) transmission — a multiple-scattering signature. |
| `T_total`     | `flux_exit(sample) / flux_exit(ref)` | Total transmittance (energy), cross-check `≈ T_co + T_cross`. |

## Files

```
z_5.0/
  transmission_n_3.4_ff_0.2369_z_5.0_Transmission_Components_average_L_{L}.txt
  transmission_n_3.4_ff_0.2369_z_5.0_Transmission_Components_average_L_{L}.png
```

- One `.txt` + `.png` pair per thickness `L` (the `L_{L}` suffix is the physical thickness
  `size × 11.44` in µm — e.g. `L_5.00`).
- `.png`: the four components vs. wavelength on a log-y axis (ballistic, co, cross plotted;
  total commented out in the notebook).

### Intended `.txt` format

Whitespace-delimited, one row per wavelength, header:

```
# lambda T_ballistic T_co T_cross T_total
```

## References

- Component definitions and full derivation: parent README
  [`20260422_Schulz_Statistics_T_Components/README.md`](../../../README.md).
- E. Akkermans and G. Montambaux, *Mesoscopic Physics of Electrons and Photons*,
  Cambridge Univ. Press (2007) — coherent (ballistic) vs. diffuse transmission and
  depolarization in multiple scattering.
- A. Ishimaru, *Wave Propagation and Scattering in Random Media*, Academic Press (1978).
- N. Cherroret, S. E. Skipetrov, B. A. van Tiggelen, *Phys. Rev. E* **82**, 056603 (2010).
- Tidy3D documentation — `DiffractionMonitor` (order amplitudes `amps`, `s`/`p`
  decomposition), https://docs.flexcompute.com/projects/tidy3d/.
