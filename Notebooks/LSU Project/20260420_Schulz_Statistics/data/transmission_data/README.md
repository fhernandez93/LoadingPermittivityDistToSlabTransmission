# Transmission data — Schulz hole-size statistics

This folder contains FDTD (Tidy3D) transmission spectra for **3D LSU disordered
structures** whose perforating holes are drawn from a **Schulz (gamma) size
distribution**. The goal is to compare how the degree of hole-size polydispersity changes
the transmission and the effective gap as a function of slab thickness.

## What was simulated

- **Structure**: 3D LSU dielectric structure (refractive index `n = 3.4`) built from
  **elliptical rods**, which are themselves perforated with **elliptical holes**. Both the
  rods and the holes have an aspect ratio `AR = 2.5`.
- **Fill fractions**:
  - Unperforated (rods only): `original_ff = 0.2172` (21.72%) — this is the `ff = 0.2172`
    in the file names.
  - Material after perforation: `material_ff ≈ 0.16857`.
  - Hole fraction (fraction of rod material removed): `holes_ff = 1 - material_ff/original_ff ≈ 0.22`.
- **Disorder model**: hole sizes follow a **Schulz distribution** parameterized by the
  shape factor `z`. Two cases are compared:
  - `z = 5`   → broad size distribution (strong polydispersity).
  - `z = 100` → narrow size distribution (nearly monodisperse).
- **Thickness sweep**: transmission was computed for a range of slab thicknesses
  `L = 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15` (in lattice / structural units).
- **Averaging**: for each thickness, **5 independent disorder realizations** were
  simulated and the resulting transmission spectra were **averaged**. All averaged files
  are tagged `average_5_samples`.
- **Smoothing**: the averaged spectra are smoothed along the wavelength axis with a
  Gaussian filter, `scipy.ndimage.gaussian_filter1d(transmission, sigma=6)`.

## Files

| File | Description |
|------|-------------|
| `transmission_n_3.4000_ff_0.2172_z_5.0_elliptical_average_5_samples.txt`   | Averaged transmission spectra, broad distribution (`z = 5`). |
| `transmission_n_3.4000_ff_0.2172_z_100.0_elliptical_average_5_samples.txt` | Averaged transmission spectra, narrow distribution (`z = 100`). |
| `transmission_n_3.4000_ff_0.2172_lambda_gap.txt`                           | Transmission at the gap wavelength vs. thickness, for both `z` values. |
| `*.png`                                                                    | Plots corresponding to the `.txt` files above. |

## File formats

### `..._z_{5,100}.0_elliptical_average_5_samples.txt`

Whitespace-delimited table. The header lists the slab thicknesses `L`:

```
# lambda  2.0  3.0  4.0  5.0  6.0  7.0  8.0  9.0  10.0  11.0  12.0  14.0  15.0
```

- **Column 1**: wavelength `lambda`.
- **Columns 2…N**: averaged transmission `T(lambda)` for each thickness in the header
  (one column per `L`).

Each row is one wavelength; values are the mean transmission over the 5 realizations.

### `..._lambda_gap.txt`

```
# L,T_gap_z5,T_gap_z100
```

- **Column 1**: slab thickness `L`.
- **Column 2**: transmission at the gap wavelength for `z = 5`.
- **Column 3**: transmission at the gap wavelength for `z = 100`.

This summarizes how the transmission at the photonic-gap wavelength decays with thickness
for the broad vs. narrow size distributions.

## Notes

- All transmission values are normalized (0–1).
- Data generated with Tidy3D FDTD via the project's automation module; see the analysis
  notebooks in `../../` (e.g. `20250618 Transmission Analysis.ipynb`).
