# Transmission Components from a Diffraction Monitor

This folder computes the **polarization- and angle-resolved transmission components**
of light through disordered (Schulz-polydisperse) dielectric slabs simulated with
Tidy3D FDTD. The quantities of interest are the **co-polarized**, **cross-polarized**,
and **ballistic (coherent)** transmission, plus the **total** transmittance.

The slabs are collections of high-index "pen-like" rods (n = 3.4) with hollow cores
whose radii are drawn from a **Schulz (gamma) size distribution**, illuminated at
normal incidence by a linearly polarized plane wave. The goal is to separate the
transmitted field into the part that stays in the incident polarization, the part that
is depolarized by multiple scattering, and the un-scattered ballistic beam.

---

## 1. Simulation setup

The structures and the FDTD runs are produced by the `AutomationModule`
(`loadStructures.py`). The relevant ingredients are:

- **Source** — a linearly polarized `td.PlaneWave` at normal incidence,
  propagating along `+z`, with `pol_angle = 0` (E-field along the reference axis).
- **Boundaries** — periodic in the transverse plane, absorbers along the
  propagation direction, so the transmitted field is naturally decomposed into a
  discrete set of **diffraction orders** `(orders_x, orders_y)`.
- **Diffraction monitor** (`name='diffraction'`) — a `td.DiffractionMonitor` placed
  *behind* the slab (at `flux_monitor_position`). It returns the complex far-field
  amplitudes `amps`, indexed by diffraction order `(orders_x, orders_y)`,
  polarization (`'s'`, `'p'`), and frequency `f`.
- **Flux monitors** — `flux1` (exit / transmission side, "to the right") and
  `flux2` (entrance side, "to the left"), used for the total transmitted flux.
- **Reference run** — an identical simulation with the structure removed
  (`structures = []`). All quantities below are normalized to this empty reference so
  that the bare incident beam corresponds to `T = 1`.

### Polarization convention

For the diffraction amplitudes, `'p'` is taken as the **co-polarized** channel (same
polarization as the incident plane wave) and `'s'` as the **cross-polarized** channel.
The `'s'`/`'p'` labels are the standard Tidy3D decomposition of each diffraction order
relative to its plane of incidence; at the specular `(0, 0)` order they coincide with
the incident linear polarization.

---

## 2. Definitions of the transmission components

All amplitudes come from the diffraction monitor. Let

- `amps` — complex diffraction amplitudes of the **sample** run,
- `amps_ref` — complex diffraction amplitudes of the **empty reference** run,
- `Pinc = |amps_ref(0,0, p)|²` — incident power carried by the specular,
  co-polarized order of the reference (the bare beam).

The components computed in
[`20260422_get_transmission_from_diffraction.ipynb`](20260422_get_transmission_from_diffraction.ipynb)
are:

### Co-polarized transmission — `T_co`
Power transmitted into **all** diffraction orders while keeping the incident
polarization, normalized to the incident beam:

```
T_co = Σ_{mx,my} |amps(mx, my, p)|² / Pinc
```

This includes both the coherent specular beam and the diffuse light that has not been
depolarized.

### Cross-polarized transmission — `T_cross`
Power transmitted into **all** orders in the orthogonal polarization:

```
T_cross = Σ_{mx,my} |amps(mx, my, s)|² / Pinc
```

A non-zero `T_cross` is a direct signature of **multiple scattering**: single
scattering at normal incidence largely preserves the polarization, whereas repeated
scattering progressively depolarizes the field. The ratio `T_cross / T_co` is therefore
a convenient proxy for the scattering strength / transport regime.

### Ballistic (coherent) transmission — `T_ballistic`
The un-scattered, forward beam. It is the ratio of the **specular, zeroth-order,
co-polarized** amplitude with and without the sample:

```
T_ballistic = |amps(0,0, p) / amps_ref(0,0, p)|²
```

Because it is built from the *coherent specular amplitude* (not the order-summed
intensity), it isolates the field that traverses the slab without being scattered out
of the incident mode. For a slab of thickness `L` this is expected to follow the
Beer–Lambert / coherent-beam law `T_ballistic ∝ exp(-L/ℓ)`, where `ℓ` is the
scattering mean free path — making `T_ballistic(L)` the standard way to *measure* `ℓ`.

### Total transmittance — `T_total`
Obtained directly from the exit flux monitor, normalized to the reference flux:

```
T_total = flux_exit(sample) / flux_exit(reference)
```

This is the energy-conserving total transmission and serves as a cross-check:
`T_total ≈ T_co + T_cross`.

---

## 3. Notebook workflow

| Notebook | Role |
|---|---|
| [`20250903_create_h5_from_ends.ipynb`](20250903_create_h5_from_ends.ipynb) | Builds the permittivity grids: hollow "pen-like" rods whose hole radii are sampled from a **Schulz distribution** (`sample_schulz`), saved as `.h5` structures with a fixed filling fraction. |
| [`20251001_numerical_experiment.ipynb`](20251001_numerical_experiment.ipynb) | Runs the FDTD simulations for a series of slab thicknesses (`cuts`), uploading each structure plus one empty **reference** to Tidy3D. Stores `flux1`, `flux2`, and the `diffraction` monitor. |
| [`20260422_get_transmission_from_diffraction.ipynb`](20260422_get_transmission_from_diffraction.ipynb) | **Core post-processing.** Loads each run, reads `amps` from the diffraction monitor, computes `T_co`, `T_cross`, `T_ballistic`, `T_total` per (n, ff, z, size, sample), and collects them into a nested dict saved as an `.h5`. |
| [`20260423_Plot_Transmission_Components.ipynb`](20260423_Plot_Transmission_Components.ipynb) | Loads the aggregated `.h5` and plots the transmission components vs. wavelength / thickness. |

> The `*_do_not_use.ipynb` notebooks are deprecated earlier attempts and are **not**
> part of this pipeline.

---

## 4. References

1. E. Akkermans and G. Montambaux, *Mesoscopic Physics of Electrons and Photons*,
   Cambridge University Press (2007). — Coherent vs. diffuse transmission, the
   ballistic (coherent) beam `exp(-L/ℓ)`, and depolarization in multiple scattering.
2. A. Ishimaru, *Wave Propagation and Scattering in Random Media*, Academic Press
   (1978). — Decomposition of the field into coherent (ballistic) and incoherent
   (diffuse) components.
3. N. Cherroret, S. E. Skipetrov, and B. A. van Tiggelen,
   "Transverse confinement of waves in three-dimensional random media,"
   *Phys. Rev. E* **82**, 056603 (2010). — Transport/localization framework for the
   disordered slabs studied here.
4. P. Sheng, *Introduction to Wave Scattering, Localization and Mesoscopic Phenomena*,
   2nd ed., Springer (2006). — Mean free path, transport regimes, and depolarization.
5. G. V. Schulz, "Über die Kinetik der Kettenpolymerisationen,"
   *Z. Phys. Chem.* **B43**, 25 (1939); see also M. Kerker, *The Scattering of Light
   and Other Electromagnetic Radiation*, Academic Press (1969). — The Schulz (gamma)
   size distribution used for the polydisperse hole radii.
6. Flexcompute, **Tidy3D documentation — `DiffractionMonitor`**,
   https://docs.flexcompute.com/projects/tidy3d/ — definition of the diffraction-order
   amplitudes `amps` and the `s`/`p` polarization decomposition.
