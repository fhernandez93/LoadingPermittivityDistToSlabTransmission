# Transmission components — n = 3.4, ff = 0.2369, collected through an NA = 0.25 aperture

Averaged transmission curves for the Schulz-polydisperse perfored-rod slabs
(refractive index n = 3.4, rod filling fraction ff = 0.2369, supercell period
11.44 µm, normal-incidence linearly polarized plane wave).

Each curve is an average over the **5 independent disorder realizations** that were
simulated for that configuration.

## What is in here

```
z_5.0/     Schulz shape parameter z = 5    (broad size distribution)
z_100.0/   Schulz shape parameter z = 100  (almost monodisperse)
```

Inside each folder there is one `.png` and one `.txt` per slab thickness,
`L = 2 … 15 µm` (the number in the file name is the thickness in µm).

The `.txt` files have 4 space-separated columns, 401 rows, wavelength running
**downwards** from 8 µm to 2 µm:

```
lambda   T_ballistic   T_co_aperture   T_cross_aperture
```

All transmissions are normalized to an empty-box reference run, so a bare beam
with no sample gives T = 1.

## How the components are pulled out of the diffraction monitor

The simulations use periodic boundaries in x and y, so the light leaving the slab
does not come out as a continuous angular pattern but as a **discrete set of
diffraction orders** `(m_x, m_y)`. Tidy3D's `DiffractionMonitor` gives, for every
order and every frequency:

* the complex amplitude `amps` in two polarizations, `p` (same polarization as the
  incident wave = **co**) and `s` (orthogonal = **cross**);
* the direction that order travels in, `theta`, `phi` (`monitor.angles`).

So the order index *is* the angle: order `(m_x, m_y)` leaves at
`sin(theta) ∝ λ · sqrt(m_x² + m_y²) / 11.44 µm`. That is what makes an
angle-resolved measurement possible.

### 1. Pick the orders that a lens would actually collect

A detector with numerical aperture NA only sees light within a cone around the
forward direction. Here we keep the orders with

```
sin(theta) <= 0.25          (theta <= 14.48 deg)
```

Evanescent orders come back with `theta = NaN` and drop out of the comparison by
themselves. The cone always contains the specular `(0,0)` order. Because the
period is large compared to the wavelength range, the cone holds between 1 and 9
orders, and for about 60% of the frequencies it holds *only* the specular order —
the first off-axis ring enters the cone at λ ≤ 2.86 µm.

### 2. Separate coherent from diffuse light before integrating

Each diffraction order is not a pencil of light: it stands for a **cell of
directions** of solid angle `λ²/(Lx·Ly·cos theta)`. At long wavelengths a single
cell can be several times wider than the whole NA = 0.25 cone, so just adding up
the orders inside the cone badly over-counts the collected power (checked against
a finely-sampled phase-screen model: +171% at λ = 8 µm). The two kinds of light
are therefore treated differently, using the 5 realizations:

* **Coherent part** — average the complex amplitudes over realizations,
  `a_avg = <a>`. What survives the average is the light that is the same in every
  sample; for a periodic structure it is a true delta function in angle, so
  summing `|a_avg|²` over the orders inside the cone is exact and needs no
  rescaling.
* **Diffuse (speckle) part** — the sample-to-sample scatter,
  `var = Σ|a_i − a_avg|² / (N−1)`. This is a smooth continuum in angle, so it is
  converted to a radiance (power per unit solid angle) using the cell solid angle,
  averaged over the orders in the cone, and then integrated over the true cone,
  which contributes `∫cos(theta) dΩ = π·NA² = 0.196 sr`.

The two are added:

```
T_co_aperture    = coherent(p) + diffuse(p)
T_cross_aperture = coherent(s) + diffuse(s)
```

`T_cross_aperture` being non-zero is a direct sign of multiple scattering: single
scattering at normal incidence mostly keeps the polarization, repeated scattering
depolarizes it.

### 3. Ballistic transmission

The un-scattered beam is the specular, co-polarized order of the
realization-averaged amplitude, normalized to the same order of the empty
reference:

```
T_ballistic = |<a(0,0,p)>|² / |a_ref(0,0,p)|²
```

Taking the average of the *amplitude* (not of the intensity) is what removes the
speckle and leaves only the coherent beam. Plotted against thickness it should
follow `exp(−L/ℓ)` and is the usual way to measure the scattering mean free path ℓ.

