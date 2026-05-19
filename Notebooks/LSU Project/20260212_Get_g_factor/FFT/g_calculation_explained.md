# How `old_rdg_spectrum` Calculates the Anisotropy Factor `g`

This note walks through the calculation of the scattering anisotropy factor
`g = ⟨cos θ⟩` performed inside `old_rdg_spectrum` in
[20250903_create_h5_from_ends.ipynb](20250903_create_h5_from_ends.ipynb).
The function operates in the **Rayleigh–Gans–Debye (RDG)** single-scattering
regime, where the differential scattering cross-section is driven entirely by
the structure factor `S(q)` of the permittivity contrast.

## Physical model

For an elastic single-scattering event with incident wavenumber

$$
k = \frac{2\pi\, n_{bg}}{\lambda}
$$

the magnitude of the scattering vector at polar angle θ is

$$
q(\theta) = 2k \sin(\theta/2).
$$

In the RDG approximation the differential cross section per unit solid angle is

$$
\frac{d\sigma}{d\Omega}(\theta,\lambda) \;\propto\; k^{4}\;\underbrace{\tfrac{1}{2}\!\left(1+\cos^{2}\theta\right)}_{\text{dipole factor}}\;S\!\big(q(\theta)\big),
$$

where `S(q)` is the (rotationally averaged) power spectrum of the permittivity
contrast `Δε(r) = ε(r) − ε_bg`. The asymmetry parameter is then the
intensity-weighted average of the cosine of the scattering angle:

$$
g \;=\; \langle\cos\theta\rangle
\;=\;\frac{\displaystyle\int_{0}^{\pi}\!\!\frac{d\sigma}{d\Omega}\,\cos\theta\,\sin\theta\,d\theta}
       {\displaystyle\int_{0}^{\pi}\!\!\frac{d\sigma}{d\Omega}\,\sin\theta\,d\theta}.
$$

A value of `g = 0` means isotropic scattering, `g → 1` is fully forward-peaked,
and `g → −1` is fully backward-peaked.

## Step-by-step in the code

### 1. Build the permittivity contrast and remove its mean
[20250903_create_h5_from_ends.ipynb](20250903_create_h5_from_ends.ipynb)
```python
deps = eps - eps_bg
deps -= deps.mean()        # kills the q = 0 (DC) spike
```
Removing the mean prevents an unphysical forward-scattering δ-peak at `q = 0`
that would otherwise dominate `g`.

### 2. 3D FFT → power spectrum `|F(q)|²`
```python
F3d = np.fft.fftshift(np.fft.fftn(deps))
P3d = np.abs(F3d) ** 2
```
This is the squared modulus of the Fourier transform of `Δε(r)`. By the
Wiener–Khinchin theorem this is proportional to the structure factor of the
contrast.

### 3. Radially average `|F(q)|²` to get `S(q)`
The reciprocal-space grid is built from `np.fft.fftfreq` (multiplied by `2π`
because we want angular wavenumbers, not cyclic frequency):
```python
qx3, qy3, qz3 = 2π * fftshift(fftfreq(N, d))   # per axis
qmag_3d = sqrt(qx² + qy² + qz²)
```
A 1D histogram over `|q|` bins accumulates the power, divided by the number of
voxels in each shell — this is the spherical average:
```python
sums,   _ = np.histogram(qmag_3d.ravel(), bins=q_bins, weights=P3d.ravel())
counts, _ = np.histogram(qmag_3d.ravel(), bins=q_bins)
S_q_raw   = sums / counts        # = ⟨|F(q)|²⟩_shell
```
`S_q_raw` is the discrete radial profile and is wrapped in an interpolator
`S_interp(q)` so it can be evaluated at any `q(θ)`.

### 4. Per-wavelength integration over θ
For each requested wavelength the code samples θ uniformly in [0, π] with
`n_theta = 500` points and evaluates the integrand:
```python
theta     = np.linspace(0, π, n_theta)
mu        = np.cos(theta)
sin_theta = np.sin(theta)

k         = 2π * n_bg / lam
q_scatter = 2 * k * np.sin(theta / 2)            # q(θ)
ang       = 0.5 * (1 + np.cos(theta)**2)         # dipole factor
Sq        = S_interp(q_scatter)                  # S(q(θ))
dsigma    = k**4 * ang * Sq                      # dσ/dΩ (up to a constant)
```
The total cross section comes from the φ-integrated denominator:
```python
sigma = 2π * np.trapz(dsigma * sin_theta, theta)
```

### 5. The actual `g` line
```python
g = 2π * np.trapz(dsigma * sin_theta * mu, theta) / sigma
```
This is a direct discretisation of

$$
g = \frac{\int (d\sigma/d\Omega)\,\cos\theta\,\sin\theta\,d\theta}{\int (d\sigma/d\Omega)\,\sin\theta\,d\theta},
$$

using the trapezoidal rule on the 500-point θ grid. The `2π` factors come from
the trivial φ-integration of an azimuthally symmetric integrand and cancel in
the ratio, but they are left in to keep `sigma` itself physically meaningful
(it is reused for `l_s = 1 / (ρ σ)` and `l* = l_s / (1 − g)`).

If `sigma` is non-positive (numerical edge case, e.g. zero contrast), the code
falls back to `g = NaN`, `l_s = l* = ∞`.

## Summary in one sentence

`g` is computed as the cosine-weighted average of the RDG differential cross
section `dσ/dΩ ∝ k⁴ · ½(1+cos²θ) · S(2k sin(θ/2))`, where `S(q)` is the
spherically averaged 3D FFT power spectrum of the mean-subtracted permittivity
contrast, integrated over θ ∈ [0, π] with the trapezoidal rule.

## Caveats baked into the function

- **DC removal is essential** — without `deps -= deps.mean()`, the `q = 0` bin
  dominates `S(q)` and forces `g → 1` regardless of structure.
- **RDG validity** — the function warns when `|m − 1| > 0.1` (Born
  approximation breaking down) or when the optical thickness phase
  `k · d · |m − 1| > 1` (multiple scattering likely). `g` returned outside
  these limits should be interpreted with care.
- **No polarisation handling** — the `½(1 + cos²θ)` factor assumes unpolarised
  illumination averaged over polarisation.
- **`S(q)` resolution** — `q_bins` has 1000 linearly spaced bins between `0`
  and `qmax`, and `S_interp` linearly extrapolates to `0` beyond the last
  bin (`fill_value=(S_plot[0], 0.0)`). For wavelengths so short that
  `2k > qmax`, the high-θ portion of `dσ/dΩ` is silently zeroed.
