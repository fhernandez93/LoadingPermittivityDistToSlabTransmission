# Methods — broadband-FFT transverse-spreading analysis of a focused beam in a 3D correlated-disordered slab

*Paper-style methods section for the validated pipeline (`bst_pipeline.py`). Literature is cited by
author/year; full references at the end. Numbers specific to this dataset are given so the section can
be adapted directly for a manuscript.*

## 1. Sample and simulation

An LSU (local self-uniform) correlated-disordered dielectric (refractive index n = 2.90, filling fraction
f = 0.2237) with characteristic length a = 2.5626 µm is arranged as a slab of thickness L = 14.3 µm
(= 5.58 a) along z. The transverse plane is built by tiling the 14.3³ µm disordered base cell 12 × 12
(total 171.6 × 171.6 µm). Finite-difference time-domain simulations use Tidy3D 2.9.1 with an auto-graded
mesh (≥ 15 points per in-medium wavelength at the pulse centre; the realized resolution falls to ≈ 9.4
points per wavelength at the top of the band), Courant 0.99, subpixel averaging, and a run time of
20 ps.

A beam is injected by a finite planar source (uniform 2.5 × 2.5 µm patch, polarization along x)
1 µm before the front face, driven by a Gaussian pulse spanning the analysis band. The transverse
electric field E = (Ex, Ey, Ez) is recorded at the exit face (z = +L/2) by a **frequency-domain**
`FieldMonitor` on N = 1700 uniformly spaced optical frequencies, i.e. the running discrete Fourier
transform accumulated over the whole record,

  Ê_c(x, y, f_k) = (Δt / √(2π)) Σ_n E_c(x, y, t_n) exp(+2πi f_k t_n),   c ∈ {x, y, z},

stored **without source normalization** (`normalize_index = None`), so each spectrum retains the
Gaussian source envelope |S(f)|. Two boundary configurations are simulated: **periodic** in x, y
(band ν ∈ [0.320, 1.025], df = 48.52 GHz) and **all-absorbing** (adiabatic absorbers on every face;
band ν ∈ [0.320, 1.250], df = 64.04 GHz). Reduced frequency is ν = a/λ = a f / c.

## 2. Time-domain reconstruction and narrow-band synthesis

Because of the +2πi f t sign convention, a causal time trace is recovered by the negative-kernel FFT
along the frequency axis,

  E_c(x, y, t_n) ∝ FFT_k[Ê_c(x, y, f_k)](n),   t_n = n Δt,  Δt = 1/(N df),  period T = 1/df,

verified both from the Tidy3D 2.9.1 source and empirically (≥ 99.4 % of reconstructed energy lies in
the first half-period; the inverse-kernel FFT time-mirrors the pulse). The optical carrier factors out
of |E|² and is dropped. For the absorbing run T = 15.62 ps < run time, so recorded signal beyond T
folds back (aliases) — reconstructed times > 15.62 ps are discarded and the fit window is kept well
inside T; for the periodic run T = 20.61 ps ≳ run time, so there is no time fold, but the incompletely
decayed field (5.5 × 10⁻⁵ at cut-off) leaves a sinc/leakage floor of ~10⁻⁵.

A per-frequency response is synthesized by multiplying the stored spectrum by a narrow spectral
window W(ν) centred on ν_c before the FFT:

  I_W(x, y, t) = Σ_c |FFT_k[W(f_k) Ê_c(x, y, f_k)]|².

We use a Gaussian window W(ν) = exp[−(ν − ν_c)²/2 σ_ν²] (no sidelobes, preferred for multi-decade
decay). The temporal pulse duration is t_p ≈ 1/(2π σ_ν c/a); the windows here (σ_ν = 0.008–0.03) give
t_p = 0.03–0.17 ps, satisfying the Cherroret condition t_p ≪ t_D ≈ 0.9 ps at ν = 0.9. Because the
window integrates ~σ_ν/δν_c independent spectral speckles (δν_c = field-correlation frequency), it
doubles as the ensemble surrogate for a single disorder realization; the residual single-shot
participation-ratio bias is (1 + 1/N_eff)⁻¹. Window shape (Gaussian/box/Tukey) and σ_ν are varied as
robustness checks. **No source-spectrum deconvolution is applied**: the stored data already carries
|S(f)| (dividing it out would double-deconvolve), and every estimator below is scale-invariant per
frequency bin, so the envelope cancels; absolute cross-ν comparisons are therefore not made on the raw
fields.

## 3. Observables

All spatial integrals use trapezoidal quadrature on the true (non-uniform) monitor coordinates. The
intensity is I = |Ex|² + |Ey|² + |Ez|² (the exit field is fully depolarized, Ex:Ey:Ez ≈ 0.34:0.34:0.31,
so a single component would discard ~2/3 of the intensity). Fields are rescaled by a global constant at
read to keep |E|² ~ O(1) and avoid float32 underflow of ∫I².

**Transverse mean-square width** (Cherroret et al. 2010, definition before their Eq. 3):

  σ²(t) = ∫ ρ² I(ρ, t) dA / ∫ I(ρ, t) dA,   ρ measured from the injection axis,

reported in units of a². A flat background (annulus mean over ρ ≥ ρ_bg) is removed exactly from the
zeroth and second moments; ρ_bg ∈ {25, 28, 30 a} is bracketed because σ² sits on a uniform-background
lever arm ⟨ρ²⟩_ap = 747 a². The diffusive prediction is σ²(t) = 4Dt (profile ∝ exp(−ρ²/4Dt)); the
localized prediction saturates at σ²_∞ = 2Lξ(1 − ξ/L) ≤ L²/2 = 15.57 a²; the mobility-edge scale is
σ²_∞ ≈ L² ≈ 31 a².

**Participation-ratio diameter** (Yamilov et al. 2023 use the identical metric with the same solver):

  d(t) = 2 √(A_eff/π),   A_eff = (∫ I dA)² / ∫ I² dA,

background-robust by construction and applied to the non-negative intensity. For the steady state this
reduces to the per-frequency d(ν) of the companion notebook.

**Hu ratio width** (Hu et al. 2008), background- and aperture-free:

  w_ρ²(t) = −ρ² / ln[ I(ρ, t) / I(0, t) ],   evaluated at ρ = 5, 10, 15, 20 a,

with I(ρ) azimuthally averaged. Diffusion gives w_ρ² = 4Dt independent of ρ; localization gives a
saturating w_ρ² that **increases with ρ** (non-Gaussian profile) — the discriminator against any
uniform, absorption-like time-dependence.

**Diagnostics/artifact clocks:** total power P(t) = ∫I dA (peak = arrival, post-peak minimum = the
reconstruction watershed), and the edge-power fraction ∫_{ρ>28a} I dA / P(t) (rises when the halo
reaches the transverse boundary → wrap in the periodic run, edge loss in the absorbing run).

## 4. Fitting discipline and the tiling limit

D(ν) is obtained from a weighted linear fit σ²(τ) = σ₀² + 4Dτ over a **valid window** τ ∈ [τ_peak, τ_hi],
where τ_hi is the earliest of: (i) σ² exceeding **25 a²** — the width of one structural tile, beyond
which transverse transport is Bloch-periodic and σ² is no longer a transport observable (the 12× tiling
gives transverse period L = 5.58 a, so σ ≲ 1 tile ⇔ σ² ≲ 30 a²); (ii) edge-power fraction exceeding
1 % (aperture/wrap onset); (iii) half-way to the power watershed (reconstruction floor). Reported R² is
flagged as optimistic because zero-padding correlates adjacent FFT frames. A saturation value is
inverted for ξ via σ²_∞ = 2Lξ(1 − ξ/L) **only if it lies below L²/2**; a plateau above that ceiling is
reported as mobility-edge/aperture scale, not a localization length. Because the sample is a single
realization at a single L in a transversely tiled geometry, thickness scaling — the decisive
localization test (Goïcoechea et al. 2026; Hu et al. 2008) — is unavailable, and no localization length
is claimed from these data.

## 5. Boundary-condition cross-validation

The absorbing run is the primary source for σ²(t)/D (no transverse wrap); the periodic run is used only
for an early-time cross-check where both must agree (below the wrap onset). The two runs share no
frequency bins (compared at nearest bins or interpolated) and differ in absolute exit power by a smooth
factor 1.4–3.3 (unresolved; candidate causes: mesh-dependent transmission, lateral absorber loss), so
only **normalized profile shapes** are compared — these agree to normalized cross-correlation 0.88–0.97
across the band, confirming a common underlying speckle field.

## 6. Validation battery (acceptance tests)

Run on both datasets, 7 windows each (`bst_pipeline`): (1) **Energy conservation** — the discrete
Parseval identity Σ_n P(t_n) = N Σ_k |W Ê_k|² holds to 4 decimals (ratio 1.0000) in all 14 windows.
(2) **Cross-notebook consistency** — the FFT single-bin steady state gives d(ν = 0.9) = 11.888,
identical to the frequency-domain notebook's d(0.9) ≈ 11.9 (same data, same estimator); the
band-window time-integrated width is 14.0 (the incoherent, speckle-mitigated width), with map
normalized cross-correlation 0.86. (3) **Robustness** — D(ν) and σ²(t) are reported with their spread
over window shape, σ_ν, ρ_bg, and fit range; on the corrected rising-edge window (test 4) D(0.9) is
stable (raw 7.4–9.3 across fit-range sweeps, all R² > 0.99; 7.95/6.36 across σ_ν = 0.03/0.015). (4)
**Physical anchor — achieved.** The 4Dt growth lives on the **pre-peak rising edge**: early-arriving
light has spread less and later light more, so σ²(t) grows *through* the power peak. Fitting σ²(t) over
the arrival-to-peak window at ν = 0.9 (σ² grows 5.6 → 20 a², below the one-tile cap, edge-power < 1 %)
gives **D = 5.8 (background-subtracted, R² = 0.999) – 8.0 (raw, R² = 0.992) a²/ps**, agreeing between
the two datasets to ~4 % and with independent estimators (background-free Hu ratio width D ≈ 7.5–9,
PR-diameter d²/8 D ≈ 4.5) — reproducing the prior **D ≈ 6 a²/ps (39 m²/s)** anchor, and D(0.65) ≈
11–17 a²/ps rising above the gap as expected. (An earlier draft mistakenly fitted the *post-peak
decay*, starting the window at the power peak, and wrongly reported D ≈ 0; an adversarial verification
pass caught the bug — the fit window now starts at pulse arrival.) So the **diffusive regime and its
D(ν) are accessible and measured**; the low-ν/in-gap windows (the beam arrives already filling the
aperture / transport is non-diffusive) and the localized regime (tiling cap, thin single L) are not.
(5) **Reproducibility** — both cleaned notebooks and `bst_pipeline` run end-to-end from the raw HDF5
files with no manual steps or hidden state.

## 7. Interpretation guardrails

A stationary width dip **cannot** distinguish localization from absorption/attenuation: Cherroret et al.
(2010) show σ²_CW ≃ 2LL_a under absorption has the identical form as the localized 2Lξ, and inside a
photonic pseudogap Bragg/evanescent decay plays the role of L_a. Localization attribution therefore
rests on the **time-resolved** σ²(t) (absorption-immune at fixed t) plus complementary signatures
(ρ-dependent Hu width, non-exponential time-of-flight with D(t) ∝ 1/t, spectral Thouless structure) and
thickness scaling. The history of 3D optical-localization claims (Sperling et al. 2013, retracted-grade
re-examination by Sperling et al. 2016 and Skipetrov & Page 2016 attributing the saturation to
fluorescence) shows that a saturating transverse width is, on its own, weak evidence; the analogous
FDTD artifacts controlled here are transverse wrap-around, aperture truncation, background-subtraction
residuals, and — decisively — the transverse tiling. For an *uncorrelated* dielectric at this index
Yamilov et al. (2023, same solver) find no localization; only the correlated/pseudogap route
(Haberko et al. 2020, n = 3.6) remains open, and at n = 2.90, fixed single L, single realization, and a
tiled transverse geometry, the defensible statement is "consistent with transverse confinement near the
gap; localization not proven."

## References

- Cherroret, Skipetrov & van Tiggelen, *Phys. Rev. E* **82**, 056603 (2010).
- Hu, Strybulevych, Page, Skipetrov & van Tiggelen, *Nat. Phys.* **4**, 945 (2008).
- Cobus, Hildebrand, Skipetrov, van Tiggelen & Page, *Phys. Rev. B* **98**, 214201 (2018).
- Sperling, Bührer, Aegerter & Maret, *Nat. Photonics* **7**, 48 (2013); Sperling et al., *New J. Phys.*
  **18**, 013039 (2016); Skipetrov & Page, *New J. Phys.* **18**, 021001 (2016); Scheffold & Wiersma,
  *Nat. Photonics* **7**, 934 (2013).
- Yamilov, Skipetrov, Hughes, Minkov, Yu & Cao, *Nat. Phys.* **19**, 1308 (2023).
- Goïcoechea, Yamilov, Ferise, Skipetrov, Cao & Davy, arXiv:2606.04897 (2026).
- Froufe-Pérez, Engel, Sáenz & Scheffold, *PNAS* **114**, 9570 (2017); Haberko, Froufe-Pérez &
  Scheffold, *Nat. Commun.* **11**, 4867 (2020); Vynck et al., *Rev. Mod. Phys.* **95**, 045003 (2023).
- van Albada, van Tiggelen, Lagendijk & Tip, *Phys. Rev. Lett.* **66**, 3132 (1991).
