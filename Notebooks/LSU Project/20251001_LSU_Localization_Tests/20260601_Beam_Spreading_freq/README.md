# Beam spreading vs frequency — fully random LSU slab 250×250×32 µm, n = 3.3

Steady-state (CW, frequency-domain) transverse beam-diameter analysis d(ν) at the exit
face of a **fully random, aperiodic** LSU-generated healed cylinder network — the run that
replaces the old 12×-tiled SHU experiment. Analysis notebooks:
**`20260806_Beam_Diameter_d_nu.ipynb`** (CW d(ν)) and **`20260808_Beam_Spreading_time.ipynb`**
(time-domain σ²(t), d(t) — §8; all physics prose lives here).
Deep-dive working documents and the adversarial-verification scripts are under `Claude/`.

**Headline (2026-08-07).** The photonic gap of this random structure sits at
ν ≈ [0.389, 0.433] (−20 dB, contiguous), matching both the design window [0.388, 0.435]
and the MPB periodic-crystal reference [0.388, 0.434]. Two genuine transverse-narrowing
dips flank the deep gap (lower: d ≈ 4.8 a at ν = 0.3894; upper: d ≈ 5.6 a at ν = 0.4370;
factor ≈ 5× vs the mid-band 27.6 a), each carried by **individual quasi-mode transmission
spots** — not a narrowed on-axis beam. Inside the deep gap (S ≈ −50 dB) the exit intensity
is floor/rim-dominated and d(ν) is *not* a beam width. kℓ* reaches ≈ 1.07 at the gap
centre (vacuum k, scale-corrected — Ioffe-Regel-*scale* scattering, but **no** kℓ* < 1
window; an earlier kℓ*=0.85 claim fell to a scale error, C11). Mid-band d(ν) is
consistent with the Cherroret diffusive baseline within the (large, one-sided) speckle and
z₀ systematics. **No localization length is claimable from this CW data at a single L** —
the honest label for any dip-derived length is an attenuation/confinement scale, and even
that swings ~15× with estimator choice at the dip bins (see Claims table, C4).

---

## 1. The numerical experiment

| item | value |
|---|---|
| structure | LSU-generated *healed* cylinder network, point file `20260804_slab_200x200x32_N854932_lsu_generated_healed.txt` (in `../Structures/`), coordinates ÷0.8 → 250×250 µm transverse; cylinder radius 0.42 µm; ε = 3.3² = 10.89 (n = 3.3); ff ≈ 0.2237 |
| slab | cylinders trimmed to \|z_center\| ≤ 16.8 µm → thickness **L = 32 µm = 12.49 a**; transverse 250 µm = 97.6 a, **aperiodic** (no tiling — verified: density autocorrelation shows only the ~2 µm structural peak; the old "Bloch-periodic beyond one tile" caveat is retired) |
| units | a = 2.562629142772549 µm is the length/frequency unit convention only (ν = a/λ = a f/c); this structure has no lattice constant |
| source | **5×5 µm uniform plane-wave patch** (hard edges) at z = −19 µm, 3 µm before the input face; Gaussian pulse spanning λ = 8→4 µm |
| boundaries | **absorbers on all six sides** (200 layers) — no wrap-around; edge-loss bias possible (quantified: negligible outside the gap, C9) |
| run | Tidy3D FDTD, run_time 40 ps, shutoff 1e−20, `normalize_index=None` (raw DFT — spectra carry the source envelope; every per-bin scale-invariant estimator cancels it) |
| monitor | exit-face `FieldMonitor` at z = +16 µm, `interval_space=(4,4,4)`, fields Ex, Ey, Ez, 1500 frequencies |
| launch script | `20251001_numerical_experiment_using_td_cylinders.py` (folder root) |

### HDF5 layout — `data/slab_250x250x32/LSU_20260804_slab_250x250x32_n_3p3__lsu_generated_healed_lambda_4_8.h5` (4.4 GB)

| dataset (group `"3.30"`) | shape / dtype | notes |
|---|---|---|
| `Ex`, `Ey`, `Ez` | (363, 363, 1, 1500) complex64 | raw running-DFT exit fields, \|E\| ~ 1e−16 |
| `x`, `y` | (363,) float64 | **non-uniform** grid, ±125.17 µm (±48.85 a), mean dx ≈ 0.69 µm |
| `z` | (1,) = [16.0] | exit face |
| `f` | (1500,) float64 | **ascending**, ν = a f/c ∈ [0.3203, 0.6407], δν = 2.137×10⁻⁴ (≙ 40 ps window) |

### File inventory (what produced what)

| file | role |
|---|---|
| `20251001_numerical_experiment_using_td_cylinders.py` | built + launched the cloud run (project `20260805_Beam_Spreading_250_250_32`) |
| `20251007_Retireve_Field_Data.ipynb` | downloaded the cloud result into `data/slab_250x250x32/…h5` |
| `20260806_Beam_Diameter_d_nu.ipynb` | **the analysis** (this experiment): d(ν), S(ν), baseline, gap/dips, interactive beam map |
| `bst_pipeline.py` | validated streaming/estimator primitives from the 2026-07 audit (docstrings = conventions; its `L_SLAB_UM=14.3` and tiling notes refer to the OLD 12× dataset) |
| `20260808_Beam_Spreading_time.ipynb` | **time-domain first look** (this run): σ²(t), d(t), P(t) + exit-face movies (§8). Replaces `20260602_IPR_Calculation_FFT.ipynb` (old 12× data; deleted 2026-08-08 — committed copy in git history, old-data backup in `backups_20260716/`) |
| `movies/beam_spreading_nu*.gif` | §8 exit-face movies: ν = 0.350, 0.389, 0.437, 0.550 (log₁₀ I, fixed norm, 300 frames / 40 ps) |
| `data/slab_250x250x32/…backdround_effective_n_1.34.h5` | **effective-medium cladding run** (project `20260810_…_effective_index_1.34_freq`, 2026-08-10): same slab, z = ±16…±22.5 µm filled with n_eff = 1.34 (Bruggeman, ff = 0.217), run_time 45 ps, 1688 freqs, 483² grid (`interval_space=(3,3,3)`). **Not a beam-spreading measurement — see §10** |
| `Claude/audit_workspace/effective_cladding/` | 2026-08-11 air-vs-cladding investigation: streaming comparison (`compare_runs.py`), angular-spectrum flux reconstruction (`flux_correction.py`), air-cone filter (`flux_aircone.py`), result npz + summary scripts |
| `data/L_1_12x/` | old 12×-tiled SHU data (n = 2.90) |
| `Claude/STEADY_STATE_IPR_REPORT.md`, `Claude/METHODS.md`, `Claude/AUDIT_REPORT.md` | old-run deep dives (still the reference for estimator derivations) |
| `Claude/audit_workspace/random_slab/` | this run's streaming pass, analysis scripts, claims file, and adversarial-verification scripts (`verify/`) |
| `backups_20260716/`, `Claude/audit_workspace/steady/…BACKUP…` | backups of the retired `20251008_IPR_Calculation.ipynb` (deleted 2026-08-07; also in git history) |

Auxiliary inputs (both **outside** this folder):
ℓs(ν): `../20251002_Ls_test/data/ls_values/20251031_ls_values_n_3p3.h5` (`"0.2237"`, 400 pts;
its ν axis is reconstructed as ν = 0.8·a·`raw_freqs`/c from
`../20251029_T(L)/data/20251030_ff_2237_circular_rods_n_3p3_2.h5`, ascending 0.256→1.025;
axis association independently confirmed: the ℓs minimum lands inside the gap, and a
Beer-Lambert fit of the co-stored T(L) data co-locates, corr 0.88).
g(ν): `../data/g_values/n_3.3_ff_0.2237_g_data.h5` — treated as faithful (generated outside
this workspace for a statistically similar structure; note its attrs say spheroids AR 2.5,
n_rod = 2.93). **Layout differs from the old n=2.90 g file**: `nu` is co-stored
**ascending** (ν·λ_um/(0.8a) = 1 exactly), groups by point count; we use
`file4_N100000/g_avg` (all `reliable` in-band). g is *negative* around the gap (−0.42 at
ν≈0.41: backscattering-dominated), so ℓ* = ℓs/(1−g) < ℓs there.
**SCALE (caught by the adversarial pass):** both files live at the *native* structure scale
a′ = 0.8a = 2.050 µm (g-file attr; ℓs-dip wavelength ratio 1.23 ≈ 1.25). At matched
(dimensionless) ν, this ×1.25-scaled slab has ℓ_phys = **1.25 ×** the file value — the
notebook applies the factor. Consequences: mid-band ℓ* ≈ 6.3 µm (L/ℓ* ≈ 5), gap-centre
ℓ* ≈ 1.1 µm, and min kℓ* = 1.07 (not 0.85).

## 2. Estimator and error model

**Primary observable: participation-ratio (PR) diameter**
d = 2√(A_eff/π), A_eff = (∫I dA)²/∫I² dA, I = |Ex|²+|Ey|²+|Ez|², product-Simpson on the
true non-uniform grid. Scale-invariant per bin (source envelope and any global scale
cancel) and 10–20× less background/aperture-sensitive than ⟨ρ²⟩ (old-run audit; re-confirmed
here: full-aperture ⟨ρ²⟩/(d²/8) ≈ 2 even mid-band — the second moment is halo-dominated and
is kept only as a diagnostic). Unit test: PR diameter of a Gaussian = 4σ exactly (asserted
in the notebook).

**Speckle error model** (theory verified against Goodman-type statistics + Monte-Carlo,
wave A; numbers re-audited by wave B2): the exit field is one fully-developed multi-
component speckle realization. The measured contrast is **estimator-dependent** —
spectral-detrend gives C ≈ 0.57 (window-dependent artifact), a window-free spatial
estimator gives C ≈ 0.65 (M_eff ≈ 2.4) — so the single-shot PR diameter is biased **LOW**
by √(1+C²) = **×1.15–1.20** (one-sided; this form depends only on the measured contrast,
so it survives near-field non-Gamma statistics). Per-bin speckle scatter ±0.76 a (2.8%
mid-band); d-residual correlation length ≈ 5.8 bins (intensity spectral correlation FWHM
≈ 4.6 bins). **Frequency-boxcar d_avg:** at w = 181 it reaches ≈ raw×1.21 ≈ the converged
ensemble value in *flat* bands (the earlier "exceeds the speckle ceiling ⇒ transport
blending" inference died with the corrected C) — but it blends transport wherever d(ν)
trends and is corrupted in/near the gap, so the estimator of record remains **raw per-bin
d(ν) ± speckle σ**, ensemble = raw×(1.15–1.20). Additional per-bin uncertainty from grid
sampling: the 2×-decimation test shows **no systematic bias** but a ±1.6% (median) / 4%
(p90) resampling scatter — and **+11…15% at the two dip bins** (quasi-mode spots have
sub-grid structure), so dip d values carry ~10–15% error.

## 3. Diffusive baseline and its validity

Cherroret, Skipetrov & van Tiggelen, PRE **82**, 056603 (2010) (all formula attributions
re-verified against the paper, wave A): T(q) = sinh(qℓ*)/sinh(qL) (Eq. 3, z₀=0),
σ²_dif = (2/3)L²[1−(ℓ*/L)²] (Eq. 4, D-independent — the baseline is FLAT in ν by design),
ℓ→ℓ* substitution endorsed for correlated disorder incl. the source depth (their
discussion), valid for L ≫ ℓ*. We push T(ρ) through the **same PR estimator** as the data
(never compare a PR diameter to a √⟨ρ²⟩): d_dif ≈ 25.9 a at L = 32 µm, moving < 8% under
ℓ*×(0.5–2).

**z₀ (internal-reflection) systematic — one-sided UP** (verified against
Zhu-Pine-Weitz/Haskell benchmark tables, wave A): z₀ = (2/3)ℓ*(1+R_eff)/(1−R_eff) with
R_eff = 0.37 (Maxwell-Garnett n_eff = 1.27, **preferred**) to 0.72 (volume-ε 1.79) ⇒
z₀/ℓ* = 1.45–4.0. Pushed through the *same PR pipeline* (extrapolated-boundary kernel
T(q,z₀) = sinh(q(ℓ*+z₀))cosh(qz₀)/sinh(q(L+2z₀))), that multiplies the baseline diameter
by **×1.37 (MG) to ×1.66 (vol-ε)** — note the closed-form √(σ²-ratio) values (×1.45–2.0)
overestimate. Comparison (mid-band): measured raw d = 27.6 a, ensemble ≈ 32 a, vs
baseline 25.7 a (z₀=0), 35 a (MG z₀), 42 a (vol-ε z₀). The ensemble value sits within
~10% of the **MG** baseline, while the **vol-ε baseline is excluded (−24%)** — so the
comparison is not merely "unquantifiable": it is *consistent with diffusion* and
**constrains z₀/ℓ* to ≈ 1–1.5** (an MG-like effective index). Residual caveat: measured
d(ν) rises ≈ +6% across 0.50–0.62 (5.7σ with correlated-bin errors) — a trend the flat
baseline does not explain (band-edge transport or ℓ*(ν) structure; unresolved).
L/ℓ* ≈ 5–7 mid-band (marginally diffusive; the paper wants L ≫ ℓ*).

### 3.1 The baseline in plain terms — d_dif, z₀, and "MG preferred"

All three objects in the summary figure's baseline (red line, green band, green dashed
line) are versions of the same thing — the **diffusive expectation** for the exit-spot
diameter — differing only in how the slab boundary is treated.

**The red line, d_dif (z₀ = 0).** If light crossing the slab simply diffuses, a focused
input beam exits as a diffuse spot with the universal Cherroret profile
T(q) = sinh(qℓ*)/sinh(qL). Its width is set almost entirely by the *thickness*:
σ²_dif = (2/3)L²[1−(ℓ*/L)²], independent of the diffusion constant and only weakly
dependent on ℓ* — which is why the red line is flat (≈ 25.7 a). It is computed through
the *same* PR estimator as the data, so the comparison is apples-to-apples. Measured d
below it ⇒ sub-diffusive narrowing; that is the entire purpose of the baseline.

**What z₀ is, and why the systematic is one-sided.** The z₀ = 0 formula assumes the
diffuse intensity vanishes exactly at the physical slab faces. A real high-index slab
internally reflects diffuse light at its boundary (Fresnel + total internal reflection),
so the intensity extrapolates to zero a distance z₀ = (2/3)ℓ*(1+R_eff)/(1−R_eff)
*beyond* the surface. The diffusion problem then effectively sees a slab of thickness
L + 2z₀ — thicker slab ⇒ wider diffuse spot — so the true diffusive baseline always lies
**above** the red line. That is why it is drawn as a one-sided band, never a symmetric
error bar.

**Why the band has two edges, and what "MG preferred" means.** R_eff depends on the
effective index n_eff the diffuse light sees at the boundary, and for a composite (22%
rods of n = 3.3 in air) there is no unique n_eff. Two standard prescriptions bracket it:
Maxwell-Garnett mixing (n_eff ≈ 1.27 → R_eff = 0.37 → z₀ = 1.45 ℓ*, the dashed green
line) and volume-averaged permittivity (n_eff ≈ 1.79 → R_eff = 0.72 → z₀ = 4 ℓ*, the top
of the band). MG is "preferred" twice over: *a priori*, volume-averaging ε overweights
the dilute high-index phase, and disordered-photonics practice uses MG-like values for
boundary reflectivity; *a posteriori* — the adversarial result — **the data itself
decides**: the speckle-corrected mid-band d ≈ 32 a lands within ~10% of the MG-z₀
baseline (34.5 a) but 24% below the volume-ε one (42 a), so if mid-band transport is
diffusive the volume-ε boundary condition is excluded and the data pins z₀/ℓ* ≈ 1–1.5.

**How to read the figure.** The honest diffusive corridor runs from the red line (hard
lower edge) up to roughly the green dashed line (best estimate). A claimed sub-diffusive
narrowing must fall below that corridor — after lifting the raw points by the ×1.15–1.20
speckle correction (single-shot speckle biases the PR diameter low). The flank dips at
4.8/5.6 a clear that bar by a factor ~5, which is why they survive; the ~7% offset
between raw mid-band data and the red line means nothing by itself, being smaller than
the z₀ + speckle corridor width.

## 4. Interpretation rules — what CW d(ν) can and cannot prove

1. **A CW width dip is not localization.** Under strong absorption (or in-gap
   Bragg/evanescent attenuation) the same stationary theory gives σ² ≃ 2L·L_a — "absorption
   plays exactly the same role as localization" (Cherroret, verbatim, re-verified). A
   single-L CW dip yields an **attenuation/confinement scale L_att**, never ξ.
2. **The critical regime alone dips too.** At the mobility edge σ² = (3/8)L² — a 25% width
   reduction with zero attenuation. Shallow dips near kℓ* ≈ 1 may be criticality.
3. **The CW localized form is the FULL Cherroret Eq. 6** — the compact
   σ² = 2Lξ(1−ξ/L) is the *pulsed-saturation* conjecture, not the CW result (a previous
   session conflated them; corrected here). Validity ℓ* ≪ ξ ≪ L.
4. **PR→σ² conversion is profile-dependent**: d²/8 = ⟨ρ²⟩ exactly for a Gaussian, ×0.83 for
   the diffusive sinh-profile, ×0.67 for an exponential profile — and it **collapses
   entirely on quasi-mode spot profiles** (dip bins: 14–21× below the halo-restricted
   second moment). Convert only with the profile stated, or don't convert.
5. **Deep-gap bins are not beam widths.** Where S ≲ −35 dB the exit map is floor/rim-
   dominated (rim ρ>45a carries up to ~29% of power) — d(ν) is masked there.
6. Localization attribution needs the time-domain σ²(t) (absorption-immune), thickness
   scaling, and/or ensemble statistics — none available from this single-L CW dataset.

## 5. Results (this run)

- **Gap:** contiguous −20 dB dip [0.389, 0.433]; floor ≈ −50 dB (reference-dependent ±5 dB)
  spanning ν ≈ 0.40–0.43. Matches design [0.388, 0.435] and MPB crystal [0.388, 0.434].
  Edges quoted to ~±0.003 (envelope-reference systematic; wave B1).
- **Flank dips:** lower d_min = 4.8 a at ν = 0.3894 (S = −29 dB, rim 0.31%); upper
  d_min = 5.6 a at ν = 0.4370 (S = −10 dB, rim 0.03%); both ±10–15% (sampling). Confirmed
  genuine (not floor, not aperture, speckle-fluke probability ~3% each, and the upper dip
  is supported by 4 bins / two distinct modes). The maps show **single quasi-mode
  transmission spots** (upper one ~8 a off-axis) carrying 30–45% of all exit power within
  2 a of one peak — transverse confinement of individual modes, not a narrowed beam.
- **kℓ*** (scale-corrected, vacuum k): minimum ≈ **1.07** at ν = 0.403 (≈ 1.35 with the
  effective-medium k) — Ioffe-Regel-*scale* scattering at the gap centre, but **no kℓ* < 1
  window**, and the flank dips do not sit at kℓ*=1 crossings (upper dip: kℓ* ≈ 1.37).
  (Context: near photonic band edges kℓ* ≈ 1 is *not required* for localization — the John
  mechanism operates on band-edge proximity; Froufe-Pérez 2017, Haberko 2020.)
- **Mid-band (0.50–0.62):** d = 27.6 a median, consistent with diffusion, constraining
  z₀/ℓ* ≈ 1–1.5 (§3); an unexplained +6% rising trend across the band is flagged.
- **Below-gap (ν ≈ 0.34–0.38):** raw d ≈ 22 a at ν = 0.36 looks 15% below the z₀=0
  baseline, but this **dissolves on inspection** (C6): the below-gap speckle contrast is
  higher (C ≈ 0.70 ⇒ bias ×1.22), and the corrected value sits *above* the z₀=0 baseline.
  The genuine relative narrowing vs mid-band closes quantitatively as band-dependent
  speckle bias + z₀·ℓ*(ν) differential + **measured gap-precursor attenuation**
  (S(ν) is already −1…−5 dB there ⇒ L_att ≈ 29→11 µm over 0.360→0.375 ⇒ predicted 3–15%
  narrowing; total closes within ~1%). A quasi-ballistic on-axis spike is excluded
  (core power *below* the diffusive reference; excising ρ<3a moves d by ~1%).
- **σ²(t) headroom for the time-domain follow-up:** with the honest aperture, a localized
  saturation at σ² = 63–77 a² (ξ = 5–9 a, full Eq. 6) is now *inside* the measurable range
  (the old 12×-tiled cap σ² ≲ 30 a² is retired) — the decisive test is time-domain, not CW.

## 6. Known systematics

| systematic | size | direction | mitigation |
|---|---|---|---|
| single-shot speckle bias on d | ×1.15–1.20 mid-band; ×1.22 below gap (**band-dependent**) | one-sided (d biased low) | stated correction; only multiple realizations remove it |
| per-bin speckle scatter | ±2.8% mid-band | random | report ±1σ band |
| grid sampling (PR denominator) | ±1.6% median, 4% p90; +11–15% at dip bins | random (no net bias) | folded into per-bin error; dip values flagged |
| z₀/internal reflection on baseline | ×1.37–1.66 on d_dif (through-PR kernel) | one-sided (baseline low at z₀=0) | band shown in figure; MG value preferred; data itself constrains z₀/ℓ* ≈ 1–1.5 |
| ℓ*(ν) uncertainty | < 8% on d_dif level; moves validity mask | both | ℓ*×(0.5–2) scan |
| edge loss (all-absorber BCs) | < 1% on d outside mask; no boundary pile-up | — | rim-fraction validity mask (< 2%) |
| deep-gap floor | d meaningless where S ≲ −35 dB | — | masked, plotted separately |
| source sidelobes (5×5 µm hard patch) | ≤ 1% on d at ν=0.33, ≤ 0.5% elsewhere, none at the flanks (C12: sidelobes evanescent) | — | verified null; broad injection NA remains an initial condition |
| ℓ*/g scale (files at a′ = 0.8a) | ×1.25 on all ℓ values | one-sided | corrected in notebook (C11) |
| exit-face monitor (interface waves, Goïcoechea 2026) | ≤6% on d for the air run (flux cross-check, §10) | — | angular-spectrum S_z reconstruction agrees with \|E\|² widths; interior monitor still preferred for a re-run |

## 7. Claims table — adversarial verdicts (2026-08-07)

Claims stated in `Claude/audit_workspace/random_slab/CLAIMS_FOR_VERIFICATION.md`;
verification scripts in `…/random_slab/verify/`. Wave A = literature/primary-source
verifiers; waves B = independent data refuters. Time-domain rows T1–T7 (2026-08-08):
claims stated in §8, verification scripts in `…/random_slab_td/verify/` (three lenses:
reconstruction-correctness, artifact (wrap/edge/floor), does-it-reproduce).

| # | claim | verdict | note |
|---|---|---|---|
| C1 | gap at [0.384, 0.436] (−20 dB), min −54 dB at 0.4156 | **PARTIALLY → corrected** | gap real & MPB-consistent, but the −20 dB "lower edge" 0.3838 was a detached spike — contiguous core **[0.389, 0.433]**; the "minimum" is a flat −50±5 dB floor over 0.40–0.43; edges good to ~±0.003 only |
| C2 | deep-gap d is not a beam width | **CONFIRMED** | rim45 median 19% (max 29%), aperture sensitivity up to 22%, contiguous failing block [0.392, 0.430]; core spots still bright per-pixel but the *integral* is rim-driven |
| C3 | two genuine flank dips (4.8 a / 5.6 a) | **CONFIRMED, reinterpreted** | not floor (halo 35–100× above floor), not aperture (≤0.6%), speckle-fluke p ≈ 0.03 each; but they are **quasi-mode spots** (upper ~8 a off-axis), and dip d carries +11–15% sampling error |
| C4 | L_att ≈ 0.12/0.16 a from d²/8, sub-ℓ*, CW cannot give ξ | **PARTIALLY** | arithmetic right, but the PR→σ² conversion collapses on spot profiles: halo-restricted ⟨ρ²⟩ gives L_att ≈ 1.6/3.3 a (>ℓ*). Inferred length swings ~15–20× with estimator ⇒ **only the dip factor (~5×) is robust; guardrail reinforced** |
| C5 | mid-band consistent with diffusive baseline within systematics | **PARTIALLY → sharpened** | baseline level & ℓ*-insensitivity reproduce; but through-PR z₀ multipliers are ×1.26–1.66 (closed-form √σ² overestimates), and the comparison is *informative*: ensemble d within ~10% of the MG-z₀ baseline, vol-ε z₀ **excluded** (−24%) ⇒ z₀/ℓ* ≈ 1–1.5. Unexplained +6% slope (5.7σ) across 0.50–0.62 flagged |
| C6 | below-gap d < baseline at ν≈0.36 — cause? | **REFUTED as framed / resolved** | the "−15%" compared raw single-shot d to an ensemble baseline with the wrong band's bias: low-band C = 0.704 ⇒ ×1.22, corrected d is *above* the z₀=0 baseline. The relative dip vs mid-band closes (≈1%) as bias difference + z₀ differential + measured gap-precursor attenuation (L_att ≈ 17–29 µm from S(ν)). Ballistic/coherent-spike mechanism killed |
| C7 | speckle stats (C=0.569, M≈3, ×1.15) & raw-d estimator choice | **PARTIALLY** | theory chain + MC confirmed (wave A), but C is estimator-dependent: spectral-detrend 0.57 is a window artifact, spatial C ≈ 0.65 (M≈2.4) ⇒ bias ×1.15–1.20; the "d_avg exceeds the speckle ceiling ⇒ transport blend" inference REFUTED (w=181 ≈ ensemble value in flat bands); raw-d-±σ estimator choice stands |
| C8 | sampling converged except narrow flagged bins | **REFUTED as characterized** | decimation differences are speckle-resampling noise (dec2/dec3 sign-uncorrelated), spread over the whole band: med 1.6%, p90 4% everywhere above the gap; no systematic bias; convergence not *demonstrable* from one speckle field |
| C9 | edge loss negligible outside gap | **CONFIRMED** | rim medians reproduce; no absorber pile-up in marg_x (decaying tails, decay length 8–14 µm); missing-power ≤ 0.35% worst case |
| C10 | design claim: geometry captures ξ near the gap | **CONFIRMED as corrected** | geometry adequate: σ_loc = 8.0–8.8 a ≤ 0.18× aperture, aperiodic, no tiling cap. But single-L CW **cannot resolve ξ in the 5–9 a target**: σ²_loc(full Eq. 6) = 63.1/68.7/76.8 a² at ξ=5/6.25/9 — all above σ²_ME = 58.5 a² (indistinguishable from critical); Δd(ξ:5→9) = 2.1 a ≈ 2.7 speckle σ, swamped by systematics. Unambiguous CW dip needs ξ ≲ 3.2 a; resolvability lost for ξ ≳ 4.2 a. (Full Eq. 6 is strictly monotonic in ξ — the truncated 2Lξ(1−ξ/L) non-monotonicity in the claims file was an artifact.) The decisive test is time-domain σ²(t) saturation, for which the geometry now has full headroom |
| C11 | kℓ* < 1 in [0.390, 0.431], min 0.85 | **REFUTED** | scale error: ls/g files live at a′ = 0.8a, so physical ℓ* = 1.25× file values ⇒ min kℓ* = 1.07 (vacuum k), 1.34–1.36 (n_eff k) — no sub-unity window; dips not at kℓ*=1 crossings; axis association itself verified correct (ℓs min in-gap, T(L) Beer-Lambert corr 0.88) |
| C12 | source sidelobes contaminate low-ν wings | **REFUTED (null)** | sinc sidelobes are *evanescent* below ν≈0.51 (5 µm patch < λ); no k-space cross at any ν; exit light depolarized (33:32:34) and k-isotropic; coherent contamination of d ≤1% at ν=0.33, ≤0.5% above, zero flank leakage. Below-gap d≈33 a is transport with broad injection NA, not sidelobe diffraction |
| T1 | causal FFT reconstruction correct (fft not ifft; Parseval; pre-arrival = window-kernel tail) | **PARTIALLY** | convention CONFIRMED: `fft` is causal (arrival 0.27–0.43 ps > earliest physical front 0.15–0.20 ps; `ifft` time-mirrors exactly, P_ifft[n]·N² = P_fft[N−n] to 5.6e−15); Parseval 1.8e−15 in float64 (3.5e−8 in the float32 pipeline) but is convention-blind. Sub-claim (c) REFUTED: pre-arrival signal is the wrap-around fold of the undecayed tail (P(0)/P(T−dt) = 0.80–1.13), *not* the window kernel — no acausality; kernel-scaling probe confirms (P(0) −29→−59→−61 dB as FWHM 0.01→0.02→0.04) |
| T2 | fold-back negligible on trusted segments; boundary >−40 dB only for edge/in-gap windows | **PARTIALLY → sharpened** | dichotomy real but the >−40 dB list has NINE windows incl. 0.370 (−25.9 dB) and 0.450 (−32.4 dB). Conclusion survives via the *decay* argument: end tails still decay (τ ≈ 6–7 ps), so the folded copy is ≤0.3% of P on trusted segments; mid-band boundary −50…−60 dB, Δσ² ≤ 0.8 a² worst case |
| T3 | mid-band σ² grows ~20→550 a² by ~11 ps, no flattening; growth ordering gap-adjacent < mid-band < below-gap | **REPRODUCES** | independent implementation (trapezoid, own FFT/blocking) matches to ≤0.4%; cutoff values within 1% across Gaussian/decimated variants; ordering never inverts (Tukey moves gap-adjacent σ²(5 ps) by 7–8%, groups separated ×2); "20 a² at arrival" is a Gaussian-window statement (box windows ring at t < 1 ps) |
| T4 | 0.370/0.450 grow 2–3× slower and genuinely bend over (not edge clock, not floor admixture) | **CONFIRMED** | slopes [2–8 ps] 18.2/15.9 vs mid-band 31.8–50.7 a²/ps; 0.450 hard-flattens 297–305 a² over 25–31.8 ps, 0.370 bends but creeps 285→363 a² (18→31.4 ps); rim never crosses 1% (edge deficit at σ²=350 is 0.55%); the "floor" is the still-decaying tail itself (late-frame σ²/d ≈ 354/32 and 295/28 — NOT the deep-gap pattern), mixture correction a no-op; bend intact under P>10–30× floor cuts. **Guardrail: the 300–370 a² plateau is ~5× above the ξ=5–9 a localization target (63–77 a², §5) — do not sell it as the localization plateau.** 0.437-wing leakage into 0.450 (2–12% of P) pulls σ² *down* 1–5% (removing it raises the plateau) |
| T5 | 0.437: persistently narrow, non-spreading profile (d ≈ 17–20 a, rim ≤0.1%), unlike 0.450 | **REPRODUCES; blending REFUTED** | plateau and 0.437-vs-0.450 contrast survive Tukey, 2× decimation, and a *halved* bandwidth — which makes the profile narrower and flatter (d med 18.2→14.1 a), the opposite of pass-band blending; σ²(5 ps) is the one window-sensitive number (74–86 a², 15%); rim 0.06–0.10% in all variants |
| T6 | deep-gap windows measure the floor, not transport | **PARTIALLY** | 0.41 CONFIRMED (P max at frame 0, trusted range 0.29 ps; σ² ~108 a², d ~13 a = floor properties). 0.43 mechanism WRONG: it is a genuine long-lived gap-edge quasi-mode (P peaks at 5.3 ps, τ ≈ 11–12 ps, σ² 67→150 a², twin of 0.437); its 0.45 ps cutoff is a 0.3–1 ps rim *transient* tripping the first-crossing clock — "not a beam-transport measurement" stands for both |
| T7 | narrow at arrival (d ≈ 9–12 a), σ² grows through the power peak | **PARTIALLY** | growth-through-peak CONFIRMED in all 12 pass-band windows (slope at t_pk +22…+53 a²/ps, rim ≤9e−4 there); arrival d = 8.80–12.24 a for 0.35–0.61; at 0.63 the 1e−3 gate misfires on the boundary floor (re-gated at 1e−2: d = 11.8 a at 0.45 ps — narrow arrival still true, protocol fails) |

Effective-cladding rows E1–E4 (2026-08-11): claims about the n = 1.34-cladding run (§10),
verified by direct computation (`Claude/audit_workspace/effective_cladding/`) plus a
config/git audit and two adversarial physics passes.

| # | claim | verdict | note |
|---|---|---|---|
| E1 | cladding-run σ²/d excess is an analysis bug or axis mismatch | **REFUTED** | notebook derives every axis-dependent quantity from the loaded h5 (git+cell audit); the only optical change between runs is cube1/cube2 permittivity 1 → n_eff² (plus 40→45 ps, (4,4,4)→(3,3,3)); the data itself differs: σ² ×2.5, d ×1.5–1.6, **frequency-flat** |
| E2 | excess = \|E\|²-at-a-plane overweighting grazing light (1/cosθ); flux fixes it | **REFUTED as dominant** | angular-spectrum S_z reconstruction (validated: air flux ≈ air \|E\|² widths ≤6%, zero flux beyond k₀ in the air run as TIR demands) removes only ~8% (CW mid-band σ² 491→454 a²; air ≈197) |
| E3 | excess flux is genuine wide-footprint transported power, delivered by cladding channels | **CONFIRMED** | 47% of transmitted flux at k∥ > k₀ (air-forbidden band; Lambertian matched-face prediction 1−1/n² = 44.4%); halo has the same angular mix as the core; air-cone filter (k∥ < k₀) still leaves σ² ≈ 470 a² — surface re-scattering relabels angles. Channels: full-cone main-lobe injection of the 5 µm patch radiating inside n = 1.34 (ν-flat — the sidelobe-threshold story would step at ν ≈ 0.38 and is excluded), matched-face reflectance recycling, interface skim/re-scatter; ~25% of exit power in the wide footprint |
| E4 | cladding run salvageable by post-processing (angular filter / time gate / deconvolution) | **REFUTED** | angular filter dead (E3); time gate dead (cladding transit 87 a/ps ≫ diffusive spread, continuously replenished); deconvolution needs an absent front-face monitor and the contamination is coherent (Goïcoechea: not removable by subtraction). Only fix = re-run (§10) |

**Corrections to carried-over context adopted this session** (wave A, primary sources):
(i) CW localized width = full Cherroret Eq. 6; σ²=2Lξ(1−ξ/L) is the pulsed conjecture.
(ii) z₀/ℓ* = 1.45–4.0 (ZPW/Haskell-verified) — the old report's Egan-Hilgeman 1.5–3.3 was
approximately right (its R values were the flux-average 2C1), an intermediate computation
this session (R_eff=0.84) was wrong (convention mixing).
(iii) d²/8 → ⟨ρ²⟩ is profile-dependent (×0.67–1.0) and invalid on quasi-mode spots.
(iv) The old n=2.90 g-file ν-descending convention does **not** apply to the new n=3.3 g
file (ν ascending, co-stored).

## 8. Time-domain first look (2026-08-08) — observe only, no fits

Notebook **`20260808_Beam_Spreading_time.ipynb`** (lean, 7 code cells). Narrow-band
**causal** reconstruction from the same h5: Gaussian spectral windows of **FWHM 0.02 in ν**
(σ_ν ≈ 8.5×10⁻³, pulse ≈ 0.16 ps) on the 0.02 grid over [0.33, 0.63] plus the two CW flank
dips 0.389/0.437; E(t) = `scipy.fft.fft` over the ascending f axis (Parseval 3.5×10⁻⁸);
per-frame estimators as in §2 (product-Simpson, σ² about the injection axis, PR d(t));
one streaming pass, no movie materialized. **No D fits, no ξ, no localization claims.**

**Validity rules** (printed by the notebook): curves are trusted only for
t_arr < t < t_valid = min(t_floor, t_rim), where t_arr = first P > 10⁻³ P_pk,
t_floor = P sinks into 3× the late-time floor, t_rim = rim power (ρ > 45 a) exceeds 2%.
Ok-window median t_valid ≈ 12 ps (rim-limited mid-band); 0.370/0.450 run to ≈ 31 ps.
**Wrap-around** (run_time ≈ T = 40 ps): boundary levels exceed −40 dB on nine windows, but
the record end still *decays* (τ ≈ 6–7 ps) so the folded copy is ≤ 0.3% of P on trusted
segments (T2); mid-band boundaries sit at −50…−60 dB. **Excluded/flagged:** deep-gap 0.41
(floor: P peaks at frame 0) and 0.43 (long-lived gap-edge quasi-mode, not beam transport —
T6); flanks 0.389/0.390 (rim > 2% at 0.5 ps; qualitative); spectrum-edge 0.33/0.63
(clipped window ⇒ aliased end sidelobe, −22.7 dB predicted vs −22.5 dB measured at 0.33 —
0.33 trusted only to ≈ 5 ps). The first-crossing rim clock is over-conservative for
0.389/0.390/0.43, where a 0.3–1 ps rim transient hides an otherwise rim-clean quasi-mode
segment. Movies (log₁₀ I/I_max, fixed norm, 10⁻⁶ floor, 300 frames / 40 ps):
`movies/beam_spreading_nu0p350|0p389|0p437|0p550.gif`.

### 8.1 t_valid in plain terms

Each curve is a movie of a light pulse spreading sideways after crossing the slab;
`t_valid` is the moment we stop believing the movie, for whichever of two reasons happens
first:

1. **The signal runs out.** The pulse dies away, and what remains on the detector is a
   faint constant background hiss. Once the real signal comes within 3× of that hiss, the
   "width" we compute is the width of the hiss, not of the beam.
2. **The light runs off the edge.** The detector window is finite and the simulation edges
   absorb light. As the spot spreads, its outskirts eventually fall off the edge; from then
   on the spot *looks* like it stops growing — not because the physics stopped spreading
   it, but because we lose the widest light. That would fake exactly the saturation we care
   about, so we cut before it happens (when > 2% of the light sits near the edge, ρ > 45 a).

Before `t_valid` the curve is drawn solid (starting at arrival, `t_arr`); after, faint.
The two clocks bite differently across the band, and that is itself informative: mid-band
light spreads fast and hits the *edge* problem first (~11–19 ps), while the gap-adjacent
windows (0.370, 0.450) spread slowly, never reach the edge, and stay believable to ~31 ps
until the *signal-runs-out* clock ends them — which is why they are the only curves where a
flattening can be watched while the data is still trustworthy.

**Observations (descriptive; verdicts in §7 rows T1–T7):**

- Every pass-band window exits **narrow first**: d ≈ 9–12 a at arrival (0.27–0.43 ps), and
  σ²(t) keeps growing straight **through** the transmitted-power peak (t_pk ≈ 2.0–2.8 ps),
  slope +22…+53 a²/ps at the peak (T7).
- **Mid-band (0.47–0.61):** σ² grows monotonically, near-linearly, to ≈ 540–565 a² and d to
  ≈ 50 a at the rim cutoff (11–19 ps) — no flattening inside the trusted range. The edge
  clock bites at σ² ≈ 500 a² (absorbing-box ceiling 624 a²), so growth is observed;
  saturation is not observable there (T3).
- **Growth rate orders by distance from the gap:** σ²(5 ps) ≈ 250–265 a² mid-band, vs 132
  (0.370) and 118 a² (0.450) at the gap flanks — a factor 2–3 slower — and ≈ 450 a² below
  the gap (0.33; spectrum-edge flags apply) (T3).
- **The two gap-adjacent windows bend over inside their trusted ranges:** 0.450 flattens at
  σ² ≈ 297–305 a² over 25–32 ps (robust to stricter power cuts, floor-mixture algebra, and
  window/grid variants); 0.370 bends but still creeps, 285→363 a² over 18–31 ps. This
  plateau sits **~5× above** the ξ = 5–9 a localized-saturation scale (63–77 a², §5) —
  an observed flattening, not the sought localization plateau (T4).
- **0.437 (upper flank):** persistently narrow, barely-spreading profile — d ≈ 14–20 a
  (window-dependent), σ² ≈ 75–183 a² over 5–28 ps, rim ≤ 0.1%; *halving* the window
  bandwidth makes it narrower and flatter, killing the window-blending hypothesis. Contrast
  0.450, which keeps spreading to d ≈ 30 a (T5).
- **P(t):** mid-band decays quasi-exponentially over 6–7 decades; decay slows toward the
  gap (0.450/0.370 τ ≈ 6–7 ps; 0.437 τ ≈ 9 ps; 0.43 τ ≈ 11–12 ps; the 0.389/0.390 windows
  retain ~10% of peak power at 40 ps).

Verifier residuals: d(t) carries a few-percent quadrature/dtype sensitivity (up to 4.3% at
0.437/0.450; σ² ≤ 1.2%) — pin quadrature and dtype before any future d(t) claim finer than
~5%; σ²(5 ps) at 0.437 is window-sensitive (74–86 a²).

## 9. What would settle the open questions

1. **Time-domain σ²(t) on THIS data** — **done at first-look level (§8)**: growth observed
   everywhere in the pass bands; no saturation below the aperture clock mid-band; the
   gap-adjacent bend at ≈ 300–370 a² is ~5× above the ξ = 5–9 a target scale, so the
   decisive localized-saturation signature remains unresolved at this aperture/geometry
   (ensemble + thickness scaling still needed).
2. **Thickness scaling** (≥ 2 more L values) — the decisive localization discriminator.
3. **Ensemble realizations** — kills the +15% speckle systematic; enables mode statistics
   at the flank dips (the quasi-mode spots beg for a Thouless/mode-counting analysis).
4. **Interior monitor / Gaussian-beam source** — removes the interface-wave and
   source-realism residuals. (The 2026-08-11 effective-cladding run, §10, is the measured
   demonstration of what happens when neither is done and the faces are index-matched:
   the exit-plane widths stop being beam widths entirely.)

## 10. Effective-medium cladding run (2026-08-11) — why its σ²/d are NOT beam widths

A second cloud run (`…backdround_effective_n_1.34.h5`, project 20260810) repeated the
experiment with the z-regions outside the slab (±16…±22.5 µm, transversally infinite, up to
the absorbers) filled with the Bruggeman effective medium n_eff = 1.34 (ff = 0.217) instead
of vacuum — source (z = −19 µm) inside the entrance cladding, monitor unchanged at z = +16.
Motivation: remove the internal-reflection (z₀) systematic. Result: **σ² ×2.5 (≈490 vs
≈195 a² CW) and d ×1.5–1.6 (≈42 vs ≈27.5 a) as a frequency-flat multiplier** at all ν and
all t; CW rim fraction 3.3% everywhere (vs 0.2%); the 2% rim clock trips at ~2.6 ps in
every TD window (0.437 immediately); σ²(t) approaches the aperture ceiling (uniform-square
max 2W²/3 = 1591 a²); the 0.437 quasi-mode narrowing is destroyed (d ≈ 21–30 a).

**Sign test.** Diffusion theory predicts the matched run should be equal or ~15–25%
*narrower* in d (z₀: ≈1.67ℓ* → 0.67ℓ*; σ²(t) ≈ 4Dt is face-reflectivity-independent).
The observed opposite ⇒ the excess is not bulk transport.

**Mechanism (settled by post-processing experiments, rows E1–E4).** The excess is
*genuine Maxwell power, wrong experiment*: (i) the 5×5 µm hard patch radiating inside
n = 1.34 launches propagating power over nearly the full hemisphere at every in-band ν
(components evanescent in air become traveling waves) ⇒ the slab is illuminated over a much
wider footprint; (ii) the slab's ~70% diffuse reflectance exits the matched entrance face
instead of being TIR-recycled, skims laterally in the scattering-free cladding
(87 a/ps vs diffusive √(4Dt) ≈ 6–14 a in the first ps) and is re-scattered into the slab
off-axis by the rough face; (iii) the matched exit face releases the steep-angle 44% of the
diffuse flux that air traps (measured: 47% of transmitted flux at k∥ > k₀, vs Lambertian
prediction 1−1/n² = 44.4%). Surface re-scattering relabels angles, so neither the
flux (S_z) correction (−8%) nor an air-cone (k∥ < k₀) filter (−2…−5%) recovers the air-run
observable, and time-gating/deconvolution fail structurally (E4). ~25% of the exit power
arrives in the wide footprint (σ²_eff ≈ (1−f)·180 + f·1300 a² ⇒ f ≈ 0.25).

**Rules.** (1) Do not quote σ²/d from this run — not even filtered variants as bounds.
(2) Any σ² ≳ 550 a² on the ±48.85 a aperture is aperture-saturated for smooth profiles
(the Gaussian-halo 2%-rim threshold), independent of run. (3) The run retains value only
as a negative control and as the measured matched-boundary angular census.

**Fix (re-run prescription).** Apodized (Gaussian) source — or source embedded ≥ ℓ* inside
the disorder — plus a monitor 1–2 ℓ* *inside* the exit face, or an exit monitor recording
E **and** H so the observable is the flux S_z; apply identically to both boundary
conditions. The angular-spectrum pipeline
(`Claude/audit_workspace/effective_cladding/flux_correction.py`) drops onto such a run
unchanged. **A by-product for the air run:** its flux-based widths agree with the |E|²
widths to ≤6% — the exit-plane |E|² convention is hereby validated for the air data
(bounds the Goïcoechea interface-wave residual in §6).

## References

- Cherroret, Skipetrov & van Tiggelen, PRE **82**, 056603 (2010) — CW/pulsed transverse widths.
- Zhu, Pine & Weitz, PRA **44**, 3948 (1991); Haskell et al., JOSA A **11**, 2727 (1994);
  van Rossum & Nieuwenhuizen, RMP **71**, 313 (1999) — z₀/internal reflection.
- Goodman, JOSA **66**, 1145 (1976); Dogariu & Carminati, Phys. Rep. **559**, 1 (2015) —
  speckle statistics (C = 1/√3 for 3-component vector speckle).
- Hu et al., Nat. Phys. **4**, 945 (2008); Yamilov et al., Nat. Phys. **19**, 1308 (2023);
  Haberko, Froufe-Pérez & Scheffold, Nat. Commun. **11**, 4867 (2020); Goïcoechea et al.,
  arXiv:2606.04897 (2026) — localization context/guardrails.
