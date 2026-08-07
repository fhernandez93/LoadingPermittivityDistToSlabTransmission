# Beam spreading vs frequency — fully random LSU slab 250×250×32 µm, n = 3.3

Steady-state (CW, frequency-domain) transverse beam-diameter analysis d(ν) at the exit
face of a **fully random, aperiodic** LSU-generated healed cylinder network — the run that
replaces the old 12×-tiled SHU experiment. Analysis notebook:
**`20260806_Beam_Diameter_d_nu.ipynb`** (lean; all physics prose lives here).
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
| `20260602_IPR_Calculation_FFT.ipynb` | time-domain (FFT) analysis of the **old 12× data** — untouched, does not apply to this run |
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
| exit-face monitor (interface waves, Goïcoechea 2026) | unquantified | — | needs an interior monitor (re-run) |

## 7. Claims table — adversarial verdicts (2026-08-07)

Claims stated in `Claude/audit_workspace/random_slab/CLAIMS_FOR_VERIFICATION.md`;
verification scripts in `…/random_slab/verify/`. Wave A = literature/primary-source
verifiers; waves B = independent data refuters.

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

**Corrections to carried-over context adopted this session** (wave A, primary sources):
(i) CW localized width = full Cherroret Eq. 6; σ²=2Lξ(1−ξ/L) is the pulsed conjecture.
(ii) z₀/ℓ* = 1.45–4.0 (ZPW/Haskell-verified) — the old report's Egan-Hilgeman 1.5–3.3 was
approximately right (its R values were the flux-average 2C1), an intermediate computation
this session (R_eff=0.84) was wrong (convention mixing).
(iii) d²/8 → ⟨ρ²⟩ is profile-dependent (×0.67–1.0) and invalid on quasi-mode spots.
(iv) The old n=2.90 g-file ν-descending convention does **not** apply to the new n=3.3 g
file (ν ascending, co-stored).

## 8. What would settle the open questions

1. **Time-domain σ²(t) on THIS data** (the h5 supports it: 40 ps window, 1500 bins) — the
   absorption-immune test; saturation at 60–80 a² now measurable (no tiling cap).
2. **Thickness scaling** (≥ 2 more L values) — the decisive localization discriminator.
3. **Ensemble realizations** — kills the +15% speckle systematic; enables mode statistics
   at the flank dips (the quasi-mode spots beg for a Thouless/mode-counting analysis).
4. **Interior monitor / Gaussian-beam source** — removes the interface-wave and
   source-realism residuals.

## References

- Cherroret, Skipetrov & van Tiggelen, PRE **82**, 056603 (2010) — CW/pulsed transverse widths.
- Zhu, Pine & Weitz, PRA **44**, 3948 (1991); Haskell et al., JOSA A **11**, 2727 (1994);
  van Rossum & Nieuwenhuizen, RMP **71**, 313 (1999) — z₀/internal reflection.
- Goodman, JOSA **66**, 1145 (1976); Dogariu & Carminati, Phys. Rep. **559**, 1 (2015) —
  speckle statistics (C = 1/√3 for 3-component vector speckle).
- Hu et al., Nat. Phys. **4**, 945 (2008); Yamilov et al., Nat. Phys. **19**, 1308 (2023);
  Haberko, Froufe-Pérez & Scheffold, Nat. Commun. **11**, 4867 (2020); Goïcoechea et al.,
  arXiv:2606.04897 (2026) — localization context/guardrails.
