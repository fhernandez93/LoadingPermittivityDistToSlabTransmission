# Physics & numerics audit — IPR / broadband-FFT beam-spreading notebooks

**Scope.** `20251008_IPR_Calculation.ipynb` (steady-state PR beam diameter d(ν)) and
`20260602_IPR_Calculation_FFT.ipynb` (time-domain σ²(t)/D(ν) from FFT of frequency-domain
field data), plus the two raw datasets in `data/L_1_12x/` and the simulation/retrieve
notebooks that produce them.

**Method.** Multi-agent audit: parallel cell-by-cell mapping, an empirical probe of both 17.6/23.9 GB
HDF5 files, a literature sweep (full texts of Cherroret 2010, Hu 2008, Cobus 2018, the Sperling
saga, Yamilov 2023, Goïcoechea 2026, Froufe-Pérez 2017, Haberko 2020, Vynck RMP 2023), and
independent re-verification of every load-bearing claim against primary sources (producer code,
Tidy3D 2.9.1 source, the data itself). Working notes and downloaded sources are in
`Claude/audit_workspace/`; the validated pipeline is `bst_pipeline.py`.

**Originals** backed up in `backups_20260716/`. Nothing under `data/` was modified; no cloud runs.

Severity order: **wrong physics > systematic bias > noise/robustness > style.** Each finding lists
evidence, the fix, and the independent-verifier verdict.

Legend: **[C]** CONFIRMED (verified in code/data/source this pass) · **[P]** PLAUSIBLE (argued, not
fully closed) · units: a = 2.562629 µm, L = 14.3 µm = 5.580 a, ν = a/λ = af/c.

---

## A. WRONG PHYSICS

### A1 [C] The "Cherroret 2010 diffusive baseline" in the IPR notebook is hand-fudged
`20251008_IPR_Calculation.ipynb` cell 12:
```python
d_diff[valid] = diffusive_diameter_cw(x_, y_, 8*L_slab, l_t_sampled[valid], a_norm=a) - 55
```
It passes **8·L_slab = 114.4 µm** as the slab thickness (true L = 14.3 µm) and **subtracts 55** in
d/a units, and the validity mask `& (l_t_sampled <= 2*L_slab)` is commented out. The cells 8/14
markdown advertises the curve as having **"no free fit parameters (only L and the measured ℓ*(ν))."**

- **Evidence.** Read directly from the notebook. Recomputation: the committed code gives a nearly
  flat d_dif/a ≈ 12.86–12.88 for ν ≥ 0.45 (12.87 at ν = 0.9); the honest call
  `diffusive_diameter_cw(x_, y_, L_slab, …)` gives 11.08–11.62 (11.33 at ν = 0.9). The fudge lifts
  the baseline ≈ 13 % and erases the ℓ* structure. Git forensics: the same commit `ad22f9a`
  (2026-06-22, "Added accurate g calculation") that removed the earlier hard-coded `L=15` bug
  replaced it with `8*L_slab` and appended `-55` — so the "L-bug fixed" note is nominally true but
  the fix was subverted in the same commit. The **saved cell-13 figure matches the fudged code**;
  the **markdown text matches the honest code** — the notebook is internally inconsistent, and
  cells 10/12 have `execution_count=None` while dependents show 14/51, so the saved state cannot say
  which code made which output.
- **Why it matters.** σ²_dif = (2/3)L²[1−(ℓ*/L)²] is *set by L* and is essentially flat in ν
  (Cherroret Eq. 4); the honest baseline is ≈ 11.3 a. The "+6 % agreement at high ν" claim, and the
  sign of the data-vs-baseline offset, both depend on the fudge (with the honest baseline the raw
  data sits *below* the line at high ν, not above).
- **Fix.** Restore `diffusive_diameter_cw(x_, y_, L_slab, l_t_sampled[valid], a_norm=a)` (no offset),
  restore the ℓ* < L (and optional ≤ 0.3 L) validity mask, and rewrite the markdown to state the
  baseline is a flat ≈ 11.3 a reference. Keep the qualitative conclusion (gap dip factor ≈ 3) — it
  survives — but drop the spurious "6 % above" precision.
- **Verifier verdict: CONFIRMED** (physics, numerical, in-code lenses all agree).

### A2 [C] The g(ν) anisotropy factor is used frequency-reversed
`20251008…` cell 10 pairs `data_g["g"]` element-wise with the **ascending** monitor frequency grid,
with a comment asserting "g is already in ascending-frequency order." **It is not.**

- **Evidence (producer code, read this pass).** `20250903_create_h5_from_ends.ipynb` cell 10:
  `results = rdg_spectrum(wavelengths = a/np.linspace(1.25, 0.32, 1700))` — i.e. ν **descending**;
  `rdg_spectrum` does `for lam in wavelengths:` and returns `g` in that order unchanged; cell 11
  writes `{"g": results["g"], "nu": a/results["wavelengths"]}` — g and nu **co-stored, both
  descending in ν**. The file confirms: `nu` is exactly `linspace(1.25, 0.32, 1700)`, g[0] = +0.344
  at ν = 1.25, g[−1] = −0.313 at ν = 0.32. Co-aligned, g **rises** with ν (corr +0.91), which is the
  physically correct trend for correlated-disorder (LSU) Born scattering (back-scattering-biased g < 0
  at low ν, forward g > 0 at high ν). The notebook's pairing puts g = +0.34 at ν = 0.32 and
  g = −0.41 at ν = 0.9 — the opposite, unphysical trend. **`ls` is correctly ascending** (its
  minimum ls = 1.328 sits at index 226 → ν = 0.4437, dead in the gap); only `g` is reversed.
- **Correction to prior record.** A prior review/memory asserted "g ascending, confirmed by the data
  owner," with empirical support that "g-as-is matches the baseline at high ν." That support was
  **circular** — the baseline it matched is the fudged one (A1). The producer code is unambiguous;
  the memory has been corrected.
- **Impact.** ℓ*(ν) = ℓ_s/(1−g) is mirrored about ν = 0.785: ℓ*(0.9) 3.17 → 4.47 µm; the ℓ* < L
  fraction and the validity mask shift; any D = v_E ℓ*/3 anchor inherited downstream moves ×~1.4.
- **Second, independent problem.** These g values are **Born/RDG single-scattering**, invalid at this
  contrast — the generator itself warns `|m−1| = 1.90 > 0.1` and `k·d·|m−1| = 83 > 1`. So ℓ* via
  this g is only *indicative regardless of ordering*. Mitigating fact: σ²_dif is dominated by L, so
  the baseline *level* barely moves; g mainly sets the ℓ*/L axis and the validity mask.
- **Fix.** Reverse g (`g_arr = data_g["g"][::-1]`) to align with ascending ν, **and** add a prominent
  caveat that ℓ* is Born-unreliable and used only to place the validity mask / secondary axis. Do not
  quote ℓ*-derived numbers to better than a factor ~2. This is flagged to the user because it
  contradicts a prior explicit confirmation.
- **Verifier verdict: CONFIRMED** (producer code + file + physics + ls-gap cross-check converge).

### A3 [C] Structural showstopper: the medium is 12×-tiled with transverse period = L
The simulation tiles the 14.3³ µm disordered base cell **12×12 in x,y** (144 `CustomMedium` boxes;
verified in both Simulation JSONs). The transverse disorder therefore **repeats every 5.58 a = L**.

- **Consequence.** The sample is, strictly, a 2D-periodic photonic crystal with a disordered L×L×L
  unit cell. Transverse eigenstates are Bloch waves; genuine transverse Anderson localization on
  scales larger than one tile is **impossible in the ideal tiled system**, and transverse spreading
  must become Bloch-ballistic once σ exceeds ~1 tile. σ²(t) is physically interpretable only while
  **σ ≲ 1 tile, i.e. σ² ≲ 30 a²**. The prior FFT fits ran to σ² = 150–230 a² (σ ≈ 2–3 tiles) — deep
  in the contaminated regime, which is the direct cause of the NaN ξ (A5) and implausible D (B2).
- **Fix.** Hard-cap every σ²(t) fit and every saturation/ξ inversion at σ² < ~25–30 a²; state the
  tiling limit as a first-class caveat in both notebooks and in METHODS. A ξ can only be claimed if a
  plateau appears *below* ~15 a² and *within* one tile.
- **Verifier verdict: CONFIRMED** (geometry from JSON; the Bloch-periodicity consequence is textbook).

### A4 [C] "IPR/localization" cannot be read off a CW width dip — absorption/gap attenuation mimics it
Not a code bug but a physics-interpretation guardrail that must stay in place. Cherroret 2010
(pp. 7–8) proves the stationary width obeys σ² ≃ 2 L L_a under absorption — the **identical form** as
the localized 2Lξ; inside a photonic gap, Bragg/evanescent decay plays exactly the role of L_a.

- **Status.** The softened language ("gap dip yields an attenuation length; localization attribution
  rests on σ²(t) and L-scaling") **is present** in the current IPR markdown (cells 8, 14) — good, keep
  it. The probe confirms the CW gap dip is real and deterministic (cross-run NCC 0.97 at ν ≈ 0.45,
  depth −33 dB) but that does not make it localization.
- **Literature guardrail.** Cobus 2018: "a transmission dip … should not be used on its own as an
  indication of localization." Sperling saga: a saturating width repeatedly turned out to be a
  systematic (there fluorescence; here candidates are aperture, background, wrap, tiling).
- **Verifier verdict: CONFIRMED correct-as-is** (do not regress the caveat).

---

## B. SYSTEMATIC BIAS

### B1 [C] The FFT notebook analyzes the periodic (wrap-around) file, not the absorbers file
`20260602…` cell 1 loads `files["no_absorbers"]` (the periodic-transverse-boundary run). Confirmed
three ways: printed T = 20.61 ps (= periodic df 48.52 GHz), window bin-counts 49/48/24/120 reproduce
only from the periodic f-grid, and the ν = 0.90 ± 0.025 window fits only the periodic band. The
absorbers file is header-printed but never analyzed; **no cross-run comparison exists.**

- **Why it matters.** The periodic run lets the spreading halo **wrap around** the transverse
  boundary and re-enter, aliasing σ²(t) and the PR upward once σ reaches the ±33.5 a edge. The
  notebook's `t_valid_max` watershed guards only FFT *time*-aliasing, not transverse wrap. The
  absorbers run has no wrap (it loses edge energy instead), so it is the correct primary source for
  σ²(t); running **both** and requiring agreement at early times is a free validation.
- **Fix.** Analyze the **absorbers** file for σ²(t)/D; use the **periodic** file only for the
  early-time cross-check (t below the wrap onset). Provide an explicit dataset switch and print which
  file is loaded.
- **Verifier verdict: CONFIRMED.**

### B2 [C] D(ν) fit is unweighted over the full saturated/contaminated range → D≈20× too small
`20260602…` cell 7 does `np.polyfit(σ²(τ), 1)` over the entire trusted range (including the
saturated plateau and σ² up to 150–230 a²). Result: R² ≤ 0.153 in every window; D ≈ 0.32 a²/ps at
ν = 0.9 — a factor ~20 below the plausible range and *below Yamilov's white-paint minimum*
(0.55 a²/ps). Negative slope in-gap is quoted in the table.

- **Root cause.** No fit-range discipline: the linear-growth regime σ²(t) = 4Dt (Cherroret) exists
  only for t between the pulse arrival and the first artifact time (aperture/tiling/wrap). Fitting
  through the whole record drives the slope toward zero.
- **Fix (in `bst_pipeline.valid_window`), as corrected in E1.** Fit only the **pre-peak rising edge**
  [t_arrival, t_hi], t_hi = earliest of {σ² > 25 a² (≈1 tile), edge-power > 1 %, the power peak}.
  Note: my *first* version of this fix started the window at the power peak and still gave D ≈ 0 — an
  adversarial verifier caught it (see E1). The corrected rising-edge fit recovers **D(0.9) ≈ 6 a²/ps**
  (R² ≥ 0.99, cross-file ~4 %), matching the anchor.
- **Verifier verdict: CONFIRMED for the original notebook** (its unweighted full-range fit genuinely
  gave D ≈ 0.32); the *remedy* required the E1 rising-edge correction.

### B3 [C] ξ inversion returns NaN because it is fed above-ceiling plateaus
`20260602…` cell 7 inverts σ²_∞ = 2Lξ(1−ξ/L) for ξ. This has real roots only for σ²_∞ ≤ L²/2 =
15.57 a²; the fed late-time σ² (26–231 a²) exceeds it → NaN in all windows, printed without comment.

- **Fix.** Only invert a plateau established *within the valid window* (σ² < 25 a²); if the plateau is
  above L²/2 the correct report is "above the localization ceiling → mobility-edge/aperture scale,
  not ξ," not a silent NaN. Implemented in `bst_pipeline.xi_from_saturation`.
- **Verifier verdict: CONFIRMED.**

### B4 [C] The two datasets are different experiments, not one experiment at two BCs
The periodic and absorbers runs differ in **three** parameters (git diff of the experiment notebook):
λ-range (ν ≤ 1.025 vs ν ≤ 1.25), df (48.52 vs 64.04 GHz → T = 20.61 vs 15.62 ps), and absorber-layer
count (130 z-only vs 120 all-axis). **Their frequency grids share no common bin** (nearest-bin ν
mismatch up to 4.3e-4 near the gap), and the meshes differ (dx 0.087 vs 0.075 µm).

- **Additional measured anomaly.** Integrated exit power differs by a **smooth factor 1.4–3.3**
  (periodic/absorbers), flat in radius within 20 a — an unresolved global scale mismatch (candidate
  causes: different staircased mesh → different transmission; lateral loss into side absorbers;
  an unrecorded launch difference). **Do not compare the two files' absolute levels.**
- **Fix.** Any cross-run comparison must be on **normalized** profiles at **nearest-bin** ν (or
  interpolated), never absolute. Cross-run agreement holds for *shapes* (NCC 0.88–0.97) not
  amplitudes. State this in METHODS.
- **Verifier verdict: CONFIRMED** (from JSONs + direct measurement).

### B5 [C] Absorbers run structurally time-aliases; both runs have a non-physical reconstruction floor
run_time (20 ps) **exceeds** the absorbers FFT period T = 15.62 ps by 4.38 ps, so recorded signal in
t ∈ (15.6, 20] ps folds onto t′ = t − 15.62 ∈ (0, 4.4] ps (amplitude ≤ 2.7e-9 — small but nonzero;
**reconstructed times > 15.62 ps do not exist**). The periodic run (T = 20.61 ps > 20 ps) does not
time-fold but its field is only decayed to 5.5e-5 at the hard stop, so its bins are sinc-correlated
and it carries a ~5e-5 ringing floor. The probe shows a reconstruction floor (~1e-5 of peak) in
*both*, and since the periodic run cannot alias, the floor is **mostly band-truncation leakage, not
physical non-decayed field**.

- **Fix.** Restrict σ²(t) analysis to t < ~0.6·(post-peak power minimum) and, for absorbers, t <
  15.6 ps; treat the last decade above the floor as unreliable. Do not read late-time "saturation"
  that sits on the floor.
- **Verifier verdict: CONFIRMED.**

### B6 [C] Stored data is NOT source-normalized (normalize_index = None)
Both Simulation JSONs have `normalize_index: null`, so the stored fields are the **raw running DFT**,
still carrying the Gaussian source envelope |S(f)| (down only to e^−0.5 = 0.61 at the band edges,
since the monitor band = ±1 fwidth). This **corrects an assumption stated to the audit fleet** (the
data is *not* pre-deconvolved).

- **Consequences, correctly handled.** (i) There is therefore **no double-deconvolution** risk — and
  the FFT notebook correctly does not deconvolve. (ii) Because the band edges are only e^−0.5 down,
  **there is no band-edge noise blow-up** (probe: the whole band is smooth and usable — this refines
  a prior "−20 dB threshold" concern which does not apply here). (iii) For the per-bin
  **scale-invariant** estimators (PR diameter, σ² ratio, normalized cross-correlation) the envelope
  is a per-bin scalar and **cancels** — so d(ν), σ²(t) shapes are unaffected. (iv) BUT any **absolute
  cross-ν** intensity/transmission comparison is |S(f)|²-weighted and must not be made without
  deconvolving by the discrete `source_time.spectrum()`.
- **Fix.** State the convention explicitly in both notebooks; forbid absolute cross-ν comparisons of
  the raw fields.
- **Verifier verdict: CONFIRMED** (Tidy3D 2.9.1 source + JSON).

### B7 [C] No incoherent/speckle averaging or ballistic removal in the time domain; no ρ-resolved width
The FFT notebook computes σ²(t) and PR from a **single** disorder realization's coherent speckle with
no ensemble surrogate, no coherent/ballistic subtraction, and no ρ-resolved (Hu) width — all of which
the credible experiments (Hu 2008: 500–3000 configs + ψ−⟨ψ⟩; Cobus 2018; Goïcoechea 2026) require.
Single-shot PR is biased low by ×(1+1/N_eff)⁻¹ (√2 for one polarization, √(4/3) for the 3-component
sum). The in-gap window's earliest frame (t_peak ≈ 0.2 ps) is a ballistic-spot signature that inflates
early σ².

- **Fix.** (i) Frequency-window incoherent averaging is intrinsic to the narrow-band synthesis (the
  window averages ~Δν/δν_c independent speckles) — report N_eff and the residual bias. (ii) Add the
  **Hu ratio width** w_ρ²(t) = −ρ²/ln[I(ρ,t)/I(0,t)] (implemented in `bst_pipeline.hu_width2_of_t`),
  which is background- and aperture-free and whose ρ-dependence is the real localization
  discriminator. (iii) Exclude t ≲ arrival + t_p from all fits. (iv) A shifted-source second FDTD run
  would be a legitimate second realization (recommendation only — not launched).
- **Verifier verdict: CONFIRMED** (literature standard).

### B8 [C] Background-subtraction lever arm is enormous and never bracketed
σ² sits on a uniform-background lever arm ⟨ρ²⟩_ap = 747 a²: a flat background at fraction ε of the
frame mean shifts σ² by ≈ ε·(747 − σ²) a². With late σ² of 26–231 a², percent-level background
errors dominate. The FFT notebook uses a single `background_rho = 25` annulus mean (unclipped) and
never runs the frame_floor bracket its own docstring prescribes; `power_floor_fraction` was overridden
from 1e-4 to 1e-9, admitting noise-floor frames into fits.

- **Fix.** Bracket `background_rho` ∈ {25, 28, 30 a} and report the σ²(t) band; restore a physical
  power gate (~1e-4); estimate the noise floor from pre-arrival frames. Implemented as `bg_rho_list`
  in `bst_pipeline`. The PR diameter is background-robust and must be applied to the **non-negative**
  intensity (the FFT notebook wrongly applied it to signed background-subtracted frames).
- **Verifier verdict: CONFIRMED.**

### B9 [P] Finite 2.5×2.5 µm PlaneWave patch is a frequency-dependent source
The injected "beam" is a hard-edged uniform PlaneWave patch of side 2.5 µm ≈ 0.31–1.22 λ over the
band — sub-wavelength at low ν. Its injected angular content and power vary strongly with ν
(Tidy3D docs warn a finite source whose profile does not decay at its edges can produce spurious
artifacts). So part of the ν-dependence of the spreading is **source** ν-dependence, not transport.

- **Fix.** Acknowledge in METHODS; the diffusive-anchor control (recover 4Dt at ν = 0.9 with a
  physically sensible D) partly absorbs this. A GaussianBeam or embedded interior source would be the
  clean fix (recommendation only).
- **Verifier verdict: PLAUSIBLE** (documented behavior; magnitude unquantified here).

---

## C. NOISE / ROBUSTNESS / CORRECTNESS-OF-DETAIL

- **C1 [C]** Mesh degrades to ≈ 9.4–9.9 steps per in-medium wavelength at the band top (AutoGrid
  anchored at λ0; the λ_min-based refinement is dead code) → numerical dispersion grows toward high ν
  and the two runs discretize the same ν differently. *Fix:* flag high-ν (ν ≳ 1) results as
  resolution-limited; a λ0/N convergence scan is the clean test (recommendation).
- **C2 [C]** FFT **sign convention CONFIRMED** (Tidy3D stores Ê = (Δt/√2π)Σ E e^{+2πift}; `scipy.fft.fft`
  reconstructs causal E(t), `ifft` mirrors it) — verified from 2.9.1 source *and* empirically (≥ 99.4 %
  of reconstructed energy in the first half-period; exact mirror under ifft). Prior claim holds; keep it,
  but the notebook should assert it with an inline causality check, not only a comment.
- **C3 [C]** `np.allclose` on raw fields is **vacuous** (amplitudes ~1e-15 ≪ default atol 1e-8) — any
  sanity check built on it proves nothing. Component powers are near-equipartitioned (Ex:Ey:Ez ≈
  0.34:0.34:0.31) — full depolarization, so an Ex-only analysis discards ~2/3 of the intensity.
- **C4 [C]** Absorbers monitor grid is slightly non-uniform (last interval 0.150 vs 0.225 µm; no exact
  x = 0 point) — use true-coordinate quadrature weights (done in `bst_pipeline`), not `dx = x[1]−x[0]`.
- **C5 [C]** Eager `read_hdf5_as_dict` materializes the whole 17.6/23.9 GB file into RAM; the pipeline
  streams row-blocks instead. Execution-order/namespace traps in both notebooks (`d`, `freq`, `I`
  reused; cells with `execution_count=None`) — the cleaned notebooks run top-to-bottom with no hidden
  state.
- **C6 [C]** Gap-edge provenance: `gap_data.hdf5["Circular"]["0.22"]` (ff 0.22 vs structure 0.2237;
  key "Circular"); the [0.432, 0.460] window matches the measured S(f) dip [0.427, 0.463] to < 2e-5
  in ν and −33 dB depth across both runs, so the value is empirically corroborated even if the MPB
  provenance is not re-derived here.

---

## D. What was checked and found CORRECT (no regression)

- FFT sign/phase convention (C2) — CONFIRMED from source and data.
- Parseval/energy conservation — the pipeline's time-vs-spectral energy ratio = **1.0000** (acceptance
  test 2, below).
- d_avg speckle mitigation (w_ν = 11 frequency boxcar before the PR) — **present** and correct in the
  IPR notebook (cell 11).
- CW localization/absorption-ambiguity caveat — **present** in the IPR markdown (A4).
- PR `diameter()` estimator: correct participation-area definition, true-coordinate Simpson weights,
  scale-invariant, resolution-benign — a sound estimator (the problem is the fudged *baseline* it is
  compared to, not the estimator).
- HDF5 axis ordering of the two field files — all axes ascending, no g_data-style reversal (the
  reversal is confined to the separate g file, A2).
- Data integrity — bit-exact match between each exported h5 and its cloud `Data.hdf5`; no
  cross-contamination between the two runs; no NaNs/zeros/duplicate frequencies.

---

## E. Acceptance-test results (validated pipeline on the real data)

Run: `bst_pipeline` streamed over **both** files, 7 windows each (raw summary in
`Claude/audit_workspace/acceptance_summary.txt`; curves in `probe/acceptance_*.npz`; cross-check in
`probe/cross_consistency_nu0p90.npz`).

| # | Acceptance test | Result | Verdict |
|---|---|---|---|
| 1 | **Cross-notebook consistency.** FFT single-bin steady state vs IPR-notebook d(ν) at ν = 0.9 | FFT single-bin **d(0.9) = 11.888** = IPR notebook **11.9** (identical — same data + estimator). Window time-integrated (incoherent band-average, the speckle-mitigated width) = 14.0 (+18 %), map NCC 0.86 — both correct, measuring coherent-single-bin vs band-averaged intensity. | **PASS** |
| 2 | **Parseval** Σ_t P(t) vs N·Σ_f\|WÊ\|² | ratio = **1.0000** in all 14 windows, both files | **PASS** |
| 3 | **Robustness** of D to window shape, Δν, ρ_bg, fit range | On the corrected rising-edge window (E1), D(0.9) is stable: raw 7.4–9.3 across fit-range sweeps (all R²>0.99), 7.95 / 6.36 across Δν = 0.03 / 0.015, and the raw–vs–background bracket is [5.8, 8.0] a²/ps | **PASS (diffusive regime)** |
| 4 | **Physical anchor** — recover 4Dt with D(0.9) ≈ 2–15 a²/ps | **ACHIEVED.** Rising-edge fit gives D(0.9) = 5.8 (bg) – 8.0 (raw) a²/ps, R² ≥ 0.99, agreeing between the two files to ~4 % — **matches the prior D ≈ 6 a²/ps anchor** | **PASS** |
| 5 | **End-to-end** from raw HDF5, no manual steps/hidden state | `bst_pipeline` streams both 24 GB files start-to-finish; cleaned notebooks run top-to-bottom | **PASS** |

### E1 [C] CORRECTED (adversarial verifier caught a pipeline bug): the diffusive D(ν) *is* recoverable; the localized regime is not
An earlier draft of this report claimed the exit-plane σ²(t) could not yield *any* robust D. An
adversarial verifier **refuted that** and exposed a real bug in my pipeline, which I then reproduced
and fixed. The correction:

- **The bug.** `bst_pipeline.valid_window` started the diffusion fit at the *power peak*
  (`i_lo = i_peak`), discarding the entire growth phase. The clean 4Dt growth lives on the **pre-peak
  rising edge** — early-arriving light has spread less, later-arriving light more, so σ²(t) grows
  *through* the power peak. Fitting only the post-peak decay gave the spurious D ≈ 0 / negative.
- **The fix (verified).** With `i_lo` = pulse arrival (P first exceeds 2 % of peak) and `i_hi` =
  earliest of {σ² > 25 a², edge-power > 1 %, the power peak}, over the rising edge t ≈ 0.16–0.56 ps
  at ν = 0.9 (σ² grows 5.6 → 20 a², below the tile cap, edge-power < 1 %, SNR ~ 5000):
  **D(0.9) = 5.8 (background-subtracted, R² = 0.999) to 8.0 (raw, R² = 0.992) a²/ps**, stable across
  fit-window sweeps (7.4–9.3) and agreeing between the absorbers and periodic files to ~4 %
  (periodic 5.5–7.1). Independent estimators concur: background-free Hu ratio width → D ≈ 7.5–9,
  PR-diameter d²/8 → D ≈ 4.5. This **reproduces the prior D ≈ 6 a²/ps anchor** and validates the
  broadband-FFT method **in the diffusive regime**.

So the corrected, nuanced answer to the mission question:
- **Diffusive regime (ν ≈ 0.65–0.9): accessible.** D(ν) is robustly measurable — D(0.9) ≈ 6 a²/ps,
  D(0.65) ≈ 11–17 a²/ps (rising above the gap, as expected). The method works here.
- **Low-ν (ν ≈ 0.40–0.42) and in-gap (ν ≈ 0.446): D not cleanly extractable.** The beam arrives
  already filling a large fraction of the aperture (raw σ² ≈ 40–48 a² at ν = 0.40, edge-power 8 %
  *at the peak*), so the rising-edge window is too short and background subtraction fails (this part
  of the earlier negative result stands). Low ν is quasi-ballistic and the gap is not diffusive, so a
  4Dt fit is not physically expected there anyway.
- **Localized regime / ξ: still not accessible.** The **12× tiling (A3)** caps interpretable σ² at
  ~1 tile, the slab is thin (L = 5.6 a) and single-L, and the in-gap window is not diffusive — so a
  saturation plateau cannot be established and ξ cannot be extracted. The localization conclusion
  (A3, A4) is unchanged: consistent with transverse confinement in/near the gap, **localization not
  proven**, and no ξ from these data.

The other robust time-domain deliverables stand: Parseval = 1, cross-file early-time agreement to
0.5 %, and a pulse-arrival width d_PR(ν) tracking the steady-state d(ν) (wide low-ν, narrow in-gap).

### E2 Adversarial verification round (3 refuters, one per load-bearing claim)
Run explicitly to *break* each claim (`Claude/audit_workspace/verify/`, workflow
`wf_d4500e04-09d`). Verdicts:

| Claim | Verdict | Outcome |
|---|---|---|
| **A2** g(ν) is stored ν-descending → must be reversed | **SURVIVES** | Refuter re-derived g independently from the 512³ structure (rdg algorithm) and reproduced the descending/co-indexed sign (g rises with ν, g(0.32) ≈ −0.47, g(0.9) ≈ −0.02, g(1.25) ≈ +0.34); confirmed AutomationModule does no reorder on read/write, that `ls` is genuinely ascending (opposite order), and that `g[::-1]` grid-matches the absorbers monitor to machine precision. Fix confirmed correct. |
| **A3** 12× tiling → transversely periodic → localization capped at ~1 tile | **SURVIVES** | Refuter traced `loadStructures.py` and proved the 144 tiles are **byte-identical translated copies** of one permittivity array (no rotation/flip/reseed; 12×12×1; pitch = L exactly), and showed the σ² ≤ 30 a² cap is *generous* (in-gap CW power beyond one period is 23 %). |
| **E-draft** exit-plane σ²(t) cannot yield *any* robust D | **REFUTED** | Caught the `valid_window` bug above; D(0.9) ≈ 6 a²/ps is robustly recoverable on the rising edge. Report corrected (E1); pipeline fixed. |

Two of three load-bearing claims survived adversarial refutation with *added* independent evidence;
the third was refuted, which is the process working as intended — the wrong conclusion was mine, and
it is now fixed and re-verified.

---

## F. Bottom line

The pipeline's **building blocks are sound** (PR estimator, FFT convention verified, Parseval = 1) but
the *committed state* contained one fabrication-grade issue (A1, the fudged baseline), one silent
data-ordering bug (A2), and fit-discipline/boundary/averaging systematics (B1–B8) that made the
headline D(ν)/ξ(ν) numbers unusable as reported.

Running the corrected pipeline on the real data (after an adversarial verifier caught a fit-window
bug, E1): the **diffusive regime is accessible** — D(ν = 0.9) ≈ 6 a²/ps is robustly recovered from the
pre-peak rising edge (R² ≥ 0.99, cross-file agreement ~4 %, matching the prior anchor), and D(0.65) ≈
11–17 a²/ps rises above the gap as expected. This validates the broadband-FFT method for measuring
D(ν). The **localized regime is not accessible**: the **12× tiling (A3)** caps interpretable σ² at one
tile, the slab is thin (L = 5.6 a) and single-L, and the in-gap window is not diffusive, so no
saturation plateau can be established and no ξ extracted. Low-ν/in-gap D is also not cleanly
extractable (aperture-filling / non-diffusive). The strongest defensible statement the data supports is
therefore: *"the diffusive regime and its D(ν) are accessible and measured; the localized regime is
not — consistent with transverse confinement in/near the gap, but localization not proven and no ξ
from these data."* This is the posture the IPR notebook already adopts on the localization question,
and the one the Sperling/Scheffold/Skipetrov history shows is the only safe one.

To actually access both regimes one needs (cheapest first, `RESIDUAL_RISKS.md`): a **thicker,
non-tiled** slab (R1, R2) and an **interior monitor plane** (R8) — none launchable from the existing
data. The cleaned notebooks + `bst_pipeline.py` implement every fix above, run end-to-end from the raw
files, and report every result with its robustness so that an unreliable number is flagged rather than
published.
