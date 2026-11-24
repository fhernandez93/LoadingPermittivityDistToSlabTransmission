# Prompt — Steady-state IPR / beam-diameter d(ν) deep-dive (next session)

Work in `H:\Codes\tidy3d\Notebooks\LSU Project\20251001_LSU_Localization_Tests\20260601_Beam_Spreading_freq`.
**Scope is deliberately narrow: the STEADY-STATE (CW, frequency-domain) analysis only** —
`20251008_IPR_Calculation.ipynb` and the participation-ratio beam diameter `d(ν)`. Do **not** work on
the time-domain / FFT σ²(t) notebook this session (`20260602_IPR_Calculation_FFT.ipynb`) except to
read it for cross-consistency of the shared steady state.

## Mandatory working rule — cross-check with agents, always

**Every nontrivial claim must be verified by independent adversarial agents before it is accepted.**
This is not optional and applies to my own conclusions too. For each claim I make, spawn 1–3 verifier
agents whose explicit job is to *refute* it (physics-correctness lens, numerical-correctness lens,
does-it-reproduce-in-the-actual-data lens). A claim survives only if the refuters fail. In the last
session an adversarial pass caught a real bug in my own pipeline that I had reported as a headline
conclusion — so run the refuters *before* reporting, not after. Keep fan-outs small (≤3–4 agents per
wave) because the session has repeatedly hit its usage limit with large fleets; prefer several small
waves and do cheap verification inline myself where possible.

## Context carried over (verify, don't assume)

Read first: `Claude/AUDIT_REPORT.md`, `Claude/METHODS.md`, `Claude/RESIDUAL_RISKS.md`, and the memory
`lsu-ipr-fft-audit-2026-07`, `lsu-diffusion-baseline-and-g-file-ordering`. Key facts from the audit
(re-verify any that a task depends on):

- **Physical system:** SHU-type disordered dielectric, n = 2.90, ff = 0.2237, a = 2.5626 µm,
  L = 14.3 µm = 5.58 a. Gap confirmed with MPB calculations ν = a/λ ∈ [0.432, 0.460] (empirically the S(f) dip is [0.427, 0.463],
  −33 dB). Data is the exit-face **frequency-domain** FieldMonitor (1700 bins), `normalize_index=None`
  (raw DFT, still carries the source envelope — but every per-bin scale-invariant estimator cancels it).
- **The steady state analyzed = the absorbers file** `data/L_1_12x/..._n_2p90_absorbers.h5`
  (ν ∈ [0.32, 1.25], 766×766 grid). The periodic file only spans ν ≤ 1.025.
- **`d(ν)` estimator** = participation-ratio diameter `2·sqrt(A_eff/π)`, `A_eff = (∫I)²/∫I²`,
  I = |Ex|²+|Ey|²+|Ez|², Simpson on the true (non-uniform) grid. Scale-invariant per bin; single-shot
  speckle biases it LOW by √(1+1/N_eff); the `d_avg` frequency-boxcar (w_ν = 11) is the ensemble
  surrogate.
- **Fixes already applied last session (confirm they are intact):** (1) the Cherroret baseline was
  hand-fudged (`8*L_slab` + `-55`) and is now the honest `diffusive_diameter_cw(x_, y_, L_slab, …)` with
  the ℓ* < L mask → flat baseline ≈ 11 a; (2) g(ν) is stored frequency-DESCENDING and is now reversed
  (`data_g["g"][::-1]`) to align with the ascending grid (ls is already ascending). Both were confirmed
  by an adversarial verifier that re-derived g from the structure.
- **Standing physics guardrails (keep, do not regress):** a CW width dip **cannot** prove localization
  (absorption / gap Bragg-attenuation give the identical σ² ≃ 2LL_a form — Cherroret 2010, we we can try to prove outside of the PBG boundaries where localization should be); the
  transverse structure is **12×-tiled** (period = L), so d(ν) beyond ~1 tile is Bloch-periodic (please elaborate on this), not
  transport; fixed single L ⇒ no localization length is claimable from the steady state alone.
- **`bst_pipeline.py`** (folder root) holds the validated primitives; reuse them. Do not modify
  anything under `data/`; back up `20251008_IPR_Calculation.ipynb` before editing.

## Goals for this session (steady state only)

1. **Make `d(ν)` and its uncertainty publication-grade.** Quantify and propagate the single-realization
   speckle error bar per frequency (N_eff from the frequency-correlation width δν_c, and/or a
   bootstrap over frequency sub-bands); verify `d_avg` (w_ν=11) stability under w_ν doubling (a check
   promised in the notebook but never run). Decide and justify the final estimator (raw d, d_avg, or a
   binned median) with its error band.
2. **Nail the diffusive baseline comparison.** With the de-fudged baseline and reversed g, re-state the
   d(ν)-vs-d_dif(ν) agreement in the diffusive band (ν ≳ 0.7) *with* the z₀ (do we need to calculate it???) and speckle systematics as
   an explicit error budget — no cherry-picked "6 %". Report where data sits above/below and why
   (band-edge enhancement vs gap dip). Cross-check the ℓ*(ν) that enters the baseline: g is
   Born/RDG-unreliable at this contrast, so treat ℓ* as indicative and show how much the baseline and
   the mask move if ℓ* is varied by a factor ~2.
3. **Quantify the gap dip rigorously.** Depth, location, and width of the d(ν) dip vs the MPB gap
   [0.432, 0.460]; convert it to an attenuation length via `sigma2_localized_cw` inversion **with the
   explicit caveat that this is L_att, not necessarily ξ**. Compare the PR-diameter dip to an RMS/second
   moment and to the transmitted-power spectrum S(ν) dip — do they locate the same feature?
4. **Aperture-truncation & background robustness of the CW d(ν).** Quantify how much d(ν) changes if the
   ±33.5 a aperture were smaller (truncate the map and recompute) and confirm the PR estimator's
   background-robustness claim on the real maps (it should be far less sensitive than ⟨ρ²⟩).
5. **Cross-consistency with the FFT side (read-only):** confirm the single-bin steady state used here
   equals the FFT notebook's per-bin intensity (last session: d(0.9) = 11.888 both ways) at 2–3 more
   frequencies, so the two notebooks provably share one steady state.


## Hard rules

- Verify against primary sources online where physics/conventions are involved (Cherroret PRE 82
  056603; the speckle-statistics / participation-ratio literature). Do not rely on memory for equations.
- No new Tidy3D cloud runs (cost money). Work from the existing HDF5 only.
- Report negative results honestly; flag any d(ν) feature that is not robust rather than interpreting it.
- **Cross-check every nontrivial claim with adversarial agents before accepting it (see the rule above).**
- It is important to first fully understand the diffusive regime and how it would saturate the spreading `d(ν)` in the steady states for each frequency, we could assume a fully diffusive system for this.

## Deliverables

1. Cleaned, fully-documented `20251008_IPR_Calculation.ipynb` (backup first) with the final estimator,
   error bars, and every assumption stated with its verification status.
2.Store a full report explaining the physics, the caveats and what needs to be changed on the simulation, or if what we have is faithful in  `Claude/STEADY_STATE_IPR_REPORT.md` — the d(ν) result with its error budget, the gap-dip quantification, the baseline-agreement statement, and each claim's adversarial CONFIRMED/REFUTED
   verdict.
1. Updated memory if anything from the carried-over context turns out to be wrong on re-verification.
