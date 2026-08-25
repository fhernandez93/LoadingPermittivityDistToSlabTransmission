# Prompt — Physics audit & cleanup of the IPR / FFT beam-spreading notebooks

Work in `H:\Codes\tidy3d\Notebooks\LSU Project\20251001_LSU_Localization_Tests\20260601_Beam_Spreading_freq`. Use multi-agent workflows: fan out subagents for the audit, literature checks, and adversarial verification of every claim.

## Context

We are doing light-transport numerical experiments on LSU disordered structures with Tidy3D FDTD. A focused beam is injected into the sample and we study its **transverse spreading as a function of frequency**, to determine whether we can access both the **diffusive** and **Anderson-localized** transport regimes in the same dataset.

Files in this folder:

- `20251008_IPR_Calculation.ipynb` — computes the **inverse participation ratio (IPR)** of the steady-state transverse intensity profiles, frequency by frequency.
- `20260602_IPR_Calculation_FFT.ipynb` — takes the **FFT of broadband time-domain field data** to reconstruct per-frequency responses and study the transverse spreading in the time domain (σ²(t), D(ν), IPR(ν)).
- `20251001_numerical_experiment.ipynb` — defines the Tidy3D simulation (source, monitors, structure). Read it to understand exactly what the raw data contains (source spectrum, monitor placement/aperture, run time, boundary conditions).
- `20251007_Retireve_Field_Data.ipynb` — downloads the raw data.
- `data/L_1_12x/field_data_12x_raw_field_Ex_Ey_Ez_L_1_ff_0p2237_n_2p90.h5` and `..._absorbers.h5` — raw Ex, Ey, Ez time-series field data. The two datasets differ **only in boundary conditions**: the `_absorbers` run uses absorbing boundaries in **all** directions (adiabatic absorbers at both ends along the propagation axis to absorb all transmitted and reflected waves, and absorbers instead of periodic boundaries in the transverse directions); the other run uses **periodic boundary conditions perpendicular to the propagation direction**. Confirm the exact setup in the experiment notebook, and account for the physical consequences in each analysis: the periodic run allows transverse wrap-around of the spreading beam (aliased re-entry contaminating σ²(t) and IPR once the profile reaches the lateral boundary), while the absorber run removes wrap-around but loses energy at the transverse edges (truncation/leakage bias on ⟨r²⟩ and on late-time dynamics). Determine which dataset — or which time/frequency window of each — is valid for each observable, and verify the choice on the data rather than assuming it.

## Mission

1. **Audit both notebooks cell by cell.** For every computation, state what physical quantity it is supposed to represent, what assumptions it silently makes, and whether the implementation actually matches the stated physics (units, normalizations, FFT conventions, axis ordering, grid weights).
2. **Verify that the physics assumptions are faithful** — nothing is accepted because "that's how it was done before."
3. **Clean up the notebooks** into a coherent, documented, reproducible pipeline (backup the originals first; do not modify or delete anything under `data/`).
4. **Converge on a validated broadband-FFT methodology.** Hard constraint: we cannot afford to run many narrow-band simulations. The broadband pulse data is the primary and preferred source; the goal is to extract per-frequency steady states and time-resolved spreading from it **without excessive noise or unphysical artifacts**. Narrow-band runs may only be *recommended* as a minimal spot-check (1–2 frequencies) if the audit finds no other way to validate a step — and then only as a recommendation, not something you launch.

## Hard rules

- **You and your agents can and should check any literature online** (WebSearch/WebFetch): original papers, follow-up critiques, and the Tidy3D documentation. Do not rely on memory for equations, conventions, or claimed results — fetch and read the sources.
- **Every nontrivial claim must be cross-checked by independent agents** before it is accepted: spawn adversarial verifiers whose explicit job is to *refute* the claim (physics-correctness lens, numerical-correctness lens, does-it-reproduce-in-the-actual-data lens). A claim survives only if the refuters fail. If agents disagree, dig deeper until they converge — we must converge to something functional, not merely plausible.
- **No shortcuts and no assumptions before verifying.** Examples: do not assume the fields have decayed by the end of the time window — measure the residual energy; do not assume the source spectrum is flat over the analysis band — compute it; do not assume the monitor aperture is large enough — quantify the truncation bias; do not assume FFT sign/phase conventions match Tidy3D's `exp(-iωt)` convention — check against the docs and against a known reference case.
- Do not launch new Tidy3D cloud simulations (they cost real money). Work only from the existing HDF5 data.
- Report negative results honestly: if a quantity cannot be extracted reliably from the broadband data, say so and quantify why, rather than tuning parameters until a plot looks nice.

## Physics & numerics checklist (verify each — this list is a floor, not a ceiling)

**Steady-state IPR notebook:**
- Exact IPR definition used (2D transverse vs 3D; intensity vs |E|²; normalization; grid-cell weights on a possibly non-uniform mesh). What are its units/dimensions and how does it map to a participation *area* / localization length? Is it grid-resolution dependent, and is that handled?
- Are the "steady states" analyzed actually steady states (frequency-domain responses), and are they obtained consistently with the FFT notebook?
- What does IPR(ν) actually diagnose in an open, absorbing, finite system? A dip in width or transmission is **not** by itself proof of localization (absorption mimics it — see the Sperling/Scheffold/Skipetrov debate). What complementary signatures (σ²(t) saturation, time-resolved profile shape, thickness scaling) are needed before claiming localization?
- The Cherroret 2010 diffusive baseline d(ν) comparison: is it implemented with correct definitions and units?

**FFT / time-domain notebook:**
- Source-spectrum deconvolution: is the response normalized by the source spectrum, and is analysis restricted to bins where source power is above a threshold (e.g., −20 dB from peak)? Quantify the noise amplification at band edges and the sensitivity to the threshold choice.
- Windowing/apodization, zero-padding, record truncation: what leakage/broadening do they introduce, and is frequency resolution Δf = 1/T sufficient for the mode structure being resolved?
- Wrap-around / non-decayed fields: quantify residual field energy at the end of the record and its bias on late-time σ²(t).
- Parseval/energy-conservation check between the time series and the spectrum.
- FFT sign and phase conventions vs Tidy3D's; axis ordering of the stored arrays; real-field vs analytic-signal handling.
- σ²(t) extraction: direct second moment vs Gaussian fit — do they agree? Background/noise-floor subtraction, finite monitor aperture (truncated-profile bias on ⟨r²⟩), ballistic/coherent component removal, ensemble vs speckle averaging (coherent vs incoherent averaging — which is physically correct for this observable?).
- D(ν) extraction: fitting window for the linear regime, uncertainty estimate, consistency between the σ²(t) slope and any steady-state estimate.
- Absorbers vs periodic-boundary datasets (see Context for what each is): quantify the transverse wrap-around contamination in the periodic run (when does the spreading profile first reach the lateral boundary, per frequency?) and the edge-absorption bias in the absorber run; establish per observable which dataset and which time window are trustworthy. Cross-comparing the two runs where both should agree is itself a cheap validation — exploit it.

## Known findings from previous reviews (re-verify they are incorporated; do not regress)

- The FFT sign convention was previously validated; the "floor" operation was found to be a no-op; background subtraction and monitor-aperture truncation were identified as the dominant systematics on σ²(t); wrap-around limits the usable late-time window. Confirm these fixes/caveats are actually reflected in the current notebook state.
- In the IPR notebook, a previous review fixed an `L` bug and a `d_avg` speckle-averaging issue, and established that a CW intensity-width dip alone is not proof of localization. Verify these are still in place.
- A previous estimate gave D ≈ 6 a²/ps at ν = 0.9 in the diffusive regime — use as a sanity anchor (explain any discrepancy; do not force agreement).
- Trap in related data files: in `g_data.h5`-style files, `g` was stored frequency-ascending while `nu` was stored reversed. Check every loaded array for consistent frequency ordering before using it.

## Acceptance tests (must pass before the cleanup is called done)

1. **Cross-notebook consistency:** the FFT-reconstructed steady state at a chosen frequency must quantitatively match the steady-state used in the IPR notebook (e.g., normalized correlation of intensity maps after amplitude normalization), or the discrepancy must be explained and fixed.
2. **Parseval consistency** between time-domain energy and spectral energy within a stated tolerance.
3. **Robustness:** IPR(ν) and σ²(t;ν) must be stable (report the sensitivity) under (a) window choice, (b) dropping the last ~10% of the time record, (c) the source-spectrum threshold. If a result is not robust, it is flagged as unreliable, not reported as physics.
4. **Physical sanity:** diffusive-regime σ²(t) slope consistent with the prior D estimate or with the Cherroret baseline within stated uncertainty.
5. Cleaned notebooks run end-to-end from the raw HDF5 files with no manual steps and no stale hidden state.

## Deliverables

1. `AUDIT_REPORT.md` — all findings ranked by severity (wrong physics > systematic bias > noise/robustness > style), each with the evidence, the fix applied, and the independent-verifier verdict (CONFIRMED/REFUTED). Include the claims that were *checked and found correct*, so we know what was verified.
2. The two cleaned notebooks (originals backed up), with markdown cells documenting every assumption and its verification status.
3. `METHODS.md` — a paper-style methods section for the validated broadband-FFT pipeline, with literature citations, suitable to adapt for a manuscript.
4. A short list of residual risks that cannot be resolved from existing data, each with the cheapest possible test that would resolve it.

## Literature anchors (starting points — verify and extend online; read critiques, not just originals)

- N. Cherroret, S. E. Skipetrov, B. A. van Tiggelen, *Transverse confinement of waves in three-dimensional random media*, PRA **82**, 022125 (2010) — the σ²(t) framework this analysis is built on.
- T. Sperling et al., *Direct determination of the transition to localization of light in 3D*, Nat. Photonics **7**, 48 (2013) — transverse-width method — **and** the follow-ups showing fluorescence/absorption artifacts (Scheffold et al.; T. Sperling et al., New J. Phys. **18**, 013039 (2016); S. E. Skipetrov & J. H. Page, *Red light for Anderson localization*, New J. Phys. **18**, 021001 (2016)).
- T. Schwartz, G. Bartal, S. Fishman, M. Segev, *Transport and Anderson localization in disordered 2D photonic lattices*, Nature **446**, 52 (2007).
- H. Hu, A. Strybulevych, J. H. Page, S. E. Skipetrov, B. A. van Tiggelen, *Localization of ultrasound in a three-dimensional elastic network*, Nat. Phys. **4**, 945 (2008) — time-resolved transverse confinement analysis.
- L. S. Froufe-Pérez, M. Engel, J. J. Sáenz, F. Scheffold, *Band gap formation and Anderson localization in disordered photonic materials with structural correlations*, PNAS **114**, 9570 (2017) — localization in correlated/hyperuniform disorder.
- Reviews: Lagendijk, van Tiggelen & Wiersma, Phys. Today (2009); Segev, Silberberg & Christodoulides, Nat. Photonics (2013); Vynck et al., Rev. Mod. Phys. (2023) on light in correlated disorder.
- Tidy3D documentation: `FieldTimeMonitor`, source normalization and spectrum, apodization, `run_time`/shutoff, field decay, and the frequency-domain convention used when comparing FFTs.

## Orchestration

- Phase 1 — *Understand*: parallel readers over the two target notebooks, the experiment notebook, and the HDF5 structure → structured map of the pipeline and every implicit assumption.
- Phase 2 — *Literature*: parallel agents per theme (transverse localization methodology; FFT-of-pulse best practices in FDTD; absorption-vs-localization artifacts; IPR definitions in open media).
- Phase 3 — *Audit*: finders propose issues; every finding goes to 2–3 adversarial verifiers with distinct lenses; only confirmed findings survive.
- Phase 4 — *Fix & validate*: apply fixes, run acceptance tests on the real data, iterate until two consecutive verification rounds produce no new confirmed issues.
- Phase 5 — *Completeness critic*: a final agent asks "which assumption is still unverified, which claim has no evidence attached, which acceptance test was skipped?" — whatever it finds becomes the next round of work.
