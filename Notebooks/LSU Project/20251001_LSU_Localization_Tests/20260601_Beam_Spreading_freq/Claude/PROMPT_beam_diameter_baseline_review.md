# Prompt — Review the diffusive baseline of `20260806_Beam_Diameter_d_nu.ipynb`

Work in `H:\Codes\tidy3d\Notebooks\LSU Project\20251001_LSU_Localization_Tests\20260601_Beam_Spreading_freq`.

This is a **review + targeted fix**, not a rewrite. The notebook layout (streaming pass, figures,
ipywidgets slider) stays. Three things need attention, in this order:

1. switch the g input to the new externally-computed files,
2. remove the rim-power diagnostic entirely,
3. audit the **σ² diffusive baseline** — it dips **below** the measured σ² in some bands, which is
   suspicious; the d(ν) baseline looks realistic.

**Cross-check everything with agents.** For every physics claim and every code change, spawn at
least two independent subagents — one to re-derive the physics from scratch (no sight of the
notebook's derivation), one to audit the code line-by-line — and reconcile before editing. Do not
trust the notebook's own comments as evidence; several past audits in `Claude/AUDIT_REPORT.md` and
`Claude/STEADY_STATE_IPR_REPORT.md` found exactly this kind of self-consistent-but-wrong baseline.

## 1. New g and ls files (use them as-is — NO scaling anywhere)

The g(ν) values were recomputed **outside this workspace** and dropped in:

```
../data/g_values/n_rod_3.3_ff_0.2249.h5
../data/g_values/n_rod_2.9_ff_0.2249.h5
```

Verified structure (both files, checked 2026-08-25): datasets `nu` and `g`, 300 points each,
`nu` **ascending** 0.0891 → 2.05, already **unitless slab-scale** `ν = a·f/c`. g ranges from ≈ −0.45
at low ν to ≈ 0.88 at high ν.

The ℓ_s(ν) values were **rebuilt 2026-08-25 into the same convention** (agent-verified against the
original runs):

```
../data/ls_values/n_rod_3.3_ff_0.2237.h5    (400 pts,  nu 0.2563 -> 1.0251)
../data/ls_values/n_rod_2.9_ff_0.2237.h5    (1700 pts, nu 0.3200 -> 1.2500)
```

Datasets `nu` and `ls` (µm), `nu` ascending unitless slab-scale, `ls` already at the **physical
slab scale** — the old 0.8a/×1.25 bookkeeping for n = 3.3 is baked in (the numbers are identical
to what the current notebook computes; only the plumbing changes). Root attrs record provenance.

Consequences for `cell-04`:

- Replace the old reads (`n_3.3_ff_0.2237_g_data.h5`, `n_2.90_ff_0.2237_g_data.h5`) with the new
  files. Those old files and any pipeline in this workspace that *generates* g are to be **ignored**
  — the new values are authoritative.
- **Delete the `[::-1]` reversal** — that was for the old n=2.90 file whose ν axis was descending.
  The new files are ascending. Interpolate `g(ν)` directly onto the field ν axis.
- **Delete ALL scale bookkeeping** — no ×0.8 on any ν axis, no ×1.25 on any length, for g **and**
  ls alike. The old handling in `cell-04` (`nu_l = 0.8*a*rf/...`, `ls = 1.25*...`, the per-n
  branches reading the dated ls files and the T(L) frequency source) goes away entirely: both n
  branches now just read `nu`/`ls` and `nu`/`g` from the `n_rod_*` files and interpolate. Do not
  read `20251031_ls_values_n_3p3.h5`, `20260608_ls_values_n_2p9.h5`, or the old `*_g_data.h5`
  files at all.
- **Keep the self-check** `assert GAP[0] <= nu_l[np.argmin(ls)] <= GAP[1]` — it already validated
  the harmonized files (min at ν = 0.4123 for n = 3.3, 0.4437 for n = 2.9, both in-gap).
- The filenames say `ff_0.2249` while the notebook's slab ff (used for Maxwell-Garnett n_eff) is
  0.2237. Keep the slab's measured ff for n_eff; the filename ff belongs to the g calculation's
  nominal geometry. If the physics agent argues these must agree, surface it in the report — don't
  silently change either.
- After the swap, assert coverage (`nu.min() ≤ ν_field.min()` and `≥ ν_field.max()`) and that
  `1 − g > 0` over the analysis band before forming `ℓ* = ℓ_s/(1−g)` (g < 0 at low ν is fine —
  it just makes ℓ* < ℓ_s). Re-print the mid-band ℓ*, z₀, σ²_dif, d_dif numbers and note in the
  report how much they moved vs the committed outputs
  (old mid-band medians: ℓ* = 4.74 µm, z₀ = 6.92 µm, σ²_dif = 185 a², d_dif = 33.8 a).

## 2. Remove the rim power

The "rim power (ρ > 45a)" fraction is currently computed in the streaming pass (`Prim`, `RIM` in
`cell-03`), printed as an aperture-saturation flag, drawn as a dotted twin-axis line in Figures 1–2,
and quoted in the slider title (`cell-07`). It sits where the transmission belongs and it looks odd.
**Remove it everywhere**: the accumulator in the streaming loop, the printouts, the twin axis
(`axr`) in the figure cell, and the slider-title mention. The lower panel of Figures 1–2 shows the
**measured T(ν)** only. Do not replace it with another proxy.

## 3. Audit the σ² diffusive baseline

Symptom: in some frequency ranges the plotted `σ²_dif(ν)` falls **below** the measured σ²(ν).
For a slab at/near the diffusion–localization crossover the measured transverse spread should sit
**at or below** the fully-diffusive prediction (localization, absorption, and finite-aperture
truncation all *reduce* the measured spread) — so measured > baseline points at a baseline error,
an estimator mismatch, or a wrong ℓ*(ν) input. The d(ν) baseline does *not* show the problem.

Current implementation (`cell-04`):

```python
def sig2_dif(lt, z0=0.0):     # closed form
    return ((2/3)*((L_slab + 2*z0)**2 - (lt + z0)**2) - 2*z0**2)/a**2
def dif_pair(lt, z0=0.0, nr=1200):
    # numeric: T(ρ) from the extrapolated-boundary kernel
    # T(q) = sinh(q(l*+z0)) cosh(q z0)/sinh(q(L+2 z0)),
    # pushed through the SAME estimators and grid as the data
```

Note the asymmetry: **d_dif goes through `dif_pair` (estimator-matched), σ²_dif uses the closed
form.** That alone is a prime suspect for why one baseline looks right and the other doesn't.

Have the agents check, at minimum — independently, before comparing notes:

- **Re-derive the small-q expansion** of the extrapolated-boundary kernel from scratch and confirm
  or refute `σ² = (2/3)[(L+2z₀)² − (ℓ*+z₀)²] − 2z₀²`. Check the z₀ = 0 limit against Cherroret
  PRE **82**, 056603 Eq. (3). Watch the usual convention traps: ⟨ρ²⟩ (2-D transverse) vs per-axis
  ⟨x²⟩ (factor 2), and the sign/coefficient of every z₀ term.
- **Estimator consistency**: compute σ²_dif numerically via `dif_pair` on the same radial grid,
  aperture, and weighting as the measurement, and compare with the closed form bin-by-bin. The
  measured σ² is truncated at the monitor aperture; an infinite-aperture closed form and a
  truncated measurement are not comparable when T(ρ) has heavy tails (σ²_dif here is
  O((L+2z₀)²) with L+2z₀ ≈ 46 µm ≈ 18 a → tails matter). Decide which version to *plot* — the
  estimator-matched one — and keep the closed form only as a printed sanity number, clearly
  labelled.
- **Speckle bias bookkeeping**: the notebook applies a ×1.15–1.20 single-shot ensemble correction
  to d but quotes σ² "speckle-unbiased". Verify the two observables are treated consistently and
  that no bias correction is applied to one side of a comparison but not the other.
- **Input sanity**: after the g swap (§1), re-check ℓ*(ν) = ℓ_s/(1−g) band-by-band. If ℓ* is
  underestimated anywhere (e.g. from interpolating across the gap where g → 1 noisily), σ²_dif is
  pulled down through the −(ℓ*+z₀)² term and can artificially undercut the data. Mask bins where
  ℓ* > L/4 (already done) *and* where the g interpolation is extrapolating or 1−g is within noise
  of zero.
- **z₀ chain**: R_internal via Zhu–Pine–Weitz on MG n_eff, z₀ = (2/3)ℓ*(1+R)/(1−R) — re-verify the
  Haskell 1994 anchor assert still passes and that z₀ enters σ²_dif and d_dif with the same
  convention.

Deliverable for §3: a short written verdict in the notebook header markdown (2–4 sentences: what
was wrong, what the baseline now is), corrected `cell-04`/figure code, and updated printed
comparison numbers. If the audit concludes the baseline was actually *correct* and the crossing is
physical, say so explicitly with the derivation — do not "fix" a non-bug.

## Execution notes

- Python: only `C:\Users\HernandF\AppData\Local\Programs\Python\Python312\python.exe` has tidy3d,
  ipywidgets, plotly. Run the notebook headlessly with nbconvert from that env after editing;
  the streaming pass over the 9.5 GB h5 takes a while — use a generous timeout and run it in the
  background.
- Focus on the `n = 2.90` dataset first (`LSU_20260804_slab_250x250x32_n_2p9_background_effective_n_1.00.h5`),
  then confirm n = 3.3 runs clean with its (unchanged) ls conventions.
- Update `README.md` (this folder) to reflect: new g source files, rim power removed, baseline fix.
- Report at the end: per-item what changed, the before/after mid-band σ²_dif and d_dif, and the
  agents' independent verdicts (agree/disagree and how it was resolved).
