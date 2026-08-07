# Prompt — Lean d(ν) notebook + README for the NEW fully-random n = 3.3 slab (250×250×32)

Work in `H:\Codes\tidy3d\Notebooks\LSU Project\20251001_LSU_Localization_Tests\20260601_Beam_Spreading_freq`.

A new numerical experiment replaces the old 12×-tiled SHU run: a **fully random (LSU-generated,
healed cylinder network) slab, 250×250×32 µm, n = 3.3 (ε = 10.89)**, designed so the transverse
aperture and thickness can capture most localization lengths expected close to the gap. The data is
already downloaded. The job this session:

1. Replace `20251008_IPR_Calculation.ipynb` with a **new, much leaner notebook** that redoes the
   steady-state d(ν) analysis on the new data.
2. Add an **interactive exit-face beam-map cell** (frequency slider, linear + log color maps,
   normalized to the max).
3. Move all the physics prose out of the notebook into a **`README.md`** at the folder root.
4. **Delete the old notebook** once the new one runs end-to-end.
5. **Verify every physics claim with adversarial agents** (mandatory rule below).

## Mandatory working rule — verify all physics claims with agents

Every nontrivial physics claim — carried over from the old analysis, newly made this session, or
stated in this prompt (including the design claim in "Physics tasks" below) — must be checked by
independent adversarial verifier agents whose explicit job is to *refute* it (physics-correctness
lens, numerical-correctness lens, does-it-reproduce-in-the-actual-data lens). A claim goes into the
notebook or README only with its verdict (CONFIRMED / PLAUSIBLE / REFUTED) settled. Run refuters
*before* writing conclusions, not after — a previous session's adversarial pass caught a headline
bug. Keep fan-outs small (≤3–4 agents per wave; several small waves), the session has hit usage
limits with large fleets before. Cheap arithmetic can be checked inline without agents.

## The new experiment — hard facts (from the launch script and the h5; re-verifiable, not claims)

Launch script: `20251001_numerical_experiment_using_td_cylinders.py` (folder root).
Raw cloud data: `../../../../data/20260805_Beam_Spreading_250_250_32/n_3.30`; downloaded via
`20251007_Retireve_Field_Data.ipynb` into
**`data/slab_250x250x32/LSU_20260804_slab_250x250x32_n_3p3__lsu_generated_healed_lambda_4_8.h5`** (4.4 GB).

- Structure: point file `20260804_slab_200x200x32_N854932_lsu_generated_healed` (coordinates ÷0.8
  → 250×250 µm transverse), cylinders of radius 0.42 µm, ε = 3.3² = 10.89, trimmed to
  |z_center| ≤ 16.8 µm → slab thickness **L = 32 µm = 12.49 a** (a = 2.562629142772549 µm, kept
  purely as the length/frequency unit convention). Transverse extent 250 µm = 97.6 a — **no tiling**:
  unlike the old 12× SHU run, the disorder is aperiodic across the full aperture, so the old
  "Bloch-periodic beyond one tile" caveat is gone. Confirm this from the point file, then retire
  that caveat explicitly.
- Boundaries: **absorbers on all six sides** (200 layers) — same convention as the old
  `_absorbers` file (edge-loss bias possible, no wrap-around).
- Source: **5×5 µm plane-wave patch** (not a Gaussian beam) at z = −19 µm, 3 µm before the input
  face; Gaussian pulse spanning λ = 8→4 µm; run_time 40 ps, shutoff 1e-20; `normalize_index=None`
  (raw DFT — still carries the source envelope; every per-bin scale-invariant estimator cancels it,
  same as before).
- Exit-face FieldMonitor at z = +16 µm, `interval_space=(4,4,4)`. In the h5 (group key `"3.30"`):
  `Ex,Ey,Ez` shape **(363, 363, 1, 1500)** complex64, `x,y` ∈ [−125.17, +125.17] µm with mean
  dx ≈ 0.69 µm on a **non-uniform** grid (Simpson on the true grid, as before), `f` ascending with
  **ν = a f/c ∈ [0.3203, 0.6407], δν = 2.137×10⁻⁴** (≈ 40 ps window).
- Design gap for this n = 3.3 family (from the 2026-07 campaign; **verify empirically**):
  ν ≈ [0.388, 0.435]; ξ target 5–9 a near the gap edge → L/ξ ≈ 1.4–2.5.

## Traps left in the workspace — do not carry these over

`20251008_IPR_Calculation.ipynb` cell 2 was half-updated (2026-08-06) to point at the new h5 but
still contains **stale constants that are wrong for this run**:

- `L_slab = 14.3` → must become **32.0** everywhere (baseline, any L-dependent formula).
- `data_g`  load **`n_3.3_ff_0.2237_g_data.h5`** — This one is faithful, it's been generated outside this workspace for a statistically similar structure, so do not overthink about it. 
- `data_ls` points at `../20251002_Ls_test/data/ls_values/20251031_ls_values_n_3p3.h5` — n = 3.3,
   **it was computed for this structure family** (fill fraction / radius / network type),
- The MPB gap line (`gap_data["Circular"]["0.22"]...` scaled by 14.3) is the periodic circular-rod
  crystal — not this random network. Locate the gap empirically from the S(ν), it's likely to have the same position as this structure is statistically similar (circular rods, ff and n)

`bst_pipeline.py` (folder root) holds the validated streaming/estimator primitives — reuse them.
Also read first: `Claude/STEADY_STATE_IPR_REPORT.md`, `Claude/METHODS.md` (if present), memories
`lsu-steady-state-ipr-deepdive-2026-07`, `lsu-ipr-fft-audit-2026-07`,
`lsu-beam-spreading-conventions`. Carried-over context: verify, don't assume.

## Deliverable 1 — the new notebook (lean)

Create `20260806_Beam_Diameter_d_nu.ipynb`. Target **≤ ~10 code cells, one-line markdown headers
only** — all physics prose goes to the README. Suggested flow:

1. Imports + constants (a, L = 32, paths, gap window once verified).
2. Data load (`AM.read_hdf5_as_dict` or lazy `h5py` — the full file is ~4.7 GB in RAM, acceptable).
3. Streaming pass (adapted from the validated cell / `bst_pipeline.py`): d(ν) (PR diameter,
   primary), S(ν) (transmitted power), ⟨ρ²⟩(ν) (background-sensitive, secondary), d_avg boxcars,
   core-pixel spectra for N_eff. Keep the analytic unit test (PR diameter of a synthetic Gaussian
   against its closed form) as a cheap assert.
4. Speckle N_eff → error band on d(ν).
5. Diffusive baseline d_dif(ν) with the honest Cherroret profile at L = 32 µm and the scanned ℓ*.
6. Gap dip: locate/quantify the d(ν) dip and the S(ν) dip; state clearly they need not coincide.
7. Summary figure: d(ν) ± error band vs d_dif(ν), gap band shaded, S(ν) in a lower panel.
8. **Interactive beam-map cell (Deliverable 2).**

No FFT/time-domain work: `20260602_IPR_Calculation_FFT.ipynb` is out of scope (it still refers to
the old 12× data — leave it untouched).

## Deliverable 2 — interactive exit-face beam map

One cell, `ipywidgets` slider over frequency (readout showing both ν and λ, snapping to bins):
side-by-side **linear** and **log** color maps of I = |Ex|²+|Ey|²+|Ez|² at the exit face,
**each normalized to that frequency's map maximum** (lin: [0,1]; log: log₁₀(I/I_max) with a floor
around 1e−5…1e−6). Equal aspect, axes in µm with a secondary scale (or tick relabel) in units of a,
slab footprint / monitor edge marked, gap window indicated (e.g. slider description or a thin S(ν)
locator strip). Must stay responsive: slice one bin per redraw from the in-memory array (or lazy
h5py) — never materialize I(x,y,ν) for all 1500 bins.

## Physics tasks (each claim → adversarial verification, verdicts recorded in the README)

1. **The design claim (from the user):** "a fully random 250×250×32 µm slab at n = 3.3 can capture
   most localization lengths close to the gap." Quantify: with ξ ∈ 5–9 a, L/ξ ≈ 1.4–2.5 — is the
   exponential z-confinement measurable at this thickness? Transversely, the un-tiled ±48.8 a
   aperture vs the CW σ² ≃ 2ξ_eff·L scale — which ξ range saturates or escapes the aperture? State
   for which ξ the claim FAILS (e.g. ξ ≳ L/2 indistinguishable from diffusive at a single L).
2. **Standing guardrail (keep unless refuted):** a CW width dip alone cannot prove localization —
   absorption / gap Bragg attenuation produce the same σ² ≃ 2LL_a form (Cherroret PRE 82, 056603);
   a single L still yields L_att, not ξ. Any dip inversion must be labeled L_att.
3. **Gap location** for THIS structure from S(ν), vs the design window [0.388, 0.435].
4. **Speckle sampling at high ν:** mean dx ≈ 0.69 µm vs speckle grain ~λ/2 (= 2 µm at ν = 0.64 in
   air) — is ∫I² (the PR denominator) converged on this grid? Check by decimating the maps 2×/3×
   and recomputing d(ν).
5. **Edge-loss bias** of the all-absorber transverse boundaries: quantify intensity at the map rim
   vs center per ν band; flag ν ranges where rim power is non-negligible.
6. **d_avg convergence:** the old data's d_avg rose monotonically with boxcar width (never
   converged). Re-test on the new data — the honest aperture may fix it. Pick and justify the final
   estimator + error bar.
7. **Source realism:** the 5×5 µm plane-wave patch has hard edges (sinc sidelobes in k-space) —
   check whether sidelobes contaminate the exit-face wings at weakly-scattering ν (e.g. below-gap
   quasi-ballistic bins) and document the impact on d(ν).

## Deliverable 3 — README.md (folder root)

The physics and notes that used to live in dense notebook markdown, written for future-you:
experiment description (geometry, source, BCs, monitor, units, h5 layout table, file inventory
incl. which script produced what), the d(ν) estimator and why (PR diameter, scale invariance,
speckle low-bias √(1+1/N_eff), N_eff from δν_c), the diffusive baseline and its validity limits,
interpretation rules (what CW d(ν) can and cannot prove), known systematics (edge loss, source
sidelobes, ℓ* uncertainty, sampling), and a claims table with each adversarial verdict. Keep
deeper working documents in `Claude/` as usual; the README is the user-facing summary.

## Deliverable 4 — retire the old notebook

After the new notebook runs end-to-end (restart-and-run-all clean) and the README exists, **delete
`20251008_IPR_Calculation.ipynb`** from the folder root. Backups already exist
(`backups_20260716/20251008_IPR_Calculation.ipynb.bak`,
`Claude/audit_workspace/steady/20251008_IPR_Calculation_BACKUP_20260717_predeepdive.ipynb`) and git
history retains it — do not create another copy. Leave committing to the user.

## Hard rules

- **No new Tidy3D cloud runs** (cost money). Work from the existing HDF5 only.
- Do not modify anything under `data/`.
- Verify physics/conventions against primary sources online (Cherroret PRE 82 056603; speckle /
  participation-ratio literature) — not from memory.
- Report negative results honestly; a non-robust feature gets flagged, not interpreted.
- Update the persistent memory files if any carried-over fact proves wrong on this data
  (in particular anything asserting the 12×-tiling caps
  ).
- Cross check all findings and claims with agents before declaring success. 
- Keep the workspace clean and organized, do not put test files or clutter in the root directory 
- You are in an institutional network with access to many journals. Take avantage of it, consult literature and papers as needed. 