# Prompt — Time-domain first look: σ²(t) and d(t) for the random n=3.3 slab (250×250×32)

Work in `H:\Codes\tidy3d\Notebooks\LSU Project\20251001_LSU_Localization_Tests\20260601_Beam_Spreading_freq`.

Follow-up to the CW session of 2026-08-07 (commit `72480c7`; read `README.md` first — it has
the experiment facts, estimator conventions, validity masks, and the 12-claim adversarial
verdicts table). The job now is the **time-domain** view of the same h5:

1. Replace `20260602_IPR_Calculation_FFT.ipynb` with a **new, much leaner notebook** that
   reconstructs narrow-band time-domain intensity at the exit face and shows **just two
   plots**: σ²(t) and the PR diameter d(t), horizontal axis = time in **ps**, nothing fancy.
2. **No localization-length extraction, no D fits, no ξ.** We only want to *observe the
   curves* this session. Saturation levels, growth slopes etc. get eyeballed, not fitted.
3. Last cell: **movies of the exit-face profile in time** (a few selected bands), saved to
   `movies/`.
4. Delete the old FFT notebook once the new one runs end-to-end; update `README.md` with a
   short time-domain section.

## Data & conventions (all verified 2026-08-07 — reuse, do not re-derive)

- h5: `data/slab_250x250x32/LSU_20260804_slab_250x250x32_n_3p3__lsu_generated_healed_lambda_4_8.h5`,
  group `"3.30"`: `Ex,Ey,Ez` (363,363,1,1500) complex64, non-uniform `x,y` ∈ ±125.17 µm
  (±48.85 a), `f` ascending, ν = a f/c ∈ [0.3203, 0.6407], δν = 2.137×10⁻⁴, df = 25.0 GHz
  ⇒ **T = 1/df = 40.0 ps, dt = T/1500 = 26.7 fs**. a = 2.562629142772549 µm, L = 32 µm =
  12.487 a. Raw DFT (`normalize_index=None`) — carries the source envelope; all per-frame
  ratio estimators (σ², d) are insensitive to it; **no deconvolution** (double-deconv trap,
  see `Claude/METHODS.md` §2).
- **FFT sign convention (audit-validated):** Tidy3D stores Ê ∝ Σ E e^{+2πift}, so
  `scipy.fft.fft` along the frequency axis reconstructs the **causal** E(t) on t_n = n·dt
  (`scipy.fft.ifft` would time-mirror it). See `bst_pipeline.py` docstring.
- **Windows:** Gaussian spectral windows of **bandwidth 0.02 in ν** (take FWHM = 0.02,
  i.e. σ_ν ≈ 0.0085 ⇒ pulse duration t_p ≈ 0.16 ps ≪ any transport time; ≈94 bins per
  FWHM vs intensity spectral-correlation FWHM ≈ 4.6 bins ⇒ N_eff ~ 20, speckle bias on d
  is ~1% — ignorable for curve-watching). Window centres: the 0.02 grid across
  [0.33, 0.63], **plus** two windows centred on the CW flank dips ν = 0.389 and 0.437.
- **Estimator code:** `bst_pipeline.reconstruct_windows` is the validated streaming pattern
  (one pass, per-window P(t) = ∫I dA, M2(t) = ∫ρ²I dA, I2(t) = ∫I² dA, Parseval check,
  never materializes a movie) — but its defaults are tuned to the OLD 14.3 µm / ±33.5 a
  dataset (`L_SLAB_UM=14.3`, `bg_rho_list=(25,28,30)`, `edge_rho=28`, tiling notes).
  Either call it with new-aperture parameters (rim/edge at 40–45 a) or write a fresh lean
  cell modeled on it. σ²(t) = M2/P in a² about the injection axis (0,0);
  d(t) = 2√((P²/I2)/π)/a. Simpson weights on the true non-uniform grid; rescale fields
  ×1e16 at read (float32 ∫I² underflow).
- **Environment:** the `cassegrain` conda env runs everything; the notebook's `python3`
  kernel resolves `python` from PATH. Execute headless with:
  `PATH="/c/Users/HernandF/.conda/envs/cassegrain:...Scripts:$PATH" python -m jupyter nbconvert --to notebook --execute --inplace <nb>`.
  Full file ≈ 4.7 GB in RAM via `AM.read_hdf5_as_dict` — fine (137 GB machine); stream in
  row-blocks for the per-window FFTs.

## Traps — check these before trusting any curve

- **Wrap-around at the record end.** run_time (40 ps) ≈ T (40.001 ps): if the field had not
  fully decayed at shutoff, late-time signal folds back to t ≈ 0⁺. Diagnose from P(t):
  floor level at large t, and whether P(t→T) merges into the floor. Only trust t below a
  cutoff you determine from the data; state it in the notebook (one printed line).
- **Deep-gap windows are floor/rim-dominated** (CW result: S_rel ≈ 10⁻⁵, rim ρ>45a carries
  up to 29% of power in [0.392, 0.430]). Windows centred there (0.41, 0.43) measure the
  floor, not transport — skip them or plot them visibly flagged (dashed/grey).
- **Flank windows blend across steep S(ν).** A 0.02-wide window centred at 0.389 or 0.437
  spans −50 dB and −10 dB bins simultaneously; the reconstruction is dominated by the
  brightest bins in the window (the CW d_avg-in-gap lesson). Qualitative reading only.
- **σ²(t) is background/rim-sensitive** (CW: full-aperture ⟨ρ²⟩ ≈ 2× d²/8 even mid-band) —
  that is exactly why BOTH σ²(t) (Cherroret's observable) and the PR d(t) (background-
  robust) are plotted. Use the rim-power fraction (ρ > 45 a) as the artifact clock: fade or
  cut each curve after rim > ~2% (halo reaching the absorbers = edge loss, not transport).
- **Growth lives on the pre-peak rising edge** (2026-07 audit lesson: early-arriving light
  has spread less; σ² grows *through* the power peak). Don't misread the post-peak region.
- The un-tiled geometry means the old σ² ≲ 30 a² tiling cap is GONE: saturation anywhere up
  to the aperture scale is now potentially physical. The interesting scale from the CW
  session: σ²_loc(ξ = 5–9 a, full Cherroret Eq. 6) = 63–77 a², vs diffusive
  (2/3)L² = 104 a² and the ~(W/6)² aperture ceiling. **Observe only — no extraction.**

## Deliverable 1 — the new notebook (lean)

Create `20260808_Beam_Spreading_time.ipynb` (date = run date). Target **≤ ~7 code cells,
one-line markdown headers**; physics prose goes into README, not the notebook:

1. Imports + constants + paths (copy the constants cell pattern from
   `20260806_Beam_Diameter_d_nu.ipynb`).
2. Load + axes + Simpson weights + quick sanity asserts (causal-reconstruction/Parseval
   check per window is cheap — bst_pipeline computes the Parseval ratio; assert ≈ 1).
3. Window set + one streaming pass accumulating per window: P(t), M2(t), I2(t), rim
   power(t). Print the wrap-around/floor diagnostic and the chosen t-cutoff.
4. **Plot 1: σ²(t) [a²] vs t [ps]** — all valid windows, colored by ν (colorbar), deep-gap
   windows flagged or absent, curves faded after the rim-clock trips. Plain axes.
5. **Plot 2: d(t) [a] vs t [ps]** — same layout.
6. (Optional, only if it stays lean) small P(t) reference panel — helps read arrival times.
7. **Movie cell (last):** for ~4 selected windows (below-gap 0.35, lower flank 0.389, upper
   flank 0.437, mid-band 0.55), animate the exit-face I_W(x,y,t): log₁₀(I/I_max,global)
   with a 10⁻⁵…10⁻⁶ floor, fixed normalization across frames, equal aspect, axes in µm,
   ν and t [ps] in the title. Decimate time (every ~5th frame ⇒ ~300 frames) and stream
   the frames (a full movie array is ~800 MB float32 per window — decimated ~160 MB is
   fine). Save `movies/beam_spreading_nu0p350.gif` (or .mp4 if ffmpeg is available in the
   env — check; imageio/Pillow gif is the safe fallback).

## Deliverable 2 — README update

Add a short "Time-domain first look" section to `README.md`: window convention, the
t-cutoff and rim-fade rules, which windows are excluded and why, pointers to the movies,
and 3–5 bullet observations of what the curves show (growth, saturation-or-not, gap-window
floor behaviour) — **descriptive language only** ("σ² grows then flattens near X a²"),
no ξ, no D, no localization claims. Any statement beyond pure description must go through
a small adversarial wave first (≤3 agents: reconstruction-correctness lens, artifact lens
(wrap/edge/floor), does-it-reproduce lens) — same working rule as the CW session; record
verdicts in the README claims table (rows T1, T2, …).

## Deliverable 3 — retire the old FFT notebook

After the new notebook runs end-to-end (restart-and-run-all via nbconvert, zero errors),
**delete `20260602_IPR_Calculation_FFT.ipynb`**. NOTE: the working-tree copy contains
uncommitted half-migration edits (the `files` dict was repointed at the new h5 on
~2026-08-06; execution counts bumped) — these are superseded by the new notebook; the
committed version is in git history and `backups_20260716/20260602_IPR_Calculation_FFT.ipynb.bak`
holds the old-data version. Verify nothing else of value was added to the working copy
(diff it) before deleting. Do not create another backup. Leave committing to the user.

## Hard rules (unchanged from the CW session)

- **No new Tidy3D cloud runs.** Work from the existing h5 only. Do not modify `data/`.
- Keep the workspace clean: scripts/npz into `Claude/audit_workspace/random_slab_td/`
  (audit_workspace is gitignored), movies into `movies/`, nothing loose in the root.
- Reuse validated conventions (`bst_pipeline.py`, README §2–4) — verify, don't re-derive;
  but treat any OLD-dataset-specific number (14.3, ±33.5 a, tiling caps, 15.62 ps fold) as
  a trap.
- Update the persistent memories if any carried-over fact proves wrong on this data.
- Report negative results honestly; a non-robust feature gets flagged, not interpreted.
