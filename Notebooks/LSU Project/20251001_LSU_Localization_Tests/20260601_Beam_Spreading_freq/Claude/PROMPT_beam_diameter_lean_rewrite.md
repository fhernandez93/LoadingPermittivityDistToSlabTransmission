# Prompt — Rewrite `20260806_Beam_Diameter_d_nu.ipynb` as a simple σ²(ν) / d(ν) notebook

Work in `H:\Codes\tidy3d\Notebooks\LSU Project\20251001_LSU_Localization_Tests\20260601_Beam_Spreading_freq`.

## Goal

Replace `20260806_Beam_Diameter_d_nu.ipynb` (currently ~13 cells, heavy with diagnostics and
self-derived calibrations) with a **simple** notebook — target **≤ 8 cells** — that measures the
steady-state transverse spread of the transmitted beam on the exit face and shows it against a
diffusive baseline and the *measured* transmission spectrum.

The single biggest simplification: **stop deriving the gap and the transmission proxy from the field
data itself.** The old notebook fitted a 7th-order polynomial envelope to `S(ν) = ∫I dA`, called the
ratio a "relative transmission", located the gap from a contiguous −20 dB run, and built a validity
mask out of it. All of that goes away. We already have both quantities measured independently:

```python
gap_data          = AM.read_hdf5_as_dict(r"../../20250630 MPB Bands analysis/Data/gap_data.hdf5")
transmission_data = AM.read_hdf5_as_dict("./data/slab_250x250x32/Transmission/LSU_20260804_slab_250x250x32_Transmission_background_n_1.00.h5")
```

Analysis files (exit-face fields):

```python
file_data =  "./data/slab_250x250x32/LSU_20260804_slab_250x250x32_n_3p3_backdround_effective_n_1.00.h5"
file_data = "./data/slab_250x250x32/LSU_20260804_slab_250x250x32_n_2p9_background_effective_n_1.00.h5" This one is the most relevant focus your tests here! 
file_data = "./data/slab_250x250x32/LSU_20260804_slab_250x250x32_n_2p4_background_effective_n_1.00.h5"
```

## Deliverables

1. **Figure 1** — `σ²(ν)` (second moment of the exit-face intensity about the injection axis, in
   units of `a²`), the **diffusive baseline** `σ²_dif`, and the **measured transmission** `T(ν)`
   in a shared-x lower panel. MPB gap band shaded.
2. **Figure 2** — identical, with the **PR diameter `d(ν)`** (in `a`) and `d_dif` in place of σ².
3. **Keep the last cell**: the interactive `ipywidgets` frequency-slider exit-face map
   (linear + log₁₀, per-frequency normalized). Only change its locator strip: it currently plots
   `S_rel`; it should plot the measured `T(ν)` (or, if `T` is unavailable for the chosen `n`, just the
   ν-axis with the MPB gap band and the current-bin marker).
4. Add a new cell that generates a  Exit-face speckle point cloud in $(x, y, \nu)$ line in the last cell of 20260808_Beam_Spreading_time

## Verified facts about the data (checked this session — do not re-derive, but do assert)

`a = 2.562629142772549` µm, `L_slab = 32.0` µm = 12.487 a, `ν = a·f/c`.

**Field files** `LSU_20260804_slab_250x250x32_n_2p9_backdround_effective_n_1.00.h5` (9.5 GB on disk):
- one group, key `"2.90"`; datasets `Ex, Ey, Ez`, plus
  `x, y` ( **non-uniform** grid → Simpson weights on the true grid, as before), `z` (1),
  `f`.
- **ν ∈ [0.330, 0.500]**, δf ≈ 12.49 GHz. Note this differs from the old notebook's file
  (1500 bins, 363², ν ∈ [0.320, 0.641]) — every hard-coded axis assumption must be re-read from the h5.
- `AM.read_hdf5_as_dict` is **not lazy**: it materializes all three field arrays (~9.5 GB) in RAM.
  Keep the existing blocked row-streaming pattern (`BL ≈ 33` rows per block, `gc.collect()` per block)
  and do **one** pass.

**Gap file** — the n = 2.9 entry exists and reproduces the known window:
```python
n_index = "2.90"
gk = gap_data["Circular"]["0.22"]
i  = np.where(np.isclose(gk["n"], float(n_index)))[0]          # use isclose, not ==
GAP = tuple(a/(14.3/gk["gap_edges"][i].flatten()))             # → (0.38775, 0.43448)
```
(Same conversion as `20260808_Beam_Spreading_time.ipynb` cell 1 — copy it verbatim, but switch the
exact float comparison to `np.isclose`.) Cross-check: the old notebook quotes MPB `[0.388, 0.434]`.

**Transmission file — BLOCKER, read this before writing code.** The h5 contains **only groups
`"2.40"` and `"2.90"`. There is no `"3.30"` group.** Confirmed at the source too: the raw cloud folder
`H:\Codes\tidy3d\data\20260813_Beam_Spreading_250_250_32_transmission` holds only `n_2.40` and
`n_2.90` — the n = 3.3 transmission run has not been executed/retrieved.

- `"2.40"`: 200 bins, ν ∈ [0.470, 0.530], T ∈ [9.2e-3, 9.7e-2]
- `"2.90"`: 250 bins, ν ∈ [0.390, 0.500], T ∈ [1.9e-6, 1.6e-1] (gap clearly resolved)

Therefore: write the notebook **parameterized by `n_index`**, with a small dict mapping `n_index` →
field-file path (all three n = 2.40 / 2.90 / 3.30 field files are present in
`data/slab_250x250x32/`). Look up `transmission_data.get(n_index)`; if absent, **print a clear
warning, leave the transmission panel empty (or hide it), and carry on** — do not fabricate a
substitute and do not silently fall back to another `n`. Default `n_index = "3.30"` per the request.
Flag in the summary that Figure 1/2's transmission panel will stay empty until the n = 3.3
transmission run is done (via `20251001_..._transmission.py` → `20251007_Retireve_Transmission_Data.ipynb`).

Do **not** interpolate `T(ν)` onto the field ν-axis — the supports differ (e.g. n = 2.90 transmission
covers 0.39–0.50 while the fields run 0.33–0.50). Plot `T` on its own ν axis, shared x-limits only.

## Estimators (one streaming pass, nothing else)

Per frequency bin, accumulate three integrals over the exit face with product-Simpson weights:

```
S1 = ∫ I dA        S2 = ∫ I² dA        M2 = ∫ ρ² I dA        (I = |Ex|²+|Ey|²+|Ez|²)
σ²(ν) = (M2/S1)/a²                     d(ν) = 2√((S1²/S2)/π)/a      (participation-ratio diameter)
```
Both are per-bin scale-invariant, so the raw-DFT source envelope (`normalize_index=None`) cancels.
Keep the Gaussian unit test (`d` of `exp(-ρ²/2s²)` is exactly `4s`) and add the matching σ² test
(`σ² = 2s²`). Optionally also accumulate the rim power fraction `∫_{ρ>45a} I dA / S1` — it is nearly
free in the same pass and is the one diagnostic worth keeping, as an aperture-saturation warning.

**Delete outright**: the `WBOX` boxcar `d_avg` family, the 2×-decimation check, the ρ<40a
sub-aperture, the core-pixel speckle-contrast cell, the `file_ls`/`file_nuls`/`file_g` loading and
`kℓ*` cell, the S-envelope/gap-finding/validity-mask cell, and the two flank-dip quasi-mode blocks.
Their conclusions are already recorded in `README.md` and the `Claude/` reports; the notebook does not
need to re-derive them.

## Diffusive baseline — closed form, no ℓ*/g files needed

Use the extrapolated-boundary slab kernel already in the old cell 7,
`T(q) = sinh(q(ℓ*+z₀))·cosh(q z₀)/sinh(q(L+2z₀))`. Small-q expansion gives an **exact closed form for
the second moment**:

```
σ²_dif = (2/3)[(L + 2z₀)² − (ℓ* + z₀)²] − 2 z₀²
```

so with `ℓ* ≪ L`, `z₀ = 0`: **σ²_dif ≈ (2/3)L² ≈ 104 a²** — essentially flat in ν and independent of
ℓ* to leading order. Verify this algebra numerically against a direct Hankel transform of the same
kernel before using it (they must agree to a few 1e-3 for `ℓ*/L ≲ 0.25`); if they disagree, trust the
numerics and say so.

For the same reason the transmitted *profile shape* — hence the PR diameter — is also ℓ*-independent
to leading order (`T(q) → (ℓ*/L)·qL/sinh(qL)`), so `d_dif` needs **one** Hankel transform → `T(ρ)` →
`diameter()`, not a per-bin scan. Draw both baselines as horizontal lines with a shaded systematic
band spanning a stated `(ℓ*, z₀)` range: use `L/ℓ* ∈ [4, 10]` and `z₀/ℓ* ∈ [1, 1.5]` (the MG-like
`n_eff ≈ 1.27` value the data already constrained — see the `README.md` / memory note; the
`z₀ = 4ℓ*` vol-ε value was excluded). Hard-code these as visible scalar constants at the top; do not
reload the ℓ*/g files.

## Guardrails that must survive into the new notebook (short comments, not essays)

- `d(ν)` on a **single realization** is biased **low** by ≈ ×1.15–1.20 (ensemble = raw × that band);
  σ² has the opposite sensitivity — it is **background- and aperture-sensitive**, unlike the
  background-robust PR diameter. State which is which in one line each.
- Inside the MPB gap band the exit-face signal is at the numerical floor: plot those bins in a muted
  color and label them **floor-dominated — not a beam width**. Do not threshold-fit anything to decide
  this; the shaded MPB band is the statement.
- A CW width dip near a gap edge gives an attenuation/confinement scale, **not ξ** (absorption and
  Bragg attenuation produce the same `σ² ~ 2 L L_att` form; a single L cannot promote it to ξ).
- The MPB gap is a **design/reference** window for the periodic parent lattice, not a measurement on
  this aperiodic slab — say so where it is shaded.

## Working rules

- Physics claims added or reworded must be checked by small adversarial verifier waves (≤3–4 agents,
  several waves — large fleets have hit usage limits here) before they go in the notebook. Cheap
  arithmetic can be verified inline. The closed-form σ²_dif above is *my* derivation — treat it as a
  claim to refute, not a given.
- Copy the current working-tree `20260806_Beam_Diameter_d_nu.ipynb` into `backups_20260824/` before
  overwriting (it has uncommitted modifications, so git alone is not a safe net), then write the new
  notebook to the **same path**.
- Run it end-to-end on `n_index = "3.30"` before declaring done. Note the full-pass runtime; if the
  1593-bin pass over 9.5 GB is slow, report the wall-clock rather than silently subsampling.
- Keep stored outputs light — the file being replaced is 784 KB of embedded output. Clear outputs of
  the streaming/diagnostic cells; keep the two figures.
- Prose belongs in `README.md`, not in the notebook. Update the README section describing this
  notebook to match the new, smaller scope (and note the retired S-envelope gap-finding machinery).

## Acceptance checks

- [ ] Gaussian unit tests pass for both `d` and `σ²`.
- [ ] `GAP` from `gap_data` reproduces `(0.3877, 0.4345)` for n = 3.30.
- [ ] Closed-form `σ²_dif` agrees with the numerical Hankel transform.
- [ ] Both figures render with the gap band shaded, baseline + band drawn, and either the measured
      `T(ν)` panel or an explicit "no transmission run for this n" note.
- [ ] The interactive slider cell still works against the new arrays.
- [ ] Mid-band (ν ≈ 0.46–0.50, outside the gap) `σ²` and `d` are reported next to their baselines,
      with the ×1.15–1.20 single-realization bias stated for `d`.
