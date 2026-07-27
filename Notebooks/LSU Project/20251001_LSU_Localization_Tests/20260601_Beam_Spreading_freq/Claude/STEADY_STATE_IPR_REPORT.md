# Steady-state participation-ratio beam diameter d(ν) — deep-dive report

**Notebook:** `20251008_IPR_Calculation.ipynb` (steady-state / CW, frequency-domain).
**Data:** exit-face frequency-domain `FieldMonitor`, absorbers run
`data/L_1_12x/…_n_2p90_absorbers.h5` (766×766 grid, ±33.51 a aperture, 1700 bins
ν = a/λ ∈ [0.32, 1.25], dν = 5.47×10⁻⁴). I = |Ex|²+|Ey|²+|Ez|².
**System:** LSU (local self-uniform) correlated-disordered dielectric — **not** hyperuniform —
n = 2.90, ff = 0.2237, a = 2.5626 µm,
L = 14.3 µm = 5.58 a. MPB pseudogap ν = [0.4321, 0.4601].
**Estimator:** PR diameter d = 2√(A_eff/π), A_eff = (∫I dA)²/∫I² dA, product-Simpson
on the true (non-uniform) monitor coordinates. Scale-invariant per bin.

All numbers below were recomputed by streaming the full 24 GB file
(`Claude/audit_workspace/steady/`, scripts + npz + figures). The streaming estimator
reproduces the notebook `diameter()` **exactly**: d(0.900) = 11.87962, d(gap 0.457) =
4.02942. Every load-bearing claim carries an adversarial-verification verdict (§8).

---

## 0. Executive summary — what changed this pass

Four quantitative results in the carried-over context were **wrong or over-stated** and
are corrected here:

1. **The single-shot PR diameter is biased LOW by ≈ +15 %** (not the "~5 %" the notebook
   assumed), and the frequency-boxcar `d_avg` (w = 11) **does not converge** — it is an
   inadequate partial correction. The right report is *raw d(ν) ± speckle error, plus an
   explicit ×1.15 (+15 %) ensemble correction*. (§2)
2. **The "6 % agreement with the diffusive baseline" is not defensible.** With the honest
   baseline (~11 a) the data sits *above* it, and the z₀ extrapolation-length systematic is
   **large and one-sided-up (×1.7–2.3)** because ℓ*/L ≈ 0.34 (the slab is only ~3 transport
   mfp thick). The comparison is systematics-dominated; only "consistent with diffusion,
   agreement not quantifiable to better than ~1.5×" is supportable. (§3)
3. **The gap "dip factor ≈ 3 at ν ≈ 0.457" conflated two different features.** The
   transmitted-power minimum is at **ν ≈ 0.444 (gap centre, −40 dB)**; the PR-**width**
   minimum is at **ν ≈ 0.459 (upper gap edge)**, offset by **27 bins (Δν = 0.015)**. At the
   gap centre the width is only ×1.8 below the diffusive level; the ×3.3 narrowing is a
   band-edge feature at the upper edge. (§4)
4. **The 2nd-moment ⟨ρ²⟩ does not track the confinement** (background/aperture-dominated) —
   the PR estimator is the correct one, and it is ~11–17× less background-sensitive. (§5)

The **robust** physical content that survives: a real, deterministic transverse **narrowing
in and around the gap** (PR width falls from ~12 a to ~3.4 a), located at the **upper band
edge**, converting to an **attenuation length L_att ≈ 0.1–0.5 a** — a deep-gap
Bragg/evanescent decay length, **not** a proven localization length ξ. Fixed single L,
single realization, 12×-tiled transverse geometry ⇒ no ξ and no localization claim from the
steady state alone (unchanged conclusion).

---

## 1. Method & anchors

One streaming pass over the (766, 766, 1, 1700) file accumulates, per bin, the
product-Simpson integrals S1 = ∫I dA, S2 = ∫I² dA, M2 = ∫ρ²I dA, giving d(ν) =
2√((S1²/S2)/π)/a, the 2nd-moment σ²(ν) = M2/S1/a², and the transmitted-power spectrum
S(ν) = S1. The same pass computes the speckle-mitigated d_avg(ν) for boxcar widths
w = 1…160 (exact: each row-block holds the full frequency axis), stores full 2D maps at 44
frequencies (aperture/background/azimuthal experiments), and 3000 complex per-pixel spectra
(spectral correlation). Fields rescaled ×10¹⁶ at read (I² would else underflow float32);
any global scale cancels in every estimator.

**Anchor (validates the pipeline against the notebook):** d(0.900) = 11.87962 (streaming) =
11.87962 (notebook `diameter()`) = 11.888 (last session's FFT single-bin) — identical.

---

## 2. The estimator and its uncertainty  (Goal 1)  — **decision made**

### 2.1 Single-shot speckle bias (systematic, LOW)
The exit field is one fully-developed speckle realization. Measured per-pixel intensity
**contrast** (core, freq-detrended, band 0.80–1.05) = **0.557**, matching ideal
3-component speckle 1/√3 = 0.577 (effective number of independent intensity contributions
M ≈ 3.2 — consistent with the near-equipartitioned Ex:Ey:Ez ≈ 0.34:0.34:0.31 depolarized
field). For such speckle the PR effective area is biased by
⟨A_eff,single⟩ = A_eff,ens/(1+1/M), i.e. the **diameter is biased LOW by √(1+1/M)**.

- **Analytic:** M = 3 → √(4/3) = **1.155** (+15.5 %); M = 3.2 → +14.9 %.
- **Monte-Carlo** (smooth Gaussian mean × resolved 3-component speckle, 60 trials):
  d_ens/d_single = **1.149 ± 0.015**, contrast 0.571 — reproduces the analytic factor.
- **Frequency-averaging plateau** (independent, on the real data): d_avg(w) climbs to
  ≈ 13.8 a by w ≈ 45–64 (where speckle is fully averaged, N_eff ≳ 9) and then creeps to 14.24
  by w = 160 — but that late creep **exceeds the speckle ceiling √(1+C²) ≤ 1.16**, so the extra
  ~3 points are **transport/profile blending** across the wide Δν ≈ 0.088 window, *not* speckle
  (refuter #2, PARTIALLY REFUTED my earlier "+18 %"). Speckle-only ⇒ **+15 %**.

⇒ **Ensemble-mean diffusive-band d ≈ raw × 1.15 (+15 % ± 2 %) ≈ 13.8 a.** The
diffusive baseline d_dif is built from a *smooth* theory profile (an ensemble quantity), so
the correct comparison is against this ensemble-corrected d, not raw d (§3).

### 2.2 Why `d_avg` (frequency boxcar) is NOT the estimator
- **It does not converge.** Under w-doubling the diffusive-band median rises 12.77 (w=11) →
  13.29 (w=22) → 13.80 (w=45) — ~4–5 % per doubling, never plateauing until w ≈ 128. The
  notebook's promised "check d_avg stable under w-doubling" **fails**: it is not stable.
  N_eff ≈ w/τ_d ≈ 11/7 ≈ **1.5 independent speckles at w = 11** — far too few (spectral
  intensity-correlation HWHM = 3.15 bins; d(ν)-residual integral correlation length
  τ_d = 7.2 bins).
- **It is corrupted where S(ν) is steep.** In the gap the power-weighted average pulls in
  out-of-gap band-edge bins carrying 30 dB more power: gap-centre d_avg is flat (~6.9) only
  to w ≤ 22, then blows up (11.8 at w = 160). So a wide window is *invalid* inside the gap.
- **Azimuthal averaging is also biased** (it over-broadens the genuinely peaked confined
  core — reports "contrast" > 1, i.e. profile peakedness, not speckle; gives 16.5 a in the
  diffusive band vs the 14.2 a plateau). Not used.

### 2.3 Final estimator (justified)
> **Primary:** report **raw per-bin d(ν)** with a **per-bin speckle 1σ error** = the local
> scatter (**±0.27 a** in the diffusive band, ~2.3 %; **±0.73 a** in the gap region).
> **Plus a stated one-sided systematic:** the ensemble-mean diameter is **raw × 1.15
> (+15 % ± 2 %)** — capped by the speckle ceiling √(1+C²) ≤ 1.16. `d_avg` (w = 11) is kept only as a *visual smoother*, flagged as
> under-corrected. A binned median over a flat-transport band is an equally valid central
> value (same 12.0 a) and inherits the same +15 % correction.

The precise ensemble d **cannot** be pinned from one realization (all ν-averaging surrogates
either don't converge or blend transport). Getting it to <5 % needs independent realizations
(shifted-source re-runs — R5/R7). This is the honest limitation.

---

## 3. Diffusive-baseline comparison — error budget  (Goal 2)

Honest Cherroret 2010 baseline (diameter of the T(q) = sinh(qℓ*)/sinh(qL) profile through
the *same* estimator, z₀ = 0), diffusive band 0.80–1.05:

| ℓ* input | d_dif(z₀=0) | ℓ*<L mask reaches |
|---|---|---|
| ℓ*/2 | 11.50 a | ν ≥ 0.32 |
| ℓ* (nominal) | 10.73 a | ν ≥ 0.32 |
| 2 ℓ* | 7.20 a | ν ≥ 0.336 |

so **d_dif ≈ 11 a**, and the ℓ*-factor-2 (Born/RDG unreliability, §9) moves it only 7.2–11.5 a
because σ²_dif is set by L, not ℓ* — the *level* is robust; only the **validity mask**
(where ℓ* < L) moves the low-ν cutoff. ℓ*/L ≈ 0.34 (range 0.25–0.44) in the band.

**The z₀ (extrapolation-length) systematic is large and one-sided up.** Effective medium
index n_eff ≈ 1.25 (Maxwell-Garnett) – 1.63 (volume-ε). The Egan-Hilgeman angle-averaged
internal reflectance gives R = 0.39 (n_eff = 1.25) to 0.67 (n_eff = 1.63), i.e.
z₀/ℓ* = (2/3)(1+R)/(1−R) ≈ **1.5–3.3**, so z₀ ≈ **0.5–1.1 L** — **not** ≪ L, so the paper's
"virtually independent of z₀" (needs z₀ ≪ L) **fails badly here** (independently confirmed by refuter #2). Using
σ²(z₀) = (2/3)[(L+2z₀)² − (ℓ*+z₀)² − 3z₀²] (ℓ*/L = 0.34):

| z₀ | σ²(z₀)/σ²(0) | baseline diameter × |
|---|---|---|
| 0 | 1.00 | ×1.00 (≈ 11 a) |
| 0.67 ℓ* (n≈1) | 1.86 | ×1.37 |
| 1.0 ℓ* (n≈1.25) | 2.29 | ×1.51 |
| 1.5 ℓ* (n_eff≈1.25) | 2.7 | ×1.70 |
| 3.3 ℓ* (n_eff≈1.63) | 5.2 | ×2.28 (≈ 25 a) |

**Budget (diffusive band):** measured raw d = 12.0 a → ensemble d ≈ **13.8 a (+15 %)**;
baseline = 11 a (z₀=0) → **19–25 a** with realistic z₀. The systematic band (11 → 25 a)
**fully brackets** the data (14 a). ⇒ **No precise agreement or disagreement is
supportable.** The prior "6 % above" was a coincidental cancellation of the (then hidden)
speckle low-bias against a z₀=0 baseline, on top of the now-removed baseline fudge.

**Defensible statement:** the data is *consistent with diffusion* in ν ≳ 0.7; it does **not**
fall below the diffusive baseline there (the confinement signature is confined to the gap);
the agreement cannot be quantified to better than ~1.5–2× given the z₀ + speckle +
Born-ℓ* systematics. Band-edge region ν ≈ 0.55–0.70 runs **above** baseline (raw d ≈ 14–15,
ensemble ≈ 16–17) — band-edge transport enhancement, not confinement.

---

## 4. Gap dip — rigorous quantification  (Goal 3)

**Two distinct features, offset by 27 bins:**

| feature | location | value |
|---|---|---|
| transmitted-power minimum S(ν) | **ν = 0.4443 (gap centre)** | −40.6 dB vs diffusive median (−28.3 dB vs local smooth) |
| PR-width minimum d_avg(w=22) | **ν = 0.4590 (upper gap edge)** | d = 3.40 a (raw 3.66 a) |
| offset | **Δν = +0.0148 = 27 bins** | |

**The width minimum is not a low-SNR null artifact:** at ν = 0.459 the transmitted power is
−27 dB, i.e. **13 dB above** the gap-centre null — there is real signal there. **It is a
genuine core feature, not a halo artifact:** shrinking the aperture ±33.5 a → ±16 a changes
the upper-edge d by only −3.5 % (3.39 → 3.27) and the gap-centre d by −4.6 % (7.01 → 6.69)
(§5). The maps confirm a bright tight central core that *tightens toward the upper edge*
(d: gap-centre 6.9 → 0.457 4.0 → 0.463 3.3 a) while the diffuse halo dims.

**Dip factor** (consistent raw-to-raw, diffusive raw ref 12.0 a): **×3.3 at the upper edge**
(raw d 3.66), but only **×1.8 at the gap centre** (raw d 6.77). The carried-over "factor ≈ 3
at ν ≈ 0.457" was the upper-edge width min mislabelled as the gap centre.

**Attenuation length** (Cherroret σ² ≈ 2 L L_att; σ² from the robust PR via σ² = d²/8):

| point | d | σ² = d²/8 | L_att = σ²/(2L) |
|---|---|---|---|
| upper-edge min (raw) | 3.66 a | 1.68 a² | **0.15 a = 0.39 µm** |
| upper-edge min (d_avg22) | 3.40 a | 1.44 a² | 0.13 a = 0.33 µm |
| gap centre (raw) | 6.77 a | 5.74 a² | 0.51 a = 1.32 µm |

**L_att ≈ 0.1–0.5 a (≲0.7 a speckle-corrected) is sub-lattice** — the signature of **Bragg/evanescent decay** inside a
photonic gap, not a localization length (ξ ≳ mfp ≈ ℓ* ≈ 1.5–2 a would be an order of
magnitude larger). This makes the CW caveat concrete: **the gap dip yields a Bragg
attenuation length, and the CW measurement alone cannot promote it to ξ** (Cherroret pp. 7–8;
absorption/attenuation give the identical σ² ≈ 2LL_a form). PR-width, 2nd-moment, and
transmission dips do **not** locate the same feature — reported explicitly.

---

## 5. Aperture & background robustness  (Goal 4) — **both claims confirmed on real maps**

**Aperture truncation** (d_PR at aperture half-size R, relative to full ±33.5 a):

| ν | R=33.5 | R=20 | R=16 | Δ(→20a) |
|---|---|---|---|---|
| 0.40 (ballistic) | 13.55 | 13.34 | 13.14 | −1.6 % |
| 0.444 (gap centre) | 7.01 | 6.82 | 6.69 | −2.8 % |
| 0.463 (width min) | 3.39 | 3.38 | 3.27 | −0.3 % |
| 0.90 (diffusive) | 11.91 | 11.88 | 11.77 | −0.3 % |

⇒ **d_PR is stable to <3 % halving the aperture to ±20 a** (<5 % to ±16 a) everywhere except
where the beam genuinely fills it. The ±33.5 a aperture is more than adequate. By contrast
the 2nd-moment ⟨ρ²⟩ changes **3.7×** across the same aperture range (gap centre 16 → 60 a²;
the *outer annulus* dominates — it is aperture-defined, not beam-defined).

**Background robustness** (add a flat background ε·⟨I⟩):

| ε | d_PR (gap) | ⟨ρ²⟩ (gap) | d_PR (diff) | ⟨ρ²⟩ (diff) |
|---|---|---|---|---|
| 0.01 | +1.0 % | +11 % | +1.0 % | +16 % |
| 0.10 | +9.9 % | +102 % | +9.7 % | +150 % |

⇒ **PR is 11–17× less sensitive to a flat background than ⟨ρ²⟩.** The 2nd moment is the
wrong estimator on single-shot data; PR is correct and must be applied to the non-negative
intensity.

---

## 6. Cross-consistency with the FFT notebook  (Goal 5) — read-only

The FFT method's single-bin (delta) window reconstructs I_W(x,y,t) = |Ê(f_k)|², constant in
t = the per-bin steady-state intensity — so d from a single-bin FFT window **equals** the
steady-state per-bin d(ν) by identity. Verified numerically:

| ν | d_perbin (this notebook = FFT single-bin) |
|---|---|
| 0.457 | 4.029 |
| 0.650 | 14.087 |
| 0.800 | 10.877 |
| 0.900 | 11.880 (last session: 11.888 "both ways") |
| 1.000 | 12.081 |

⇒ the two notebooks provably share one steady state.

---

## 7. What is faithful vs. what needs changing in the simulation

**Faithful / usable:** the PR estimator (background/aperture-robust, exact-Simpson);
d(ν) as a *relative* spectrum (scale-invariant, so the source envelope and absolute
cross-run mismatch cancel per bin); the deterministic gap narrowing and its L_att; the
transmission spectrum shape S(ν).

**Needs changing to make d(ν) publication-grade beyond a relative spectrum:**
1. **Multiple realizations** (shifted-source or re-seeded runs) — the *only* way to pin the
   ensemble-mean d and kill the ±15 % speckle bias/scatter. Highest value.
2. **A non-tiled, wider transverse domain** — the base cell is tiled 12× (period = L), so
   transverse structure beyond ~1 tile is Bloch-periodic; d(ν) beyond one tile is not
   transport. (Not limiting for the steady-state d, which is core-dominated, but forbids any
   localization-length claim.)
3. **A GaussianBeam / interior source** — the ±2.5 µm hard PlaneWave patch is
   sub-λ at low ν (0.31–1.22 λ across the band), so part of the low-ν broadening (d → 20 a at
   ν → 0.32) is *source* diffraction, not transport.
4. **Regenerate ℓ*(ν)** with a multiple-scattering method (the current g is Born/RDG,
   |m−1| = 1.9 ≫ 0.1 — factor-~2 unreliable), or extract ℓ* from an internal I(z) monitor.
5. **Interior detection plane** — an exit-face monitor maximizes coherent interfacial-wave
   contamination (Goïcoechea 2026).

---

## 8. Adversarial verification verdicts

_(Claims stated in `Claude/audit_workspace/steady/CLAIMS_FOR_VERIFICATION.md`; refuters
worked from the saved npz.)_

| Claim | Verdict | Note |
|---|---|---|
| **A** raw d biased low; d_avg(w=11) under-corrects | **PARTIALLY REFUTED → +15 %** | Mechanism + magnitude confirmed 5 ways (my MC 1.149, analytic √(1+1/M)=1.155, contrast 0.557, refuter #1's frequency-ensemble map 14.18, refuter #2's MC 1.151). But refuter #2 correctly showed my "+18 %" (w=160 plateau) **exceeds the speckle ceiling √(1+C²) ≤ 1.16** → ~3 pts are transport blending. **Honest value: +15 % ± 2 %** (ensemble d ≈ 13.8 a). d_avg-non-convergence claim stands. |
| **B** width-min ≠ transmission-min, offset ~27–37 bins, genuine core | **SURVIVES** | Refuter: S-min ν=0.44316, raw-d-min ν=0.46341 (offset 37 bins); monotonic 7.1→3.3 a ramp, residual scatter 0.039 a → ~99σ (not speckle); d-min sits at S1 = 2624 = **200× above** the null → not SNR; core contrast survives to R<8 a (gap-centre 6.05 vs edge 3.17). |
| **C** ⟨ρ²⟩ background/aperture-dominated; PR 11–17× more robust | **SURVIVES** | Refuter: ⟨ρ²⟩ 3.69× with aperture; flat 1 % bg shifts ⟨ρ²⟩ 12–13× more than d_PR (31× at the confined edge); ⟨ρ²⟩ even *computation-fragile* at the gap (square-vs-circular aperture disagree 35 %, d_PR agrees 2.5 %). |
| **D** baseline comparison systematics-dominated; no precise agreement | **SURVIVES** | Refuter #2 independently reproduced the σ²(z₀) multipliers and computed z₀/ℓ* = 1.5 (n_eff 1.25) to 3.3 (n_eff 1.63) via Egan-Hilgeman → baseline diameter **×1.7–2.3** (my ×1.5–1.9 was *conservative*). z₀ ≈ 0.5–1.1 L, so z₀ ≪ L fails badly. Prior "6 %" not defensible; honest bound ~1.5–2×. |
| **E** azimuthal averaging over-broadens (biased ensemble surrogate) | **SURVIVES** | Refuter #1 (decisive partial): azimuthally averaging an *already speckle-free* frequency-ensemble map (d = 14.18) inflates it to **16.46 (+16 %)** with no speckle to remove — proving azimuthal averaging adds a pure broadening artifact; not the ensemble estimator. |

_Note on the width-min location:_ the smoothed d_avg(w=22) minimum is at ν = 0.459; the raw
per-bin minimum is at ν = 0.4634 (just above the MPB upper edge 0.4601). Both are "upper gap
edge," offset 27–37 bins (Δν ≈ 0.015–0.020) from the transmission null at gap centre 0.444.

---

## 9. Corrections to carried-over context (memory / prior audit)

- **"Honest baseline ≈ 11.3 a, data median 1.06 above (~6 %)"** → the *level* (≈ 11 a,
  z₀=0) is right, but the "6 % agreement" is **not defensible** once the +15 % speckle
  low-bias (on data) and the ×1.7–2.3 z₀ systematic (on baseline) are included — both
  one-sided up, comparable to the effect (§3).
- **"Gap: min d = 4.03 at ν = 0.457, factor ≈ 3"** → this is the **upper-edge** width min
  (true davg min 3.40 at ν = 0.459), **offset 27 bins from the transmission null** at the
  gap centre 0.444; the gap-*centre* dip is only ×1.8. The ×3.3 is a band-edge feature (§4).
- **"d_avg (w = 11) raises high-ν d ~5 %, speckle-mitigated estimate"** → it raises it
  +6.4 % but is **not converged** (N_eff ≈ 1.5) and **not stable under w-doubling**; the true
  ensemble correction is +15 % (§2). Kept as a visual smoother only.
- **Unchanged & re-confirmed:** g stored ν-descending (reverse to align); ℓ* Born-unreliable
  (factor ~2, sets only mask/axis); PR estimator sound; CW-dip ≠ localization caveat correct;
  12×-tiling forbids ξ; d(0.9) = 11.88 cross-notebook.

---

## 10. Bottom line

The steady-state d(ν) is a sound **relative** transverse-width spectrum. Its single headline
physical result is a **deterministic transverse narrowing at the upper edge of the photonic
gap** (PR width 12 a → 3.4 a, L_att ≈ 0.1–0.5 a (≲0.7 a speckle-corrected)), distinct in location from the
transmission null at gap centre — a **Bragg/evanescent attenuation length, not a proven
localization length**. Quantitatively, the measurement is limited by (i) a +15 % single-shot
**speckle low-bias** that only multiple realizations can remove, and (ii) a diffusive-baseline
comparison that is **systematics-dominated** (z₀ ×1.7–2.3, Born-ℓ* ×2) and cannot claim
precise agreement. The 2nd moment is background-dominated; the PR estimator is the correct,
robust choice. No localization length is claimable from the CW data at fixed single L in a
12×-tiled geometry — consistent with the companion time-domain analysis.
