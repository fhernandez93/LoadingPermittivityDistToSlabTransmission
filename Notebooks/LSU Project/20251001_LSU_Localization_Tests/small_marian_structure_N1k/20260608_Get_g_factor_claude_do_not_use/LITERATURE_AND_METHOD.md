# Accurate calculation of the scattering anisotropy factor *g* for a correlated disordered slab

This folder reworks the `20260212_Get_g_factor` calculation. It does **not** modify the
original notebooks; it adds corrected tooling (`g_tools.py`) and two new notebooks that
mirror the project structure (`*_numerical_experiment.ipynb` for the simulations,
`*_get_g.ipynb` for the analysis).

---

## 1. What *g* is, precisely

The asymmetry (anisotropy) parameter is the **first moment of the single‑scattering phase
function** — the average cosine of the scattering angle of **one scattering event**:

```
g = <cos θ> = ( ∫ (dσ/dΩ) cos θ dΩ ) / ( ∫ (dσ/dΩ) dΩ )
```

where `θ` is the angle between the incident and scattered directions and `dσ/dΩ` is the
**differential scattering cross section** (the phase function, up to normalization).
`g = 0` is isotropic scattering, `g → +1` is strongly forward, `g → −1` strongly backward.

It connects the scattering and transport mean free paths:

```
ℓt = ℓs / (1 − g)        ⇔        g = 1 − ℓs/ℓt
```

* `ℓs` = **scattering** mean free path = mean distance between scattering events
  = decay length of the **coherent (ballistic)** intensity, `I_coh ∝ exp(−z/ℓs)`
  (what `20251002_Ls_test` measures).
* `ℓt = ℓ*` = **transport** mean free path = distance over which direction is randomised
  = the length entering the diffusion constant `D = v_E ℓt / 3` and Ohm's‑law transmission
  `T ∝ ℓt/L` (what `20251002_Lt_test` measures).

This relation is the *physical meaning* of `g`; here we compute `g` directly from the phase
function rather than from `ℓs`/`ℓt` (see Methods below).

### Correlated media: the structure factor

For a medium of scatterers with **positional correlations** (these **local self‑uniform (LSU)**
networks have a characteristic length `a`, i.e. a ring in `S(q)`), the
differential cross section is **not** that of an isolated particle. In the standard
"decoupling"/independent‑scattering picture,

```
dσ/dΩ (θ)  ∝  |f(θ)|²  ·  S(q),        q = |k_s − k_i| = 2k sin(θ/2),   k = 2π n_bg/λ
```

* `|f(θ)|²` — single‑particle **form factor** (shape of one scatterer),
* `S(q)`   — **static structure factor** (how the scatterers are arranged).

Correlations suppress `S(q→0)` (forward) and pile up scattering near `q ≈ 2π/a`. This is the
*dominant* physics for `g` in correlated disorder and **must** be captured. References:
Fraden & Maret 1990; Rojas‑Ochoa et al. 2004; Froufe‑Pérez et al. 2017 (§5).

### Coherent vs. incoherent — the crux of the error below

Write the field as a mean plus a fluctuation, `E = ⟨E⟩ + δE` (⟨·⟩ = ensemble average).

* `⟨E⟩` is the **coherent / ballistic** field. Its intensity `|⟨E⟩|²` lives almost entirely
  in the **forward (specular) direction** and decays as `exp(−z/ℓs)`.
* The **diffuse** intensity is `I_d = ⟨|E|²⟩ − |⟨E⟩|²`. The phase function and therefore
  `g` are defined from `I_d`, the **incoherent** differential cross section.

A single realisation's `|E_scat|²` is `|⟨E⟩ + δE|²`; near `θ = 0` it is dominated by the huge
coherent forward lobe. Using it as the phase function pushes `g → 1` artificially.
References: van Rossum & Nieuwenhuizen 1999; Akkermans & Montambaux 2007 (§2, §4).

---

## 2. What is wrong with `20260212_Get_g_factor`

The original `numerical_experiment` illuminates the **whole finite slab** with a TFSF source,
records the **scattered far field of one realisation** with a `FieldProjectionAngleMonitor`,
and the analysis computes `g = Σ I cosθ ΔΩ / Σ I ΔΩ`. Concretely:

1. **No ensemble averaging / no coherent subtraction (the main error).** It uses
   `intensity = |Eθ|² + |Eφ|²` of a single realisation. This is `|⟨E⟩+δE|²`, dominated near
   forward by the coherent lobe, so `g` is biased toward +1. The diffuse part
   `⟨|E|²⟩ − |⟨E⟩|²` is never formed. (See the "forward fraction" diagnostic in the original
   `get_g.ipynb` cell 5 — it sits near 1, the symptom.)

2. **It measures the whole slab, not the medium's single‑scattering phase function.** The
   far field of a finite slab includes coherent forward diffraction (the transmitted/refracted
   beam) and the finite‑aperture pattern of the rectangular block — neither belongs in the
   transport phase function.

3. **Angular grid bug.** `get_sphere(1000)` returns a **Fibonacci** set of *paired*
   `(θ_i, φ_i)` directions meant to be summed with **equal weights** (each ≈ 4π/N sr). But
   `FieldProjectionAngleMonitor` treats `theta` and `phi` as **independent axes** and projects
   onto the **tensor grid** `θ × φ` (output dims `(r, θ, φ, f)`). So passing 1000 θ's and
   1000 φ's silently builds a `1000×1000` grid, **destroying the equal‑area property**.
   The analysis then uses `dphi = np.diff(phi)[1]` and `np.gradient(theta)` on the *scrambled,
   non‑monotonic* Fibonacci `φ` axis → the solid‑angle weights `ΔΩ = sinθ Δθ Δφ` are wrong.
   *Fix:* use a **regular** `θ = linspace(0,π)`, `φ = linspace(0,2π)` grid and proper
   `sinθ dθ dφ` quadrature (or, if you keep Fibonacci, sum with equal weights — but the monitor
   cannot consume paired directions, so a regular grid is the correct choice here).

4. **TFSF box geometry.** The slab is `t_slab = 14.3 µm` wide, the TFSF box is `t_slab+0.5`,
   and the near‑field monitor is `t_slab+1`. The slab nearly fills the TFSF box, so the TFSF
   injection plane runs **through the structure** on the lateral faces (TFSF assumes its faces
   sit in homogeneous background) → corrupted injection, plus strong edge/aperture diffraction
   with sub‑µm margins. For a laterally‑large slab, **plane‑wave + empty reference subtraction**
   (as in `Lt_test`) is the correct way to isolate the scattered field; reserve TFSF for an
   *isolated, fully‑enclosed* scatterer.

5. **No single‑scattering (optically‑thin) check and no convergence checks.** `g = <cosθ>` of
   the single‑scattering phase function only equals the transport `1−ℓs/ℓt` when the sample is
   thin enough that **single scattering dominates** (`t ≪ ℓs`). The chosen `cut = 0.5/14.3`
   (≈ 0.5 µm) is not justified against `ℓs`, and there are no scans of thickness, grid
   resolution, run time, or angular sampling.

6. **`S(q)` is neither imposed nor validated.** For these correlated structures `S(q)` carries
   the dominant angular dependence of `g`; it must be measured from the structure and the
   result cross‑checked against it.

7. **`g` is used circularly.** `get_g.ipynb` cell 7 *uses* `lt = ls/(1−g)` to *derive* `ℓt`
   from the (biased) `g`, rather than computing `g` from first principles and checking it.
   (An independent `g = 1 − ℓs/ℓt` check exists via the `Ls`/`Lt` data but is intentionally
   out of scope here.)

---

## 3. The corrected method

`g` is computed two independent ways; agreement (or controlled disagreement) is the result.

### Method A — Born / power‑spectrum estimate from the structure (cheap, structure‑aware)
In the **first Born (Rayleigh–Gans) approximation** the single‑scattering differential cross
section of the *actual index pattern* is

```
dσ/dΩ (θ)  ∝  P(q),    P(q) = ⟨ |δε̃(q)|² ⟩   (radial average of the permittivity power spectrum),
q = 2k sin(θ/2),   k = 2π n_bg/λ
```

`P(q)` is obtained directly from the `.h5` permittivity by FFT of `δε = ε − ⟨ε⟩` (it is the
FFT pair of the autocorrelation already used by `AM.get_a_from_h5_eps`). Because it is built
from the real index distribution it contains **form factor and structure factor together**,
and it works for connected **networks** where an isolated "particle" is ill‑defined. Then

```
g_Born(λ) = ∫ P(q(θ)) cosθ sinθ dθ / ∫ P(q(θ)) sinθ dθ
```

**Low‑`q` handling — and an open caveat about the long‑wavelength limit.** These are **LSU
(local self‑uniform)** networks (Sellers, Man, Sahba & Florescu, *Nat. Commun.* **8**, 14439,
2017) — a *real‑space* design principle (each local region is statistically similar), which is
**distinct from hyperuniformity and does not imply `S(q→0)=0`**. Fitting `P(q) ∝ q^α` gives
`α ≈ 2`, but the fit window `[q_box, ~0.6 q_peak]` lies **above the box floor** `q_box = 2π/L`
and is really the **rising flank of the first correlation peak**, not the asymptotic `q→0`
scaling. The finite box simply **cannot reach `q→0`**, so this data **cannot decide** whether
- `S(0)=0` (hyperuniform‑like) → `g → g∞ = −α/(α+4)` (e.g. `−1/3`), or
- `S(0)` finite (generic disorder, plausible for LSU) → `g → 0` at long `λ`.

So the long‑wavelength limit of `g` is **genuinely undetermined** here; the negative values are
only established in the **resolved** band (roughly `ν ≳ 0.3`, where `2k` reaches the
mode‑rich part of `P(q)`).

Numerically we still (i) read the flank exponent with `power_spectrum_robust` (log bins +
minimum‑mode‑count cut + per‑bin SEM — windowing/apodization was tested and **rejected**, it
smears the peak into the low‑`q` region), and (ii) optionally produce a *smooth* `g(λ)` with
`power_spectrum_regularized`, which blends the measured spectrum into the fitted `C q^α` law
below `q_box`. **That `q < q_box` extension is a modelling choice (it assumes the `q^α`
suppression continues to `q→0`); it is NOT measured data and, for an LSU network, may be the
wrong limit.** `g_born_from_eps(..., spectrum=…)` exposes `"raw"` (data, NaN below the floor)
and `"regularized"` (smooth, but model‑dependent below the floor). Settling the limit requires
a **larger structure box** (smaller `q_box`), for either Method A or Method C.

*Limitation:* Born is a weak‑scattering expansion; with `n = 3.3` it is only a baseline. It
correctly captures the **angular position** of the correlation features (the `S(q)` ring) and
the small‑`q` exponent, but not strong‑scattering renormalisation — hence Method C.

> **Note.** The transport route `g = 1 − ℓs/ℓt` (from `20251002_Ls_test` /
> `20251002_Lt_test`) is a valid independent check, but it is **out of scope here** by
> request. The relation is kept above only as the physical interpretation of `g`.

### Method C — direct full‑wave diffuse far field (gold standard, non‑perturbative)
Full FDTD of a **thin slab** (`t ≪ ℓs`) with a plane wave; record the scattered far field with
a `FieldProjectionAngleMonitor` on a **regular** `(θ,φ)` grid, using a **closed box** around the
slab so the projection covers the full `4π`. (Note: `window_size` apodization is only valid for
single‑surface projection monitors, not a closed box, so it is not used; lateral truncation is
instead limited by the `Absorber` boundaries plus the lateral buffer. For strong apodization one
would use separate `+z`/`−z` surface monitors and combine them.) Then:

1. subtract the empty‑reference projection → scattered field `E_s(θ,φ)` per realisation;
2. form the **coherent** part `|⟨E_s⟩|²` and the **diffuse** part
   `I_d = ⟨|E_s|²⟩ − |⟨E_s⟩|²` by averaging over **independent realisations** (and/or, for a
   statistically isotropic slab, over azimuth `φ` and lateral sub‑apertures as a proxy when
   only one realisation is available, exactly as `Ls_test` uses lateral averaging for ⟨E⟩);
3. `g = <cosθ>` of `I_d` with correct `sinθ dθ dφ` quadrature.

Validate the FDTD→near‑to‑far pipeline against **analytic Mie** for a single dielectric sphere
(`g_tools.mie_*`) before trusting it on the disorder.

### Convergence / validation checklist
- [ ] NTFF pipeline reproduces Mie `g` for a test sphere (< few %).
- [ ] `g` independent of slab thickness `t` for `t ≪ ℓs` (single‑scattering plateau).
- [ ] `g` converged in grid resolution (`min_steps_per_lambda`), run time, and angular grid `(Nθ,Nφ)`.
- [ ] `g` converged in number of realisations (coherent lobe fully subtracted; forward fraction stabilises).
- [ ] Methods A and C agree within their stated regimes; the `S(q)`/`P(q)` ring lines up with the angular feature in `I_d(θ)`.

---

## 4. Literature

**Definitions, phase function, Mie (validation)**
- C. F. Bohren & D. R. Huffman, *Absorption and Scattering of Light by Small Particles*, Wiley (1983). — `g = <cosθ>`, phase function, Mie series (used to validate the NTFF pipeline).
- H. C. van de Hulst, *Light Scattering by Small Particles*, Dover (1981).

**Mesoscopic transport, ℓs vs ℓt, coherent vs incoherent**
- E. Akkermans & G. Montambaux, *Mesoscopic Physics of Electrons and Photons*, Cambridge (2007). — coherent/diffuse fields, `ℓt = ℓs/(1−g)`.
- M. C. W. van Rossum & T. M. Nieuwenhuizen, "Multiple scattering of classical waves: microscopy, mesoscopy, and diffusion," *Rev. Mod. Phys.* **71**, 313 (1999).
- P. Sheng, *Introduction to Wave Scattering, Localization, and Mesoscopic Phenomena*, 2nd ed., Springer (2006).
- A. Ishimaru, *Wave Propagation and Scattering in Random Media*, IEEE Press (1997). — radiative transfer, phase function, transport MFP.

**Structure factor & transport in correlated/dense media (why S(q) matters for g)**
- S. Fraden & G. Maret, "Multiple light scattering from concentrated, interacting suspensions," *Phys. Rev. Lett.* **65**, 512 (1990). — `dσ/dΩ ∝ F(q)S(q)`; `ℓ*` modified by `S(q)`.
- L. F. Rojas‑Ochoa, J. M. Mendez‑Alcaraz, J. J. Sáenz, P. Schurtenberger, F. Scheffold, "Photonic properties of strongly correlated colloidal liquids," *Phys. Rev. Lett.* **93**, 073903 (2004).
- P. D. García, R. Sapienza, C. López, "Photonic glasses: a step beyond white paint," *Adv. Mater.* **22**, 12 (2010).

**Local self‑uniformity (these structures) and correlated disordered photonics**
- S. R. Sellers, W. Man, S. Sahba, M. Florescu, "Local self‑uniformity in photonic networks," *Nat. Commun.* **8**, 14439 (2017). — **the design principle behind these LSU networks** (a real‑space measure of local structural uniformity; distinct from hyperuniformity, does not require `S(0)=0`).
- M. Florescu, S. Torquato, P. J. Steinhardt, "Designer disordered materials with large, complete photonic band gaps," *PNAS* **106**, 20658 (2009). — hyperuniform photonics (related but distinct class; for contrast).
- L. S. Froufe‑Pérez, M. Engel, J. J. Sáenz, F. Scheffold, "Band gap formation and Anderson localization in disordered photonic materials with structural correlations," *PNAS* **114**, 9570 (2017). — `ℓs`, `ℓt`, `S(q)`, transport in exactly this class of media.
- S. Torquato, "Hyperuniform states of matter," *Phys. Rep.* **745**, 1 (2018). — `S(q)`, hyperuniformity classes.
- R. Monsarrat, R. Pierrat, A. Tourin, A. Goetschy, "Pseudogap and Anderson localization of light in correlated disordered media," *Phys. Rev. Research* **4**, 033246 (2022).

**FDTD, TFSF, near‑to‑far transformation**
- A. Taflove & S. C. Hagness, *Computational Electrodynamics: The FDTD Method*, 3rd ed., Artech House (2005). — TFSF total/scattered‑field formulation, NTFF transform.
- Tidy3D docs: `FieldProjectionAngleMonitor`, `TFSF`, near‑to‑far‑field projection, apodization (`window_size`), `Absorber`/PML boundaries.

**Anderson localization of light (context for the broader project)**
- S. John, "Strong localization of photons in certain disordered dielectric superlattices," *Phys. Rev. Lett.* **58**, 2486 (1987).
- D. S. Wiersma, "Disordered photonics," *Nat. Photonics* **7**, 188 (2013).
- S. E. Skipetrov & J. H. Page, "Red light for Anderson localization," *New J. Phys.* **18**, 021001 (2016).

---

## 5. Files in this folder
- `g_tools.py` — corrected, dependency‑light helpers: regular angular grid, proper `sinθ dθ dφ`
  asymmetry integration, coherent/incoherent split, `P(q)`/`S(q)` from the `.h5`, `q(θ)` map,
  Born `g`, and a self‑contained Mie reference (validation).
- `20260608_numerical_experiment.ipynb` — corrected FDTD setups (Mie‑validation sphere +
  thin‑slab diffuse far field with regular grid, closed‑box 4π projection, and reference subtraction).
  Uploads are guarded by a `RUN` flag (default `False`) so nothing bills automatically.
- `20260608_get_g.ipynb` — computes `g` by Methods A and C and compares; saves to `./data/g_values`.
