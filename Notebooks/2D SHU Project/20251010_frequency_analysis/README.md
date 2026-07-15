# Synthetic Beam Focusing Through Disordered SHU Cavities

This folder analyzes how a **focused microwave beam** couples through a disordered
**stealthy hyperuniform (SHU)** slab sitting inside a thin 3D cavity, and how the
cavity **air gap** changes that coupling. The central object is the notebook
`20260714 Analysis Focusing Cavity.ipynb`, which synthesizes the focused beam *in
post-processing* from a set of single-emitter FDTD simulations.

This README explains three things people keep asking about:

1. **How the focusing actually happens** in this setup.
2. **Why the "beam at entry" has a size that varies with frequency.**
3. **The math** behind both.

All numbers quoted below were measured directly from
`data/cavity_data/0mm_Gap_Cavity_3D_field_intensities.h5` (the 0 mm-gap dataset)
and cross-checked with independent scripts.

---

## 1. The idea: superposition instead of a real beam

Maxwell's equations are **linear**, so we never simulate a Gaussian beam hitting the
slab. Instead we run **one FDTD simulation per point-dipole emitter**, placed at
transverse positions

$$x_j \in \{\pm0.5,\ \pm1.5,\ \dots,\ \pm5.5\}\ \text{mm}\qquad(12\ \text{emitters, 1 mm spacing, full aperture } D = 11\ \text{mm}),$$

and record the complex field $E_z^{(j)}(x,\nu)$ on a transverse cut-line. **Any**
input beam is then a **weighted coherent sum** of these single-emitter responses:

$$\boxed{\,E_z^{\text{beam}}(x,\nu)\;=\;\sum_{j} w_j(\nu)\,E_z^{(j)}(x,\nu)\,}$$

The complex weights $w_j(\nu)$ encode the amplitude **and phase** a converging beam
would carry at each emitter. Because the simulations are reusable, we can synthesize
*any* beam — focused at any distance, any aperture — without re-running FDTD.

Two field lines are recorded per emitter:

| dataset key | plane | meaning |
|---|---|---|
| `field_empty_cavity` | pre-slab, $y=-5.5555$ mm | **"beam at entry"** — empty cavity, no disorder |
| `field_slab_cavity`  | post-slab, $y=+10$ mm    | transmitted / exit field through the SHU slab |

---

## 2. Geometry — where the focus lands

From the source-simulation notebook (`20250708 Numerical Experiment Cavity 3D`), the
dipoles are placed at

$$y_{\text{emitter}} = -\tfrac{1}{2}(100\,a\,\text{slicing}) - 22.5 = -(\text{slab half-length}) - 22.5\ \text{mm}.$$

The slab front face is at $y_{\text{front}} = -(\text{slab half-length})$, so the
`slab half-length` terms **cancel**:

$$\boxed{\ \text{emitter} \to \text{slab entrance} = 22.5\ \text{mm, independent of the lattice constant } a.\ }$$

With the actual run ($a=1.5873$, $\text{slicing}=0.07$): slab spans $y\in[-5.56,\,+5.56]$
(thickness $\approx 11$ mm → the "L=11" label), the front face is at $-5.5555$, and the
pre-slab reference line coincides with it. **The propagation distance from the emitter
plane to the pre-slab plane is $z = 22.5$ mm**, so building the weights with focal
distance $z_{\text{focus}} = 22.5$ mm places the designed focus **exactly on the
pre-slab plane** — i.e. on the "beam at entry" line. The exit line ($y=+10$) is a
further 15.56 mm downstream.

```
   emitter plane            pre-slab / entrance         exit
   y = -28.06               y = -5.5555                 y = +10
      |------------ 22.5 mm ------------|--- SHU slab ---|--- 4.5 mm ---|
      * dipoles ±0.5..±5.5             (focus here)     (measured here)
```

---

## 3. The weights — math of the Gaussian design

`calculate_gaussian(emitter, lambda_0, w_0, w_ap, z_focus)` builds each $w_j$ as a
product of three factors:

$$w(x,\lambda)\;=\;\underbrace{\exp\!\Big[-\tfrac{x^{2}}{w(z)^{2}}\Big]}_{\text{waist envelope}}\;
\underbrace{\exp\!\Big[-\tfrac{x^{2}}{w_{\mathrm{ap}}^{2}}\Big]}_{\text{aperture taper}}\;
\underbrace{\exp\!\Big[-\,i\,k\,\tfrac{x^{2}}{2R(z)}\Big]}_{\text{converging phase}},\qquad k=\tfrac{2\pi}{\lambda}.$$

with the standard Gaussian-beam relations at $z = z_{\text{focus}}$:

$$z_R=\frac{\pi w_0^2}{\lambda},\qquad R(z)=z\Big[1+\big(\tfrac{z_R}{z}\big)^2\Big],\qquad w(z)=w_0\sqrt{1+\big(\tfrac{z}{z_R}\big)^2}.$$

- **Waist envelope** ($w_0 = 2$ mm at the focus). Sets the physical Gaussian envelope
  and, more importantly, the wavefront radius $R(z)$ used by the phase. Because
  $\lambda \gg$ aperture here, the emitters sit deep in the far field of a 2 mm waist
  ($z\gg z_R$), so this envelope is $\approx$ flat across the emitters — but it still
  fixes $R(z)$.
- **Aperture taper** ($w_{\mathrm{ap}} = 5$ mm at the emitter plane). This is what
  actually rolls the outer emitters off: $|w| \approx 0.99$ at $x=0.5$ → $0.30$ at
  $x=5.5$.
- **Converging phase.** Pre-compensates the extra path from an off-axis emitter to the
  on-axis focus so all emitters add in phase there. The **negative** sign is the
  converging one under Tidy3D's $e^{-i\omega t}$ convention — empirically confirmed
  (see §4). Flip only if your field convention is $e^{+i\omega t}$.

### Units gotcha (important)

The geometry is in **mm** and `freqs` are in **GHz**, so the wavelength must be built
with $c$ in mm·GHz:

```python
lambdas = (td.C_0 / 1e12) / freqs      # c = 299.79 mm*GHz
```

Using `td.C_0` directly (µm·Hz) gives $\lambda \sim 10^{13}$ and collapses **every**
weight to $1+0j$ — no shaping at all. This was a real bug in an earlier version and
is exactly what made the entry beam look "unfocused."

---

## 4. How the focusing actually happens

Here is the surprise, and the answer to "does it matter if it's one emitter or many?"
Measured transverse width $\sigma=\sqrt{\langle x^2\rangle-\langle x\rangle^2}$ of the
normalized intensity at 6 GHz, pre-slab plane:

| what we sum | $\sigma$ (mm) | note |
|---|---|---|
| single dipole (no sum) | **25.2** | a subwavelength source radiates **broadly** (true FWHM $\approx$ 78 mm, clipped by the ±50 mm window) |
| flat weights, all 12 ($w_j=1$) | **7.55** | concentration from **coherent summation** alone |
| + aperture taper (no phase) | **5.11** | apodization shrinks the effective source cluster |
| + converging phase (full) | **4.98** | the tuned wavefront adds only **2.5 %** |
| random phase $\exp(i\,2\pi j/12)$ | **28.6** | scrambling coherence **destroys** the focus |

Emitter-count sweep (flat weights): $\sigma = 25.2\,(1) \to 23.8\,(2) \to 16.9\,(4)
\to 12.2\,(8) \to 7.55\,(12)$ mm.

**Interpretation.**

- **Number of coherent emitters matters, decisively.** One dipole is broad; 12 coherent
  dipoles concentrate to ~7.5 mm. Adding emitters *widens* the physical source cluster
  yet *narrows* the beam — classic **antenna-array / aperture-synthesis** behavior
  (wider coherent aperture → narrower main lobe).
- **Coherence is essential.** Randomizing the relative phases blows the width up past a
  single emitter. So the concentration is genuine constructive interference, not just
  "sources sitting near $x=0$."
- **The tuned converging phase barely matters (2.5 %).** The aperture ($\pm 5.5$ mm) is
  tiny compared to $\lambda \approx 50$ mm, so the curvature it needs is a fraction of a
  radian — there is almost nothing for the phase to correct (see §6, Fresnel number).

So the beam **is** focused — but by **coherence + aperture (how many emitters and how
wide)**, not by the curvature phase you tune. The lever for *better* focusing is the
aperture, not the phase.

### Why the phase sign is what it is

Empirically, the negative (converging) sign gives the narrowest pre-slab spot
($\sigma_{6\,\text{GHz}} = 4.98$ mm) versus $5.25$ mm for the positive sign and $5.11$ mm
for aperture-only. This both confirms the sign and pins down the $e^{-i\omega t}$
convention on the data.

---

## 5. Why the beam at entry has a **variable size**

The synthesized entry-beam width is **not constant** — it scales with wavelength:

| $\nu$ (GHz) | $\lambda$ (mm) | $\sigma_{\text{entry}}$ (mm) |
|---|---|---|
| 3.40 | 88.1 | 10.37 |
| 4.81 | 62.3 | 6.45 |
| 6.23 | 48.1 | 4.82 |
| 7.65 | 39.2 | 4.72 |
| 9.07 | 33.0 | 3.88 |

$\text{corr}(\sigma,\lambda) = 0.978$, and a one-parameter fit gives

$$\boxed{\ \sigma_{\text{entry}}(\nu)\ \approx\ 0.107\,\lambda(\nu)\ =\ 0.107\,\frac{c}{\nu}\ }\qquad(\text{RMS residual } \approx 6\%).$$

The width grows from ~3.9 mm at 9 GHz to ~10.4 mm at 3.4 GHz. Because the empty cavity
has no disorder, this variation is **smooth** in frequency (no speckle) — a clean
$\propto\lambda$ trend.

### The math behind it

The transverse structure of the sum comes entirely from the **relative phases** between
emitter contributions. Two emitters separated by $\Delta x_j$, observed at distance $z$,
produce interference fringes in $x$ with period

$$\Lambda \;=\; \frac{\lambda z}{\Delta x_j}.$$

Every interference length in the pattern is $\propto \lambda$ (this is the van
Cittert–Zernike / two-slit scaling: the only length the wave equation adds beyond the
fixed geometry is $\lambda$, and features live where the phase difference
$k\,\Delta r = 2\pi\Delta r/\lambda$ is $O(1)$). The finest fringe comes from the widest
emitter pair, $\Delta x_j = D = 11$ mm:

$$\Lambda_{\min} = \frac{\lambda z}{D} = \frac{22.5\,\lambda}{11} \approx 2.05\,\lambda.$$

So the **whole pattern — including the central lobe — scales linearly with $\lambda$**.
As $\nu$ rises, $\lambda = c/\nu$ falls, and every transverse feature (hence
$\sigma_{\text{entry}}$) shrinks in proportion. That is the entire reason the "beam at
entry" has a variable size: it is a coherent interference pattern of fixed geometry, and
interference patterns scale with wavelength.

(The measured prefactor $0.107$ is smaller than $\Lambda_{\min}/\lambda$ because
$\sigma$ measures the RMS of the central peak, not the fringe period, and the aperture
taper + subwavelength-source details set the exact constant. The **proportionality** to
$\lambda$ is the robust, physical part; the prefactor is geometry-specific.)

A mild flattening appears at the short-$\lambda$ end ($\sigma$ barely moves between
$\lambda=48$ and $39$ mm), where the spot starts to be limited by the finite aperture
rather than by $\lambda$.

---

## 6. Regime and hard limits

Two dimensionless numbers frame what is and isn't achievable:

$$N_F = \frac{a^2}{\lambda z}\ \Big|_{a=5.5,\,z=22.5} = 0.015\ \text{–}\ 0.041,\qquad
\frac{z}{\lambda} = 0.26\ \text{–}\ 0.68.$$

- **$N_F \ll 1$.** The quadratic phase across the aperture spans only $\pi N_F \approx
  0.05$–$0.13$ rad, where genuine focusing needs $\sim\pi$. This is *why* the tuned
  converging phase contributes only 2.5 % (§4). To reach $N_F \approx 1$ at $z=22.5$ mm
  you would need emitters out to $a=\sqrt{\lambda z}\approx 27$–$34$ mm (vs $\pm5.5$
  now).
- **Subwavelength aperture.** The full aperture $D = 11$ mm is *smaller* than
  $\lambda = 33$–$88$ mm. A subwavelength emitter array radiates like a compact source
  (a single dipole's field is broad and nearly frequency-independent), and the
  "focus" is coherent aperture synthesis rather than a lens-like diffraction focus.
- **Diffraction floor.** Even with a perfect aperture, the spot cannot beat
  $\sim\lambda/2 \approx 15$–$44$ mm in this band — comparable to the whole $\pm 50$ mm
  measurement line. In the microwave regime a sub-cm focus is physically impossible.

### What to change to focus better

| lever | requirement | note |
|---|---|---|
| widen aperture $a$ | emitters out to $\pm27$–$34$ mm → $N_F\!\approx\!1$ | the real fix; needs new FDTD runs |
| higher frequency | $\lambda < D$ needs $\nu > 27$ GHz | new sims; SHU lattice moves to a new regime |
| closer focus $z$ | fixed at 22.5 mm by dipole placement | not available in post-processing |

Since none of these are free, the pragmatic move is to **change the observable**:
report the **focusing enhancement** — e.g. $|E_{\text{focused}}(0)|^2 /
|E_{\text{flat-phase}}(0)|^2$, or $\text{PR}_{\text{flat}}/\text{PR}_{\text{focused}}$ —
which isolates the air-gap / disorder effect and is meaningful even at $N_F \ll 1$,
instead of chasing an absolute spot size the geometry cannot deliver.

---

## 7. Focusing metrics

Per frequency, on the exit intensity $I(x) = |E_z^{\text{beam}}(x)|^2$:

- **Participation ratio (effective spot size):**
  $$\text{PR} = \frac{\big(\int I\,dx\big)^2}{\int I^2\,dx}.$$
  Small PR ⇒ concentrated; large PR ⇒ spread out.
- **Beam width:** $\displaystyle \sigma = \sqrt{\langle x^2\rangle - \langle x\rangle^2}$,
  with $\langle f\rangle = \int f\,I\,dx / \int I\,dx$.

Sweeping frequency turns each into a spectrum $\text{PR}(\nu)$ / $\sigma(\nu)$. Across
the 3D datasets only the **air gap** changes, so differences isolate the effect of the
gap on transmission through the SHU slab; the 2D experiment and the beam at entry serve
as baselines.

---

## 8. Caveats to carry forward

- The pre-slab reference cut is **hardcoded** at $y = -5.5555$ in the analysis. It
  coincides with the slab entrance only for this $a\cdot\text{slicing}=0.111$. If you
  rerun with a different lattice constant, replace it with a computed
  `-(slab_half_length)`.
- Absolute PR/$\sigma$ mostly report the array footprint, not the tunable focus. Prefer
  the **enhancement** metric (§6) for physics conclusions.
- The phase sign assumes $e^{-i\omega t}$; verify against your field convention if you
  reuse the code elsewhere.
