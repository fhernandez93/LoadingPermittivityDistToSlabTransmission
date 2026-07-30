# Scattering anisotropy `g` of LSU structures — size series

Scattering anisotropy factor

```
g = <cos(theta)>
```

for a set of **statistically identical LSU (local self-uniform) structures that differ only in
size**. All samples are drawn from the same generation recipe, with the same rod geometry,
refractive indices and (to within 0.1%) the same filling fraction — the only thing that changes
from one curve to the next is how many rods / how large a box the sample contains.

`g` is computed per wavelength and averaged over the sample, so it is the usual transport
anisotropy factor: `g = 0` is isotropic scattering, `g > 0` forward-dominated,
`g < 0` backward-dominated.

## Files

| File | Contents |
|---|---|
| `n_rod_2.9275_ff_0.2174.txt` | `g(lambda)` for all five sizes, plain text, whitespace-separated |
| `n_rod_2.9275_ff_0.2174.png` | The same data plotted vs `lambda` (left) and vs reduced frequency `nu = a/lambda` (right) |

The file name encodes the rod index (`n_rod = 2.9275`) and the filling fraction of the reference
`N1000` sample (`ff = 0.2174`).

## Physical parameters (common to every structure)

```
a_scale      = 2.0501033142180396   # scale constant a [um], used for nu = a / lambda
aspect_ratio = 2.5                  # rod elongation along z
minor_radius = 0.2252               # transverse rod radius [um]
n_bg         = 1.0                  # background index (air)
n_rod        = 2.9275               # rod index
```

The scatterers are **rods elongated along z**: transverse radius `0.2252 um`, stretched by ans AR=
`2.5` along z, i.e. a half-extent of `2.5 * 0.2252 = 0.5630 um` in z. The structure is therefore
anisotropic by construction — `z` is not equivalent to `x`/`y`, and `g` here is the value for this
fixed rod orientation, not an orientation-average.

## The size series

| Column | N (points the rods are generated from) | Cubic slab side `L` [um] | Filling fraction |
|---|---|---|---|
| `N1000_L_11.4400`   | 1 000   | 11.4400 | 0.21744 |
| `N4000_L_18.1599`   | 4 000   | 18.1599 | 0.21786 |
| `N10000_L_24.6467`  | 10 000  | 24.6467 | 0.21899 |
| `N51296_L_42.5064`  | 51 296  | 42.5064 | 0.21889 |
| `N100000_L_53.0998` | 100 000 | 53.0998 | 0.21908 |

`N` is the **number of points of the underlying point pattern the rods are placed from** — not the
rod count of a mesh or a grid size. `L` is the **side of the cubic slab**, i.e. every structure is a
cube of side `L` (`L x L x L`), so the sample volume is `L^3`. `L` therefore grows roughly as
`N^(1/3)`, as expected for a fixed density in 3D.

**`N1000` is the original structure.** Everything else in the series is a larger sample of the same
statistical ensemble, generated to test how much of the `N1000` result is intrinsic and how much is
finite-size.

## Text file format

- 300 data rows, 7 whitespace-separated columns, one header line (no comment character).
- Columns:

  | # | Name | Meaning |
  |---|---|---|
  | 1 | `lambda_um` | vacuum wavelength [um] |
  | 2 | `nu` | reduced frequency `nu = a / lambda` |
  | 3–7 | `N…_L_…` | `g` for each structure, in the size order of the table above |

- The sweep is **uniform in `nu`**, not in `lambda`:
  `nu` runs from `0.08913493` to `2.05010331` in 300 equal steps (`d_nu = 0.00655842`).
  Consequently `lambda = a / nu` is **descending**: row 1 is `lambda = 23 um`, the last row is
  `lambda = 1 um`.

Read it with, e.g.:

```python
import numpy as np
d = np.loadtxt("n_rod_2.9275_ff_0.2174.txt", skiprows=1)
lam, nu, g = d[:, 0], d[:, 1], d[:, 2:]   # g[:, 0] = N1000 … g[:, 4] = N100000
```

## What the data shows

**Short wavelengths / high `nu`: fully converged.** For `nu >~ 0.65` (`lambda <~ 3.2 um`) all five
curves lie on top of each other to within `0.02` in `g`, and above `nu ~ 1.2` to within `0.01`. In
this regime `lambda` is much smaller than every box, the scattering is set by the local rod geometry
and correlations, and sample size is irrelevant. All sizes agree on the large-`nu` plateau
`g -> ~0.85` (strongly forward scattering) and on the position and depth of the backward-scattering
dip, `g ~ -0.42 … -0.46` at `nu ~ 0.40` (`lambda ~ 5.1 um`).

**Long wavelengths / low `nu`: size-dependent, and it is a finite-size effect.** Below `nu ~ 0.5` the
curves separate; the spread between largest and smallest sample grows from `~0.04` at `nu = 0.5` to
`0.1` at `nu = 0.3` and `~0.23` at `nu = 0.2`, peaking at `0.26` near `nu ~ 0.16` and staying of
order `0.2` down to the lowest `nu`. The trend with size is systematic: the larger the box, the earlier (in `lambda`) the
curve leaves the dip and the closer it comes to `g -> 0`, which is the expected effective-medium
limit — once `lambda` greatly exceeds all structural correlations, the sample should look homogeneous
and scattering should become isotropic. The small boxes instead stay stuck at a residual negative
`g` of order `-0.2` … `-0.3`, i.e. they cannot represent that limit. `N100000` is the only sample
that actually reaches `g ~ 0` (and goes slightly positive, `g <= +0.01`, over
`nu ~ 0.096–0.129`, i.e. `lambda ~ 16–21 um`).

**So: `g` converges better as the structure grows.** The convergence is with respect to the
long-wavelength end of the spectrum — increasing `N` pushes the wavelength down to which the result
is trustworthy, while leaving the already-converged short-wavelength part untouched. Note this is a
trend, not a smooth monotone sequence: the successive curves are not equally spaced (the mean
`|Δg|` for `nu < 0.5` is `0.031`, `0.037`, `0.015`, `0.076` going up the series), because with a
single realization per size there is also realization-to-realization speckle on top of the
systematic size effect.

**Rule of thumb for the usable range.** A curve should not be trusted where `lambda` approaches or
exceeds the cubic slab side, `nu <~ a / L`:

| Structure | Cubic slab side `L` [um] | `nu` at `lambda = L` |
|---|---|---|
| `N1000`   | 11.4400 | 0.179 |
| `N4000`   | 18.1599 | 0.113 |
| `N10000`  | 24.6467 | 0.083 |
| `N51296`  | 42.5064 | 0.048 |
| `N100000` | 53.0998 | 0.039 |

The tabulated range extends to `nu = 0.0891` (`lambda = 23 um`), which is *beyond* this limit for
every sample and roughly `2 L` for `N1000` — the extreme low-`nu` rows are reported for completeness
but are dominated by finite-size artefacts, most severely for the smaller boxes.

**Interpretation of the dip.** The backward-scattering minimum near `lambda ~ 5 um` sits where the
structure's dominant correlation length produces Bragg-like backscattering, and its position and
depth are essentially size-independent across the whole series — further evidence that the
short-wavelength physics is intrinsic to the ensemble and only the long-wavelength tail is a
sample-size question.
