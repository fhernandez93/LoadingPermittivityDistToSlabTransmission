# Stealthy hyperuniform 2D point patterns and photonic band structures

This folder contains ensembles of two-dimensional stealthy hyperuniform (SHU)
point patterns and photonic band-structure calculations for arrays of
dielectric rods placed at those points. Band structures were computed with
MPB (MIT Photonic Bands) using a periodic supercell that contains one whole
point pattern, plus one set of ensemble statistics stored in HDF5.

Everything needed to interpret the data is described here. No files outside
this folder are required.

## Physical system and conventions

- **Structure.** Infinitely long dielectric cylinders (rods) in air
  (epsilon = 1), with axes along z, centred at the points of a 2D SHU pattern.
  Rod radius r = 0.189 in every run. Rod permittivity is 11.56 (n = 3.4) in all
  runs except the one folder labelled `eps_47`, where it is 47.
- **Length unit.** The number density of points is 1: a pattern with N points
  sits in a square periodic box of side L = sqrt(N) (for example N = 198 gives
  L = 14.0712). Lengths are therefore in units of the mean inter-particle
  spacing. Coordinates are given relative to the box centre, in [-L/2, L/2).
- **Stealthiness parameter chi.** Each ensemble is labelled by chi, the fraction
  of constrained wavevectors used when the SHU pattern was generated. Values
  from 0.30 to 0.40 appear here.
- **Polarisation.** All MPB runs are TM (electric field parallel to the rods),
  unless the file name contains `TE`.
- **Frequency unit.** MPB reports frequencies as omega / (2 pi c) in units of
  1/(length unit), i.e. as (length unit)/lambda. The length unit is the mean
  spacing defined above, not the supercell side L.
- **Wavevectors.** In the CSV and log files, `k1, k2, k3` are the components of
  k in the basis of the supercell's reciprocal lattice vectors (fractions of
  2 pi / L). `kmag/2pi` is |k| / 2 pi in units of 1/(length unit). Named points
  of the supercell Brillouin zone: Gamma = (0, 0), X = (0.5, 0), M = (0.5, 0.5).
  Because the supercell is large, the bands are strongly folded and the k
  path is mostly a formality; the Gamma-point spectrum is the most used quantity.

## Folder map

```
.
├── SHU 2D point patterns (N=200)/     ensembles of point patterns, 11 values of chi
│   ├── chi_0.XX_posics.dat            500 samples per chi (text)
│   └── CTL_Files/chi_0.XX_posics/     MPB runs for the first 5 samples of each chi
├── bands_SHU_chi_0.37_eps_11p56/      one N=198 sample, eps 11.56, Gamma-X-M-Gamma path
├── bands_SHU_chi_0.37_eps_47/         same sample, eps 47, TM and TE
├── bands_SHU_chi_0.37_eps_11p56_N10000/  one N=10006 sample, eps 11.56, Gamma point only
└── band_structure_stats/              HDF5 ensemble band data and gap summary
```

## 1. `SHU 2D point patterns (N=200)/`

### Point-pattern ensembles: `chi_0.XX_posics.dat`

One file per chi (0.30, 0.31, ..., 0.40), each holding 500 independent
samples. Plain text, whitespace separated:

```
         500  #number of samples
         198   14.0712481       14.0712481       #num partics, box.x,box.y
  -3.90044332      -4.28830862
   6.06070995      -2.45731974
   ...                                   (N lines of x y)
         198   14.0712481       14.0712481       #num partics, box.x,box.y
   ...                                   (next sample)
```

Line 1 gives the number of samples. Each sample then consists of a header line
`N Lx Ly` followed by N lines of `x y`. N and L are the same for all samples of
a given chi but differ slightly between chi values:

| chi  | N   | L = sqrt(N) |
|------|-----|-------------|
| 0.30 | 200 | 14.1421347  |
| 0.31 | 194 | 13.9283876  |
| 0.32 | 197 | 14.0356684  |
| 0.33 | 197 | 14.0356693  |
| 0.34 | 203 | 14.2478075  |
| 0.35 | 198 | 14.0712471  |
| 0.36 | 203 | 14.2478065  |
| 0.37 | 198 | 14.0712481  |
| 0.38 | 193 | 13.8924437  |
| 0.39 | 195 | 13.9642401  |
| 0.40 | 203 | 14.2478075  |

Python reader:

```python
import numpy as np

def read_ensemble(path):
    with open(path) as f:
        n_samples = int(f.readline().split()[0])
        samples = []
        for _ in range(n_samples):
            n, lx, ly = f.readline().split()[:3]
            n = int(n)
            pts = np.array([[float(v) for v in f.readline().split()[:2]] for _ in range(n)])
            samples.append((float(lx), float(ly), pts))
    return samples          # list of (Lx, Ly, (N, 2) array)
```

### MPB runs: `CTL_Files/chi_0.XX_posics/`

For every chi, samples 1 to 5 of the ensemble above were converted into MPB
supercell calculations. File names encode the parameters, for example
`chi_0.38_posics_Lx_13.8924437_Ly_13.8924437_N_193.0_rods_sample_1`. Each sample
has four files:

| suffix          | content |
|-----------------|---------|
| `.ctl`          | MPB input (Scheme). Lists every rod centre and all solver settings. |
| `-epsilon.h5`   | Dielectric map as discretised by MPB (see HDF5 layout below). |
| `_out.out`      | Full MPB standard output: geometry echo, per-k `tmfreqs:` lines, `Gap from band ...` lines, timings. Wrapped by a shell script that adds `[MPB START]` / `[MPB END] exit code` lines. |
| `_out.csv`      | Band frequencies extracted from the log. |

Settings common to all 55 runs: eps = 11.56, r = 0.189, 500 TM bands,
k path Gamma -> X with 5 interpolated points (7 k-points in total),
resolution 32 grid points per length unit, eigensolver tolerance 1e-4,
run with `mpb-mpi` on 12 processes between 12 and 14 December 2025.
All runs finished with exit code 0; a typical run took about 45 minutes.

CSV layout (one row per k-point, comma separated, header row included):

```
kindex,k1,k2,k3,kmag/2pi,tmband1,tmband2,...,tmband500
```

Known issue: `chi_0.31_posics/..._sample_2_out.csv` contains only the Gamma
point (1 row), and its log shows a 2-minute single-k run. Treat that sample as
incomplete; the other 54 have all 7 k-points.

## 2. `bands_SHU_chi_0.37_eps_11p56/`

A single pattern: sample 1 of the chi = 0.37 ensemble (N = 198, L = 14.0712481),
rods with eps = 11.56, 600 TM bands along Gamma -> X -> M -> Gamma with 25
interpolated points per segment (79 k-points), resolution 32, tolerance 1e-4.
Run time about 3.5 hours.

| file | content |
|------|---------|
| `1_sample_chi_0.37_posics.dat` | the pattern, single-sample format: line 1 `N`, line 2 `Lx Ly`, then N lines `x y` |
| `1_sample_chi_0.37_bands.ctl`  | MPB input |
| `1_sample_chi_0.37_bands-epsilon.h5` | MPB dielectric map, 451 x 451 grid |
| `output.txt` | full MPB log |
| `output.csv` | extracted bands, 79 rows, columns `kindex,k1,k2,k3,kmag/2pi,tmband1..tmband600` |

## 3. `bands_SHU_chi_0.37_eps_47/`

Same pattern as folder 2 (the two `posics.dat` files are identical) but with
rod permittivity 47.

| file | content |
|------|---------|
| `settings_mpb_ctl_supercell.txt` | parameters fed to the Fortran script that wrote the `.ctl`: positions file, eps = 47, r = 0.189, 600 bands, 9 interpolation points, resolution 32, tolerance 1e-4 |
| `1_sample_chi_0.37_bands.ctl` | MPB input, TM, Gamma -> X -> M -> Gamma, 9 interpolated points per segment (31 k-points) |
| `1_sample_chi_0.37_bands.csv` | TM bands, 31 rows of 605 values (`k index,k1,k2,k3,kmag/2pi` then 600 bands). The header row is malformed: it stops at `tm band 320`, so read this file by column position (see the last section). |
| `1_sample_chi_0.37_bands_TE.csv` | TE bands from a separate run with 400 bands on the same k path. Rows for k indices 8 and 13 are missing (29 rows). The TE `.ctl` and log were not kept. |
| `1_sample_chi_0.37_posics.dat` | the pattern, single-sample format as in folder 2 |
| `README.txt` | the original short note that accompanied this folder. It mentions an `-epsilon.h5` file that is not present here. |
| `disk_usage.txt` | unrelated `df` listing from the machine that ran the job; safe to ignore |
| `README.csv`, `disk_usage.csv`, `settings_mpb_ctl_supercell.csv` | empty (0 bytes), accidental duplicates; safe to ignore |

## 4. `bands_SHU_chi_0.37_eps_11p56_N10000/`

A much larger single pattern at chi = 0.37: N = 10006 points, L = 100.029991,
eps = 11.56, r = 0.189. Only the Gamma point was computed, at resolution 10
(grid 1001 x 1001), tolerance 1e-4. Two runs with different band counts:

| files | bands | run time |
|-------|-------|----------|
| `1700_freqs.csv`, `1700_freqs.txt` | 1700 | about 30 hours |
| `2100_bands.csv`, `2100_bands.txt` | 2100 | about 121 hours |

Each `.csv` has one data row (Gamma) with columns
`kindex,k1,k2,k3,kmag/2pi,tmband1..tmbandN`; the `.txt` is the full MPB log.

Other files:

| file | content |
|------|---------|
| `chi_0.37_positions.txt` | the pattern: header line `N Lx Ly`, then 10006 lines `x y` |
| `create_ctl_from_positions.py` | script that wrote `mpb_all_rods.ctl` from the positions file |
| `mpb_all_rods.ctl` | MPB input as used for the 2100-band run (the 1700-band run used the same file with `num-bands` set to 1700) |
| `mpb_all_rods-epsilon.h5` | MPB dielectric map, 1001 x 1001 grid |
| `output.csv` | not from this system: a byte-identical copy of `output.csv` from folder 2 (N = 198, 600 bands). Ignore it here. |
| `chi_0.37_positions.csv` | empty (0 bytes); ignore |

**Known issue, affects both runs in this folder.** The generator script does
not skip the header line of the positions file, so it emitted one extra rod
with centre (10006, 100.03). After periodic wrapping this spurious rod sits at
approximately (3.00, 0.00) inside the cell, giving 10007 rods instead of 10006
(the MPB logs list 10007 cylinders). The effect of one extra rod among ten
thousand is small, but it is there.

## 5. `band_structure_stats/`

HDF5 files with ensemble band data for chi = 0.30 to 0.37, all at eps = 11.56,
r = 0.189, 1000 TM bands, 12 k-points. The solver settings stored in these
files (`NPx_x`, `NPx_y`, `N_reciprocal`, `subsampling`, `sub_subsampling`) are
those of the code that produced them; that code and its exact method are not
recorded in the file.

### `chi_0.XX_band_sampling.h5`

All datasets live at the root of the file.

| dataset | shape | meaning |
|---------|-------|---------|
| `all_bands` | (1000, Nk=12, Nsamples) | frequencies, indexed [band, k, sample], same units as the MPB CSVs |
| `all_posics` | (2, Npartic, Nsamples) | rod centres, indexed [xy, point, sample], same units and box as the `.dat` ensembles |
| `k_array` | (12, 2) | the 12 k-points in supercell reciprocal units: Gamma -> X -> M -> Gamma in steps of 0.125, with the closing Gamma omitted |
| `Gamma`, `X`, `M` | (2,) | the three named points |
| `G_index`, `X_index`, `M_index` | scalar | 1-based positions of Gamma, X, M in `k_array` (1, 5, 9) |
| `k_path_length` | (12,) | cumulative path coordinate; its values repeat and then decrease, so prefer recomputing it from `k_array` |
| `chi`, `eps`, `radius`, `Npartic`, `Nsamples`, `Nbands`, `Nk` | scalar | parameters |
| `NPx_x`, `NPx_y`, `N_reciprocal`, `subsampling`, `sub_subsampling` | scalar | solver settings (500, 500, 8000, 5, 5 in every file) |

Nsamples is 125 for chi 0.30 to 0.36 and 500 for chi 0.37. Sample i in
`all_posics` is sample i of the corresponding `chi_0.XX_posics.dat` ensemble
(checked for chi 0.30 and 0.37). Band 1 at Gamma is exactly 0.

```python
import h5py, numpy as np
with h5py.File("band_structure_stats/chi_0.37_band_sampling.h5") as f:
    bands = f["all_bands"][...]      # (1000, 12, 500)
    pos   = f["all_posics"][...]     # (2, 198, 500)
    k     = f["k_array"][...]        # (12, 2)
```

### `20251218_bandgaps_chi_0p30_0p40.h5`

Gap summary. One dataset per chi, named `"0.30"`, `"0.31"`, ..., `"0.37"`
(despite the file name, 0.38 to 0.40 are not present). Each is a pair of
frequencies `[lower, upper]` in the same units as above, intended as the edges
of the main TM gap for that ensemble. The precise rule used to define the
edges across samples is not stored in the file.
`20251218_bandgaps_chi_0p30_0p40_tolerance_0.02.h5` is byte-identical to the
file without the suffix.

## MPB `-epsilon.h5` layout

Standard MPB output, 2D grid of `resolution x L` points per axis:

| dataset | content |
|---------|---------|
| `data` | scalar epsilon on the grid (1 in air, up to 11.56 or 47 in the rods, intermediate values on smoothed boundaries) |
| `epsilon.xx`, `.xy`, `.xz`, `.yy`, `.yz`, `.zz` | components of the (smoothed, anisotropic) epsilon tensor |
| `epsilon_inverse.*` | same for the inverse tensor |
| `lattice vectors` | (3, 3) supercell vectors |
| `description` | the string `dielectric function, epsilon` |

## Reading the band CSV files

Header names differ slightly between folders (`tmband1` versus `tm band 1`,
`kindex` versus `k index`), and the header of
`bands_SHU_chi_0.37_eps_47/1_sample_chi_0.37_bands.csv` is truncated after
`tm band 320` and ends with a stray log fragment, although its data rows do
hold all 600 bands. Skip the header and rely on column positions instead:

```python
import numpy as np

def read_bands(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    kmag  = data[:, 4]          # |k| / 2 pi
    bands = data[:, 5:]         # (Nk, Nbands)
    return kmag, bands
```

The first five columns are always k index, k1, k2, k3, |k|/2 pi, in that
order; the bands follow in ascending order.
