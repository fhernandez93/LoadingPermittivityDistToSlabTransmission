# TE plane-wave transmission through 2D dielectric networks (100 x 20 slabs)

Frequency-domain electric fields from 2D FDTD simulations (Tidy3D) of a TE plane wave
incident on a slab of a trivalent dielectric network. Two network geometries
(a gyromorph and a stealthy hyperuniform pattern) were simulated at two refractive
indices, plus one empty-box reference run.

## Folder contents

```
networks_100x20/
├── README.md
├── n_1p53/                                   # network index n = 1.53 (eps = 2.3409)
│   ├── gyromorph24_2d_K120_TE_Network.h5
│   ├── SHU2D_N15700_K100.0_TE_Network.h5
│   └── Reference.h5
└── n_3p4/                                    # network index n = 3.4  (eps = 11.56)
    ├── gyromorph24_2d_K120_TE_Network.h5
    ├── SHU2D_N15700_K100.0_TE_Network.h5
    └── Reference.h5
```

Each file is about 2.6 GB (uncompressed). The two `Reference.h5` files are the same
simulation (empty box, no structure) and are provided in both folders for convenience.

| File | Structure | Points in source pattern | Rods / nodes in slab |
|---|---|---|---|
| `gyromorph24_2d_K120_TE_Network.h5` | gyromorph pattern `gyromorph24_2d_K120` | 14 403 | 6 084 / 4 238 |
| `SHU2D_N15700_K100.0_TE_Network.h5` | stealthy hyperuniform pattern `SHU2D_N15700_K100.0` | 15 700 | 6 418 / 4 478 |
| `Reference.h5` | none (vacuum) | – | – |

## Units

All lengths are in units of `a`, the mean inter-point spacing of the underlying point
pattern (number density of points = 1 / a²). Frequencies are given as the reduced
frequency `nu = a / lambda` (vacuum wavelength). The dielectric is non-dispersive, so the
results are scale-invariant.

## Structure

1. The 2D point pattern is rescaled to unit number density and centred at the origin
   (full pattern roughly 120 x 120 a for the gyromorph, 125 x 126 a for the SHU pattern).
2. A trivalent network is built from the points: Delaunay triangulation, one node at the
   centroid of each triangle, and a dielectric rod joining the centroids of every pair of
   edge-sharing triangles (Florescu–Steinhardt–Torquato construction, PNAS 2009).
3. The network is clipped to the slab region `|x| <= 50 a`, `|y| <= 10 a`
   (slab size Lx x Ly = 100 a x 20 a). Rods crossing the slab boundary are cut at the boundary.
4. Rods are straight rectangles of half-width `r = 0.1 a` (full width 0.2 a); a disk of
   radius 0.1 a is placed on every node so the joints are filled.
5. The network is a lossless dielectric with refractive index `n = 1.53` (folder `n_1p53`)
   or `n = 3.4` (folder `n_3p4`), embedded in vacuum (n = 1). Rods are infinite along z (2D).

## Simulation setup (plane-wave transport experiment)

Geometry (x is the slab direction, y is the propagation direction, z is invariant):

```
 y = +30  ---------------------------------------  absorber (200 layers) on +y
 y = +20  ===== field monitor top edge ===========
 y = +10  ##### slab top ##########################
          #####  dielectric network, 100 x 20  ####
 y = -10  ##### slab bottom #######################
 y = -20  ----- plane-wave source (propagates +y), also monitor bottom edge
 y = -30  ---------------------------------------  absorber (200 layers) on -y
          x in [-50, +50], periodic in x
```

| Parameter | Value |
|---|---|
| Solver | Tidy3D FDTD, 2D (zero thickness in z) |
| Simulation domain | 100 a (x) x 60 a (y) |
| Boundaries | x: periodic; y: 200-layer absorbers (top and bottom) |
| Grid | uniform, dl = 1/(15 x 0.85) a ≈ 0.0784 a (15 cells per shortest vacuum wavelength) |
| Source | plane wave at y = -20 a, propagating in +y, normal incidence |
| Polarization | TE: electric field in the xy plane (E along x for the incident wave), H along z |
| Source spectrum | Gaussian pulse, centre nu0 = 0.475, width (fwidth) 0.2625 in nu, covering nu = 0.10 … 0.85 |
| Run time | 20 ps, with early shutoff when fields decay to 1e-20 of their peak |
| Field monitor | plane z = 0, x in [-50, 50] a, y in [-20, 20] a, 250 frequencies |
| Frequencies | nu = 0.100 … 0.850, 250 values, uniformly spaced (step ≈ 0.00301) |

Monitor fields are the steady-state (frequency-domain) complex amplitudes, normalised
by the source spectrum, so each frequency corresponds to a unit-amplitude continuous-wave
plane wave. Fields are recorded at every grid cell and colocated at cell centres.

The reference run uses exactly the same domain, grid, source and monitor with the
network removed. It gives the incident field for normalisation (e.g. transmitted
intensity = |E|² / |E_ref|², or use it to subtract the incident field in the region
below the slab).

Region guide for the monitor: `y < -10` contains incident + reflected field,
`-10 < y < 10` is inside the slab, `y > 10` contains transmitted field only.

## HDF5 layout

Identical for all six files:

| Key | Shape | dtype | Description |
|---|---|---|---|
| `coords/x` | (1276,) | float64 | x positions of the monitor grid, -50 … 50 a, uniform step ≈ 0.0784 a |
| `coords/y` | (512,) | float64 | y positions of the monitor grid, -20.04 … 20.04 a, same step |
| `nu` | (250,) | float64 | reduced frequency a/lambda, 0.10 … 0.85 |
| `fields/Ex` | (1276, 512, 250) | complex64 | x component of E, axes (x, y, nu) |
| `fields/Ey` | (1276, 512, 250) | complex64 | y component of E, axes (x, y, nu) |

Ez is identically zero for TE and is not stored. Datasets are contiguous (no chunking or
compression); reading a single frequency slice `[:, :, k]` reads a strided subset, so for
repeated access it is faster to load one component fully (about 1.3 GB) or re-chunk.

## Example (Python)

```python
import h5py, numpy as np

with h5py.File("n_3p4/gyromorph24_2d_K120_TE_Network.h5", "r") as f, \
     h5py.File("n_3p4/Reference.h5", "r") as ref:
    x, y, nu = f["coords/x"][:], f["coords/y"][:], f["nu"][:]
    k = np.argmin(abs(nu - 0.5))           # pick nu = 0.5
    Ex, Ey = f["fields/Ex"][:, :, k], f["fields/Ey"][:, :, k]
    Ex0, Ey0 = ref["fields/Ex"][:, :, k], ref["fields/Ey"][:, :, k]

I = abs(Ex)**2 + abs(Ey)**2
I0 = abs(Ex0)**2 + abs(Ey0)**2
# mean transmitted intensity in the free-space region above the slab
mask = y > 12
T = I[:, mask].mean() / I0[:, mask].mean()
```
