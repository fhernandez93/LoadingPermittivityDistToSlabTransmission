"""Convert Luis' MPB band CSVs into `*_band_sampling.h5` files.

Input  : a directory tree of MPB outputs, e.g.
             Luis Data/Cavity_Like_Structures/eps_43_bands/
                 002_chi_0.37_bands.ctl            <- MPB input (eps, radius, rods, k-path)
                 002_chi_0.37_bands_eps_43.csv     <- MPB output (header + one row per k-point)
                 ...
Output : one HDF5 per (chi, eps) group, with the same key/shape/dtype schema as the
         pre-existing `Luis Data/band_structure_stats/chi_0.XX_band_sampling.h5` files,
         so `Computing_nDOS.ipynb` / `PBG_boundaries.ipynb` (and
         `20260112_Get_DOS_From_Luis_Data.ipynb`) read them unchanged.

         all_bands is (Nbands, Nk, Nsamples) -- bands, k-points, samples.

Layouts:
  nested (default)  <out>/eps_43_bands/chi_0.37_band_sampling.h5
                    -> matches the notebooks' `'chi_0.'+chi+'_band_sampling.h5'` pattern;
                       point the notebook at one eps subfolder at a time.
  flat              <out>/chi_0.37_eps_43_band_sampling.h5
                    -> all groups side by side (os.walk-style loops, `eps` key distinguishes them).

Usage:
    python csv_to_band_sampling.py                       # defaults to Cavity_Like_Structures
    python csv_to_band_sampling.py --input <dir> --output <dir> --layout flat
    python csv_to_band_sampling.py --keep-closing-k      # keep the duplicated closing Gamma

Notes / conventions inherited from the existing files:
  * G_index / X_index / M_index are 1-BASED (they come from the Fortran ctl generator).
  * The closing Gamma of the Gamma-X-M-Gamma path duplicates the opening one, so by default
    it is dropped (the existing files do the same: interpolate 3 -> 13 k-points stored as 12).
    Keeping it would double-weight Gamma in DOS histograms.
  * `k_path_length` here is a *recomputed* cumulative arc length along the path in units of
    1/a (i.e. |k|/2pi, same units as the CSV's `kmag/2pi` column, also stored as
    `kmag_over_2pi`). The values in the old files are not reproducible/monotonic and are
    not used by any notebook.
  * Keys tied to Luis' SHU point-pattern generator (NPx_x, NPx_y, N_reciprocal,
    subsampling, sub_subsampling) are not recoverable from the ctl/csv and are omitted.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "Luis Data" / "Cavity_Like_Structures"

# floats as written by Fortran / MPB: 1.0, -2.98E-01, 3.4d+00
_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[dDeE][-+]?\d+)?"
_CYLINDER_CENTER = re.compile(
    r"\(center\s*\(c->l\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*\)\s*\)"
)
_BAND_COL = re.compile(r"^(t[em])?\s*band\s+(\d+)$", re.IGNORECASE)


def _to_float(text: str) -> float:
    return float(text.strip().replace("D", "E").replace("d", "e"))


# --------------------------------------------------------------------------- ctl


def parse_ctl(path: Path) -> dict:
    """Pull the physics out of an MPB ctl file: eps, radius, lattice, k-path, rod centers."""
    text = path.read_text(errors="replace")

    def scalar(name, pattern):
        m = re.search(pattern, text)
        return _to_float(m.group(1)) if m else None

    out = {
        "eps": scalar("eps", r"define-param\s+eps\s+(" + _FLOAT + r")"),
        "radius": scalar("rad", r"define-param\s+rad\s+(" + _FLOAT + r")"),
        "resolution": scalar("resolution", r"set!\s+resolution\s+(" + _FLOAT + r")"),
        "num_bands": scalar("num-bands", r"set!\s+num-bands\s+(" + _FLOAT + r")"),
        "interpolate": scalar("interpolate", r"interpolate\s+(" + _FLOAT + r")"),
        "polarization": "tm" if re.search(r"\(run-tm\)", text) else ("te" if re.search(r"\(run-te\)", text) else None),
    }

    m = re.search(r"make lattice \(size\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")", text)
    if m:
        out["lattice_size"] = np.array([_to_float(m.group(1)), _to_float(m.group(2))], dtype=np.float64)

    for label in ("Gamma", "M", "X"):
        m = re.search(r"define\s+" + label + r"\s+\(vector3\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")", text)
        if m:
            out[label] = np.array([_to_float(m.group(1)), _to_float(m.group(2))], dtype=np.float64)

    centers = [(_to_float(a), _to_float(b)) for a, b, _c in _CYLINDER_CENTER.findall(text)]
    if centers:
        out["posics"] = np.asarray(centers, dtype=np.float64).T  # (2, Npartic), cartesian, units of a

    return out


# --------------------------------------------------------------------------- csv


def parse_bands_csv(path: Path) -> dict:
    """Read an MPB bands CSV -> k-points and a (Nbands, Nk) frequency array."""
    with path.open() as fh:
        header = [c.strip() for c in fh.readline().split(",")]
        rows = [line for line in (ln.strip() for ln in fh) if line]

    band_cols = [i for i, name in enumerate(header) if _BAND_COL.match(name)]
    if not band_cols:
        raise ValueError(f"{path.name}: no 'band N' columns in header")
    first_band = band_cols[0]
    if band_cols != list(range(first_band, len(header))):
        raise ValueError(f"{path.name}: band columns are not contiguous to the end of the row")

    pol_tags = {m.group(1).lower() for m in (_BAND_COL.match(header[i]) for i in band_cols) if m.group(1)}
    if len(pol_tags) > 1:
        raise ValueError(f"{path.name}: mixed polarizations {sorted(pol_tags)}; expected a single run-tm/run-te")

    def col(name):
        try:
            return header.index(name)
        except ValueError:
            return None

    i_k1, i_k2, i_kmag = col("k1"), col("k2"), col("kmag/2pi")

    n_k = len(rows)
    n_bands = len(band_cols)
    freqs = np.full((n_k, n_bands), np.nan, dtype=np.float64)
    k_array = np.full((n_k, 2), np.nan, dtype=np.float64)
    kmag = np.full(n_k, np.nan, dtype=np.float64)

    for r, line in enumerate(rows):
        fields = line.split(",")
        if len(fields) != len(header):
            raise ValueError(f"{path.name}: row {r + 2} has {len(fields)} fields, header has {len(header)}")
        for c, i in enumerate(band_cols):
            token = fields[i].strip()
            if token:
                freqs[r, c] = float(token)
        if i_k1 is not None:
            k_array[r] = (float(fields[i_k1]), float(fields[i_k2]))
        if i_kmag is not None:
            kmag[r] = float(fields[i_kmag])

    return {
        "all_bands": freqs.T,  # (Nbands, Nk)
        "k_array": k_array,
        "kmag_over_2pi": kmag,
        "polarization": pol_tags.pop() if pol_tags else None,
    }


# ------------------------------------------------------------------- discovery


def discover(root: Path) -> dict:
    """Group CSVs by (chi, eps, source directory). Returns {key: [(sample_id, csv, ctl), ...]}."""
    groups = defaultdict(list)
    for csv_path in sorted(root.rglob("*.csv")):
        name = csv_path.stem
        m_chi = re.search(r"chi[_-]?(\d+(?:\.\d+)?|0p\d+)", name, re.IGNORECASE)
        m_eps = re.search(r"eps[_-]?(\d+(?:[.p]\d+)?)", name, re.IGNORECASE)
        if m_chi is None:
            print(f"  skip (no chi in name): {csv_path.name}")
            continue
        chi = float(m_chi.group(1).replace("p", "."))
        eps = float(m_eps.group(1).replace("p", ".")) if m_eps else None

        ctl = None
        stem = re.sub(r"_eps[_-]?[\d.p]+$", "", name)
        for cand in (csv_path.with_name(stem + ".ctl"), csv_path.with_suffix(".ctl")):
            if cand.exists():
                ctl = cand
                break
        if ctl is None:  # fall back to the sample's numeric prefix
            prefix = name.split("_")[0]
            matches = sorted(csv_path.parent.glob(f"{prefix}*.ctl"))
            ctl = matches[0] if matches else None

        sample_id = name.split("_")[0]
        groups[(chi, eps, csv_path.parent)].append((sample_id, csv_path, ctl))
    return groups


# ----------------------------------------------------------------- conversion


def build_group(chi, eps, entries, keep_closing_k: bool) -> dict:
    """Stack the samples of one (chi, eps) group into the band_sampling dictionary."""
    bands, posics, sample_ids, ctl_meta = [], [], [], None
    k_array = kmag = None

    for sample_id, csv_path, ctl_path in entries:
        csv = parse_bands_csv(csv_path)
        meta = parse_ctl(ctl_path) if ctl_path is not None else {}

        if bands and csv["all_bands"].shape != bands[0].shape:
            raise ValueError(
                f"{csv_path.name}: shape {csv['all_bands'].shape} != {bands[0].shape} of "
                f"{entries[0][1].name}; samples in one group must share Nbands and Nk"
            )
        if k_array is None:
            k_array, kmag, ctl_meta = csv["k_array"], csv["kmag_over_2pi"], meta
        elif not np.allclose(k_array, csv["k_array"], equal_nan=True):
            raise ValueError(f"{csv_path.name}: k-path differs from {entries[0][1].name}")

        bands.append(csv["all_bands"])
        sample_ids.append(sample_id)
        if "posics" in meta:
            posics.append(meta["posics"])
        n_nan = int(np.isnan(csv["all_bands"]).sum())
        if n_nan:
            print(f"    warning: {csv_path.name} has {n_nan} missing frequencies (stored as NaN)")

    ctl_meta = ctl_meta or {}
    all_bands = np.stack(bands, axis=-1)  # (Nbands, Nk, Nsamples)

    # Gamma-X-M-Gamma closes on itself: the last k-point duplicates the first.
    closes = k_array.shape[0] > 1 and np.allclose(k_array[0], k_array[-1])
    if closes and not keep_closing_k:
        all_bands = all_bands[:, :-1, :]
        k_array = k_array[:-1]
        kmag = kmag[:-1]
        print("    dropped the duplicated closing k-point (use --keep-closing-k to retain it)")

    n_bands, n_k, n_samples = all_bands.shape
    lattice = ctl_meta.get("lattice_size", np.array([1.0, 1.0]))
    # cumulative arc length along the path, in units of 1/a (same units as kmag/2pi)
    dk = np.diff(k_array, axis=0) / lattice[None, :]
    k_path_length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(dk, axis=1))])

    out = {
        "all_bands": all_bands,
        "Nbands": np.int32(n_bands),
        "Nk": np.int32(n_k),
        "Nsamples": np.int32(n_samples),
        "chi": np.float32(chi),
        "k_array": k_array,
        "kmag_over_2pi": kmag,
        "k_path_length": k_path_length,
        "sample_ids": np.array(sample_ids, dtype=h5py.string_dtype()),
        "lattice_size": np.asarray(lattice, dtype=np.float64),
    }

    eps_value = eps if eps is not None else ctl_meta.get("eps")
    if eps_value is not None:
        out["eps"] = np.float32(eps_value)
    for key in ("radius",):
        if ctl_meta.get(key) is not None:
            out[key] = np.float32(ctl_meta[key])
    for key in ("resolution", "interpolate"):
        if ctl_meta.get(key) is not None:
            out[key] = np.int32(ctl_meta[key])
    if ctl_meta.get("polarization"):
        out["polarization"] = ctl_meta["polarization"]

    # high-symmetry points and their 1-based index along the stored k path
    for label, short in (("Gamma", "G"), ("X", "X"), ("M", "M")):
        point = ctl_meta.get(label)
        if point is None:
            continue
        out[label] = point
        hits = np.flatnonzero(np.all(np.isclose(k_array, point[None, :]), axis=1))
        if hits.size:
            out[f"{short}_index"] = np.int32(hits[0] + 1)  # 1-based, as in the existing files

    if len(posics) == n_samples and all(p.shape == posics[0].shape for p in posics):
        out["all_posics"] = np.stack(posics, axis=-1)  # (2, Npartic, Nsamples)
        out["Npartic"] = np.int32(posics[0].shape[1])
    elif posics:
        print(f"    warning: rod counts differ across samples {[p.shape[1] for p in posics]}; "
              "all_posics not written")

    return out


def save_dict_hdf5(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fh:
        for key, value in data.items():
            if isinstance(value, np.ndarray) and np.iscomplexobj(value):
                fh.create_dataset("re_" + key, data=value.real)
                fh.create_dataset("im_" + key, data=value.imag)
            else:
                fh.create_dataset(key, data=value)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="directory tree of MPB csv/ctl files")
    ap.add_argument("--output", type=Path, default=None, help="output directory (default: <input>_h5)")
    ap.add_argument("--layout", choices=("nested", "flat"), default="nested",
                    help="nested: <out>/<subdir>/chi_0.37_band_sampling.h5 (notebook-compatible name); "
                         "flat: <out>/chi_0.37_eps_43_band_sampling.h5")
    ap.add_argument("--keep-closing-k", action="store_true",
                    help="keep the closing Gamma that duplicates the first k-point")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    args = ap.parse_args(argv)

    root = args.input.resolve()
    if not root.is_dir():
        print(f"error: input directory not found: {root}", file=sys.stderr)
        return 1
    out_root = (args.output or root.parent / (root.name + "_h5")).resolve()

    print(f"input : {root}")
    print(f"output: {out_root}\n")

    groups = discover(root)
    if not groups:
        print("error: no MPB band CSVs found", file=sys.stderr)
        return 1

    written = 0
    for (chi, eps, source_dir), entries in sorted(groups.items(), key=lambda kv: (str(kv[0][2]), kv[0][0])):
        entries.sort(key=lambda e: e[0])
        rel = source_dir.relative_to(root)
        eps_tag = "" if eps is None else f"_eps_{eps:g}".replace(".", "p")
        label = f"chi={chi:g}" + (f", eps={eps:g}" if eps is not None else "") + f"  [{rel or '.'}]"
        print(f"{label}: {len(entries)} sample(s)")

        missing_ctl = [s for s, _c, ctl in entries if ctl is None]
        if missing_ctl:
            print(f"    warning: no ctl found for sample(s) {missing_ctl}; metadata will be partial")

        data = build_group(chi, eps, entries, args.keep_closing_k)

        if args.layout == "nested":
            out_path = out_root / rel / f"chi_{chi:.2f}_band_sampling.h5"
        else:
            out_path = out_root / f"chi_{chi:.2f}{eps_tag}_band_sampling.h5"
        if out_path.exists() and not args.overwrite:
            print(f"    exists, skipping (use --overwrite): {out_path}")
            continue

        save_dict_hdf5(out_path, data)
        written += 1
        print(f"    all_bands {data['all_bands'].shape} (Nbands, Nk, Nsamples), "
              f"freq range [{np.nanmin(data['all_bands']):.4f}, {np.nanmax(data['all_bands']):.4f}] a/lambda")
        print(f"    -> {out_path}")

    print(f"\ndone: {written} file(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
