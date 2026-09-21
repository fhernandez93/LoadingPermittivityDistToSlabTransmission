from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()
import tidy3d as td
from tidy3d import web
import numpy as np
from pathlib import Path
from stl import mesh


import sys
import os

# Assuming /AutomationModule is in the root directory of your project
sys.path.append(str(Path(__file__).resolve().parents[4]))

from AutomationModule import * 

import AutomationModule as AM

def create_cylinder_from_ends(top_center, bottom_center, radius):
    '''td.Cylinder between two end-points. Axis-aligned only (woodpile rods / defect segments).'''
    p_top = np.asarray(top_center, dtype=float)
    p_bot = np.asarray(bottom_center, dtype=float)
    diff = p_top - p_bot
    nonzero = np.flatnonzero(np.abs(diff) > 1e-9)
    if len(nonzero) != 1:
        raise ValueError(f"End-points must differ in exactly one coordinate (axis-aligned rod); "
                         f"got top={p_top}, bottom={p_bot}, differing axes={nonzero}")
    axis = int(nonzero[0])
    length = float(abs(diff[axis]))
    center = tuple((p_top + p_bot) / 2)
    return td.Cylinder(center=center, radius=float(radius), length=length, axis=axis)


def woodpile_geometry_from_tables(rods, defects, params):
    '''Build the woodpile as one td.Structure (GeometryGroup of td.Cylinder) from the h5 tables.

    Returns (structure, n_cylinders, n_rods, n_defects).
    '''
    n_rods = len(np.atleast_1d(rods["x1"]))
    n_def = len(np.atleast_1d(defects["x1"])) if "x1" in defects else 0

    if n_def > 0:
        kappas = np.atleast_1d(defects["kappa"])
        if np.any(kappas < 0):
            raise NotImplementedError(
                f"Negative kappa found (min {kappas.min():.3f}): thinner/missing segments need the parent rod split "
                "into pieces (see woodpile_helpers._rod_pieces). Only positive (thicker) defects are supported here.")
        if np.any(np.atleast_1d(defects["minor_radius"]) < np.atleast_1d(rods["minor_radius"])[np.atleast_1d(defects["rod_index"])]):
            raise NotImplementedError("Defect thinner than its parent rod: negative defect, not supported (rod splitting needed).")

    cyls = []
    for i in range(n_rods):
        cyls.append(create_cylinder_from_ends(
            (rods["x2"][i], rods["y2"][i], rods["z2"][i]),
            (rods["x1"][i], rods["y1"][i], rods["z1"][i]),
            rods["minor_radius"][i]))
    for i in range(n_def):
        cyls.append(create_cylinder_from_ends(
            (defects["x2"][i], defects["y2"][i], defects["z2"][i]),
            (defects["x1"][i], defects["y1"][i], defects["z1"][i]),
            defects["minor_radius"][i]))

    structure = td.Structure(geometry=td.GeometryGroup(geometries=cyls),
                             medium=td.Medium(permittivity=float(params["permittivity"])))
    print(f"  geometry: {len(cyls)} cylinders = {n_rods} rods + {n_def} defects")
    return structure, len(cyls), n_rods, n_def


def spectral_sampling(T_ps, band_width, a, lambdas, n_current=None):
    '''Frequency sampling needed to IFFT-reconstruct an unaliased time window of T_ps picoseconds.

    band_width : width of one simulation band in a/lambda units
    n_current  : optionally, the linspace point count you currently use over the full range
    '''
    u_min, u_max = a / np.max(lambdas), a / np.min(lambdas)
    span = u_max - u_min

    dnu_max = 1 / (T_ps * 1e-12)          # Hz
    du_max = dnu_max * a / td.C_0         # a/lambda units

    n_per_band = int(np.ceil(band_width / du_max)) + 1
    n_bands = int(np.ceil(span / band_width))
    n_total = int(np.ceil(span / du_max)) + 1

    print(f"target window        : {T_ps} ps")
    print(f"u range              : [{u_min:.4f}, {u_max:.4f}]  (span {span:.4f})")
    print(f"max spacing          : d(nu) = {dnu_max/1e9:.2f} GHz,  d(a/lambda) = {du_max:.4e}")
    print(f"points per {band_width} band : {n_per_band}  "
          f"(actual window {a/(td.C_0 * band_width/(n_per_band-1))*1e12:.2f} ps)")
    print(f"bands to cover range : {n_bands}")
    print(f"points, full range   : {n_total}  "
          f"(actual window {a/(td.C_0 * span/(n_total-1))*1e12:.2f} ps)")
    if n_current is not None:
        du_cur = span / (n_current - 1)
        print(f"current {n_current} points   : d(a/lambda) = {du_cur:.4e} "
              f"-> window {a/(td.C_0*du_cur)*1e12:.2f} ps")
    return n_total
