import tidy3d.web as web
import tidy3d as td
import trimesh as tri 
from stl import mesh
import xarray as xr
from tidy3d.plugins.resonance import ResonanceFinder
from pathlib import Path
from tidy3d.components.data.data_array import SpatialDataArray
import h5py

class calculateBandStructure:
