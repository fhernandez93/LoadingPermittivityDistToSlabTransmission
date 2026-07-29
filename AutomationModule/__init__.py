# --- trimesh 4.x compatibility shim ---
# trimesh>=4 renamed Path3D.to_2D -> Path3D.to_planar (same `to_2D=` kwarg and
# `(planar, to_3D)` return). Tidy3D 2.9.1 still calls `.to_2D(...)` when slicing
# Transformed/mesh/polyslab geometries for plotting (e.g. sim.plot_eps on rotated
# cylinders), which otherwise raises "'Path3D' object has no attribute 'to_2D'".
try:
    import trimesh.path as _tm_path

    if not hasattr(_tm_path.Path3D, "to_2D") and hasattr(_tm_path.Path3D, "to_planar"):
        _tm_path.Path3D.to_2D = _tm_path.Path3D.to_planar
except Exception:
    pass
# --- end shim ---

from AutomationModule.loadStructures import *
from AutomationModule.loadFromFile import *
from AutomationModule.tools import *