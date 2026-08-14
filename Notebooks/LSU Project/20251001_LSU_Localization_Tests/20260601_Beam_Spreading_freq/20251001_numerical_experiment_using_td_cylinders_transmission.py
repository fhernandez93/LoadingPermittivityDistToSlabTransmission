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


tidy3dAPI = os.environ["API_TIDY3D_KEY"]
# Function to create a solid cylinder with specified center coordinates
def create_cylinder_from_ends(top_center, bottom_center, radius):
    # Calculate height of the cylinder
    height = np.linalg.norm(np.array(top_center) - np.array(bottom_center))
    bottom_center=bottom_center

    # Calculate the vector direction of the cylinder
    axis_direction = np.array(top_center) - np.array(bottom_center)
    axis_direction /= np.linalg.norm(axis_direction) #Unitary vector to calculate rotation angle 

    # Calculate the rotation matrix to align cylinder with the given axis direction
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, axis_direction)
    rotation_angle = np.arccos(np.dot(z_axis, axis_direction))
    rotation_matrix = mesh.Mesh.rotation_matrix(rotation_axis, rotation_angle)
    # Create a 4x4 identity matrix
    matrix_4x4 = np.eye(4)

    # Insert the 3x3 matrix into the top-left corner of the new matrix
    matrix_4x4[:3, :3] = np.copy(rotation_matrix)

    # Set the fourth column and fourth row for homogeneous transformation
    # The typical homogeneous transformation uses [0, 0, 0, 1] for the last row and column
    matrix_4x4[3, :3] = [0, 0, 0]  # Fourth row
    matrix_4x4[:3, 3] = [0, 0, 0]  # Fourth column
    matrix_4x4[3, 3] = 1           # Bottom-right corner remains 1

    
   
    cylinder_center = tuple((np.array(top_center) + np.array(bottom_center))/2)
    trans = td.Transformed.translation(cylinder_center[0],cylinder_center[1],cylinder_center[2])
    transformed = td.Transformed(geometry=td.Cylinder(center=(0,0,0), radius=radius, length=height),transform=trans@matrix_4x4)

    return transformed

#### Parameters for the simulation
a=2.562629142772549
lambdas =a/np.array([0.39,0.50])
n=2.90
min_steps_per_lambda = 13
runtime_ps = 40e-12
ref=True
nfreqs = 250
id0=""
###############################



# T should cover the full run_time (40 ps), not just the 35 ps analysis window
ff  = 0.217                     # dielectric filling fraction
e1, e2 = 1.0, n**2            # air, n=3.3
b   = (3*(1-ff) - 1)*e1 + (3*ff - 1)*e2
e_eff = (b + np.sqrt(b**2 + 8*e1*e2))/4
# n_eff = np.sqrt(e_eff)
n_eff=1.0        
run = True

SCRIPT_DIR = Path(__file__).resolve().parent
folder_path = SCRIPT_DIR.parent / "Structures"
postprocess_results = []
t_slabx, t_slaby, t_slabz = 250, 250, 32
project_name = f"20260813_Beam_Spreading_{t_slabx}_{t_slaby}_{t_slabz}_transmission"
h5_bg = None

for dirpath, dirnames, filenames in os.walk(folder_path):
    try:
        for filename in filenames:
            if filename.endswith(".txt"):
                if not filename.startswith("20260804_slab_200x200x32_N854932_lsu_generated_healed"): #The name of the file has the size scaled to 0.8
                    continue
                if os.path.isfile(os.path.join(dirpath, filename)):
                    file=os.path.join(dirpath, filename)
                   
                    data = np.loadtxt(file)
                    print(f"Processing {project_name}...")
                    tops = (data[:,:3])/0.8
                    bottoms = (data[:,3:])/0.8
                    centers = (tops+bottoms)/2
                    radius = 0.42
                    cyl_group = []
                    
                    for cil,item in enumerate(tops):
                        if centers[cil][2] < -t_slabz/2-0.8 or centers[cil][2] > t_slabz/2+0.8:
                            continue
                        cyl_group.append(create_cylinder_from_ends(tops[cil], bottoms[cil], radius))
                    medium=td.Medium(permittivity=n**2)
                    structure = td.Structure(geometry=td.GeometryGroup(geometries=cyl_group),  medium=medium)
                    Lx, Ly, Lz =t_slabx, t_slaby, 45
                    
                    f_min, f_max = td.C_0/lambdas[0], td.C_0/lambdas[1]
                    f0 = (f_min + f_max) / 2
                    source_def = td.PlaneWave(
                        source_time = td.GaussianPulse(
                            freq0=f0,
                            fwidth=0.5*(f_max - f_min),
                            offset=10,
                            phase=0
                        ),
                        size= (td.inf, 
                              td.inf, 
                              0) 
                              ,
                        center=( 0, 
                                 0, 
                                (-t_slabz/2)-3),
                        direction='+',
                        name='planewave',
                        )
                    
                    freq_range = td.C_0/np.array(lambdas)
                    
                    monitor_freqs = np.linspace(freq_range[0],freq_range[1],nfreqs)
                    
                    transmission_monitor_exit = td.FluxMonitor(
                            center=[0,0,t_slabz/2+2],
                            size=[
                                    Lx,
                                    Ly,
                                    0
                                ],
                            freqs=monitor_freqs,
                            name="transmission_monitor_exit",
                        )

                    transmission_monitor_entry = td.FluxMonitor(
                            center=[0,0,-t_slabz/2-1],
                            size=[
                                    Lx,
                                    Ly,
                                    0
                                ],
                            freqs=monitor_freqs,
                            name="transmission_monitor_entry",
                        )
                    
                    dl = (lambdas[1] /min_steps_per_lambda) / n #  grids per smallest wavelength in medium
                    boundaries= td.BoundarySpec(
                        x=td.Boundary.periodic(),
                        y=td.Boundary.periodic(),
                        z=td.Boundary(plus=td.Absorber(num_layers=200),minus=td.Absorber(num_layers=200))
                    )
                    cube1 = td.Structure(geometry= td.Box(
                        center=(0,  0 , -(Lz/2+t_slabz/2)/2),
                        size=(
                                td.inf, 
                                td.inf, 
                                Lz/2-t_slabz/2
                              ),
                        ),medium=td.Medium(permittivity=n_eff**2))
                    cube2 = td.Structure(geometry= td.Box(
                        center=(0,  0 , (Lz/2+t_slabz/2)/2),
                        size=(
                                td.inf, 
                                td.inf, 
                                Lz/2-t_slabz/2
                              ),
                        ),medium=td.Medium(permittivity=n_eff**2))
                    sim = td.Simulation(
                        center = (0, 0, 0),
                        size = (Lx, Ly, Lz),
                        grid_spec = td.GridSpec.auto(min_steps_per_wvl=min_steps_per_lambda,wavelength=lambdas[0],
                                    dl_min=dl,
                                    max_scale=1.2),
                        sources = [source_def],
                        monitors = [transmission_monitor_exit, transmission_monitor_entry],
                        run_time = runtime_ps,
                        shutoff =1e-20, #Simulation stops when field has decayed to this 
                        boundary_spec = boundaries,
                        normalize_index = None,
                        structures = [structure,cube1,cube2],
                        subpixel=True,
                        medium=td.Medium(permittivity=1)
                        )
                    print(f"Simulation {project_name} created with {len(cyl_group)} cylinders.")
                    if run:
                        folder_desc = SCRIPT_DIR.parents[3] / "data" / project_name / f"n_{n:.2f}"
                        os.makedirs(folder_desc, exist_ok=True)
                        sim_name=rf"project_{project_name}_n_{n:.2f}"
                        if os.path.exists(os.path.join(folder_desc, sim_name+".txt")):
                            print("Exist!")
                        else:
                            if ref:
                                sim_ref = sim.copy(update={'structures':[]})
                                id0 =web.upload(sim_ref, folder_name=project_name,task_name=sim_name+"_reference", verbose=True)
                                web.start(id0)
                                web.monitor(id0)

                            id =web.upload(sim, folder_name=project_name,task_name=sim_name, verbose=True)
                            ids = id0 + '\n' + id
                            with open(os.path.join(folder_desc, sim_name+".txt"), "w") as file:
                                # Write the string to the file
                                file.write(ids)
                            web.start(id)
                            web.monitor(id)
                           
                    else: 
                        id =web.upload(sim, verbose=True,task_name="test") 
                        print(web.estimate_cost(id)  )
                        web.delete(id)
                        fig, ax = plt.subplots(1, tight_layout=True, figsize=(16, 8))
                        freqs_plot = (freq_range[0], freq_range[1])
                        sim.plot_eps(x=0, freq=freqs_plot[0], ax=ax)
                        plt.show()
    except Exception as e:
        print(f"Error processing {dirpath}: {e}")