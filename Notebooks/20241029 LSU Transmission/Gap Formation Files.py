import scipy.io as sio
from dataclasses import dataclass
from typing import List, Tuple
import os
from dotenv import load_dotenv
load_dotenv()
import tidy3d as td
from tidy3d import web
import numpy as np
from pathlib import Path
from stl import mesh
import matplotlib.pyplot as plt

import sys
import os

# Assuming /AutomationModule is in the root directory of your project
sys.path.append(os.path.abspath(fr'H:\phd stuff\tidy3d'))

from AutomationModule import * 

import AutomationModule as AM

tidy3dAPI = os.environ["API_TIDY3D_KEY"]

a = 3.4
lambdas = a/np.array([0.3,0.8])


# folder_path = rf"H:\phd stuff\tidy3d\structures\LSU H5\20250116"
folder_path = rf"H:\phd stuff\tidy3d\structures\LSU H5\20250124"
project_name = "02_07_2024 Florescu LSU H5 Abraham T(L) Calculation"
postprocess_results = []
runtime_ps = 25e-12
min_steps_per_lambda = 20
cuts = [0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.3,1.5,1.7,1.9,2]
for direction in ["z"]: 
    for f,filename in enumerate(os.listdir(folder_path)):
     for cut in cuts:
        if not (Path(filename).suffix==".h5" or Path(filename).suffix==".stl"):
            continue 

           
        if os.path.isfile(os.path.join(folder_path, filename)):
            file=os.path.join(folder_path, filename)
            structure_1 = AM.loadAndRunStructure(key = tidy3dAPI, file_path=file
                                            ,direction=direction, lambda_range=lambdas,
                                            box_size=14.3,runtime_ps=runtime_ps,min_steps_per_lambda=min_steps_per_lambda,
                                           scaling=1,shuoff_condtion=1e-20, verbose=True, 
                                           monitors=["flux"], freqs=280, 
                                           cut_condition=cut, source="planewave", absorbers=120, use_permittivity=False,sim_name=rf"{Path(filename).stem}_size_{cut}"
                                           )
            file_desc = rf"H:\phd stuff\tidy3d\data\{project_name}_perm_{structure_1.permittivity_value}\z_incidence\{structure_1.sim_name}.txt"
            try:
                print(file_desc)
                if os.path.exists(file_desc):
                    print("Exist!")
                else:
                   print("Creating...")
                   structure_1.run_sim(run_free=False,load=False,add_ref=True,folder_description=rf"{project_name}_perm_{structure_1.permittivity_value}")

                del structure_1
            except:
                print(file_desc + " Error!!!!!!!!!!!!!!!!!")

            





