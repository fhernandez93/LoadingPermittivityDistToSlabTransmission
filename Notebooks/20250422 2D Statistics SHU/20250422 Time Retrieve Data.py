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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from natsort import natsorted
import numpy as np
import matplotlib.animation as animation
import xarray as xr
import h5py
import imageio
import matplotlib
import gc
import sys
import io
import time

# Delay for 1 hour (3600 seconds)
time.sleep(5000)


# Assuming /AutomationModule is in the root directory of your project
sys.path.append(os.path.abspath(rf'H:\phd stuff\tidy3d'))

from AutomationModule import * 

import AutomationModule as AM

tidy3dAPI = os.environ["API_TIDY3D_KEY"]

folder = rf"H:\phd stuff\tidy3d\data\20250428 Beam Spreading Time L=14 Absorbers"

output_file = rf"G:\20250428 2D SHU Statistics\data\20250428 Beam Spreading Time L=14 Absorbers/intensities_exit_average.h5"  # HDF5 file path
Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

with h5py.File(output_file, 'w') as h5f:
 for i,item in enumerate(os.listdir(folder)):
    print(item) #chi
    chi_path = os.path.join(folder, item)
    chi_group = h5f.create_group(item)
    for j,item2 in enumerate(natsorted(os.listdir(chi_path))):
        print(item2) #freq range
        intensities = []
        fluxes = []
        chi_freq_folder=os.path.join(chi_path, item2)
        for k,item3 in enumerate(natsorted(os.listdir(chi_freq_folder))):
            #sample
            file=os.path.join(chi_freq_folder, item3)
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                try:
                    sim_result = (AM.loadFromFile(key = tidy3dAPI, file_path=file, save_path=rf"G:\20250428 2D SHU Statistics",only_download=False,get_ref=False))
                finally:
                    # Reset sys.stdout back to the original state
                    sys.stdout = old_stdout #seems to work to avoid memory related crashes 
            except:
               print("failed: ", item3)
               continue
        
               
            intensity = sim_result.sim_data.get_intensity("time_monitorFieldOut").squeeze()
            flux =sim_result.sim_data['time_monitorT'].flux.values
            intensity = intensity[np.where(np.bool_(intensity["x"]>=-49.5) & np.bool_(intensity["x"]<=49.5))[0],
                                  np.where(np.bool_(intensity["y"]>=6) & np.bool_(intensity["y"]<=7))[0][-1],
                                  :]
            intensities.append(intensity)
            fluxes.append(flux)

              
        avg_intensity = np.mean(intensities, axis=0)
        avg_flux = np.mean(fluxes, axis=0)
        chi_group.create_dataset(item2, data=avg_intensity)  # Save each frequency as a dataset
        chi_group.create_dataset(rf"{item2}_x",data=intensity["x"][np.where(np.bool_(intensity["x"]>=-49.5) & np.bool_(intensity["x"]<=49.5))])
        chi_group.create_dataset(rf"{item2}_t",data=intensity["t"])
        chi_group.create_dataset(rf"{item2}_flux",data=avg_flux)
        chi_group.create_dataset(rf"{item2}_flux_time",data=sim_result.sim_data['time_monitorT'].flux.t)



