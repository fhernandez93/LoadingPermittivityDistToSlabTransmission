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
import imageio
import matplotlib
import gc
import sys
import io


# Assuming /AutomationModule is in the root directory of your project
sys.path.append(os.path.abspath(rf'H:\phd stuff\tidy3d'))

from AutomationModule import * 

import AutomationModule as AM

tidy3dAPI = os.environ["API_TIDY3D_KEY"]


folder_str = "11_29_2024 Beam Spreading Freq Domain Lossy Background"
folders = []
L_abs = [4]
for abs_length in L_abs:
    folders.append(rf"{folder_str}_{str(abs_length)}")


for item_folder in folders:
    folder =  rf"H:\phd stuff\tidy3d\data\{item_folder}"         
    data_flux_right = np.zeros(shape=(13,16,5,150))
    data_flux_left = np.zeros(shape=(13,16,5,150))
    x,y,f=[],[],[]


    for i,item in enumerate(os.listdir(folder)):
        chi_path = os.path.join(folder, item)
        print(item)
        for j,item2 in enumerate(natsorted(os.listdir(chi_path))):
            chi_freq_folder=os.path.join(chi_path, item2)
            for k,item3 in enumerate(natsorted(os.listdir(chi_freq_folder))):
                file=os.path.join(chi_freq_folder, item3)
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    sim_result = (AM.loadFromFile(key = tidy3dAPI, file_path=file, save_path=rf"F:\2D SHU Chi Statistics",only_download=False,get_ref=False))
                finally:
                    # Reset sys.stdout back to the original state
                    sys.stdout = old_stdout

                try:
                    data_flux_right[i,j,k]=(sim_result.sim_data['flux1'].flux.values) #Right
                    data_flux_left[i,j,k]=(sim_result.sim_data['flux2'].flux.values)#Left
                except:
                    print(item3)




    AM.create_hdf5_from_dict({
        "data_field_intensities": data_flux_right,
        "chi":np.array([0.2,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0]),
        "size":np.array([0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        "sample":np.linspace(0,4,5),
        "f":sim_result.sim_data.get_intensity("freq_monitorFieldOut").f.values,
    },rf"F:\2D SHU Chi Statistics\data\{item_folder}\fluxes_right.h5")

    AM.create_hdf5_from_dict({
        "data_field_intensities": data_flux_left,
        "chi":np.array([0.2,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0]),
        "size":np.array([0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        "sample":np.linspace(0,4,5),
        "f":sim_result.sim_data.get_intensity("freq_monitorFieldOut").f.values,
    },rf"F:\2D SHU Chi Statistics\data\{item_folder}\fluxes_left.h5")