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


folder = rf"H:\phd stuff\tidy3d\data\20250505 2D Beam Spreading Antennas Freq analysis"
       
data_field_intensities = np.zeros(shape=(12,1,4,618,150)) #chi, size, sample, x, freq
x,y,f=[],[],[]

sizes = np.array([7])
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
                sim_result = (AM.loadFromFile(key = tidy3dAPI, file_path=file, save_path=rf"G:\2D SHU Antennas Freq",only_download=False,get_ref=False))
            finally:
                # Reset sys.stdout back to the original state
                sys.stdout = old_stdout
            try:
                intensity =(sim_result.sim_data.get_intensity("freq_monitorFieldOut").squeeze())
                indices_x = np.where(np.bool_(intensity["x"]>=-55/2) & np.bool_(intensity["x"]<=55/2))[0]
                indices_y = np.array([np.where(np.bool_(intensity["y"]<=(-4.2)))[0][-1]]) #-4.2 is where the back of the slab is located
                data_field_intensities[i,j,k]= ((intensity[indices_x,indices_y,:]).values).squeeze()

            except:
                print(np.shape((intensity[indices_x,indices_y,:]).values))
                print(np.shape(np.array(sim_result.sim_data.get_intensity("freq_monitorFieldOut").values.squeeze()*1e30)))
                print(item3)

            

            

AM.create_hdf5_from_dict({
    "data_field_intensities": data_field_intensities,
    "chi":np.array([0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0]),
    "size":sizes,
    "sample":np.array([0,1]),
    "x":sim_result.sim_data.get_intensity("freq_monitorFieldOut").x.values[indices_x],
    "f":sim_result.sim_data.get_intensity("freq_monitorFieldOut").f.values,
},rf"G:\2D SHU Antennas Freq\data\20250505 2D Beam Spreading Antennas Freq analysis\intensities_exit.h5")