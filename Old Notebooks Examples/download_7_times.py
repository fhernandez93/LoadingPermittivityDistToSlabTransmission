
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
import AutomationModule as AM
from scipy.optimize import curve_fit
from natsort import natsorted
import numpy as np
import matplotlib.animation as animation
import xarray as xr
import imageio


tidy3dAPI = os.environ["API_TIDY3D_KEY"]

#Compute beam diameter d(t)
import scipy.integrate


def diameter(intensity_array):
    x,y = intensity_array['x'],intensity_array['y']
    intensity = intensity_array.values
    integral_1 = scipy.integrate.trapezoid(scipy.integrate.trapezoid(intensity, x=y, axis=1), x=x, axis=0)**2
    integral_2 = scipy.integrate.trapezoid(scipy.integrate.trapezoid(intensity**2, x=y, axis=1), x=x, axis=0)

    return 2*np.sqrt((integral_1/integral_2)/np.pi)

def create_movie(field_time_out, monitor_lambdas,name=''):
    frames = []
    field_log = np.log10((field_time_out))

    for i, time in enumerate(field_time_out['t']):
        fig, ax = plt.subplots()
        pcolormesh = (field_log).isel(t=i).squeeze().plot.pcolormesh(ax=ax,cmap="viridis")
        plt.title(f'Time step: {time.values}')

        # Save the frame
        plt.savefig(f'frame_{i}.png')
        plt.close(fig)
        frames.append(f'frame_{i}.png')

    name_movie = f'output/anderson/d(t) analysis/{name}.mp4' if name else f'output/anderson/d(t) analysis/Diameter d(t) at output of the structure Range - {monitor_lambdas[0]:.3g} - {monitor_lambdas[-1]:.3g}.mp4'
    with imageio.get_writer(name_movie, fps=6) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    # Optionally, remove the individual frames if no longer needed
    for frame in frames:
        os.remove(frame)

plt.figure(figsize=(15, 10))
a=5/3
slices =np.array([0.1])
for path_direction in [
                       "RCP 2 Tight pulse 36x36x(slicesx18a) 0.416-0.419 nu 7 times/z_incidence",
                       ]:

      folder_path = f"data/{path_direction}"
      

      
      for i,filename in enumerate(natsorted(os.listdir(folder_path))):
            print(filename)
            
            
            if not Path(filename).suffix==".txt":
                  continue
            
            if os.path.isfile(os.path.join(folder_path, filename)):
                file=os.path.join(folder_path, filename)
                structure_1 = AM.loadFromFile(key = tidy3dAPI, file_path=file)
                sim_data = structure_1.sim_data
                field_time_out = sim_data.get_intensity("time_monitorFieldOut")
                plt.plot(field_time_out['t']*1e12 ,diameter(field_time_out).squeeze()/a,"o", label=f"L=125x125x12.5a")

                if i == 0:
                      monitor_lambdas = a/(td.C_0/np.array(np.array(sim_data.simulation.monitors)[0].freqs))
                      name = Path(filename).name
                      #create_movie(field_time_out,monitor_lambdas,name)

            plt.ylabel("d(t)(in units of a)")
            plt.xlabel("Time[ps]")
            plt.title(f"L=125x125x12.5a Diameter d(t) - {monitor_lambdas[0]:.3g} - {monitor_lambdas[-1]:.3g}")
            plt.legend()
            plt.grid()
            plt.savefig(f'output/anderson/d(t) analysis/Diameter d(t) {Path(filename).name}.pdf', format='pdf')
            plt.show()
              


