
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

# Assuming /AutomationModule is in the root directory of your project
sys.path.append(os.path.abspath(fr'H:\phd stuff\tidy3d'))

from AutomationModule import * 

import AutomationModule as AM

tidy3dAPI = os.environ["API_TIDY3D_KEY"]

folders = [
        # rf"H:\phd stuff\tidy3d\data\09_10_2024 Beam Spreading\z_incidence",
        rf"H:\phd stuff\tidy3d\data\08_23_2024 chi 0.3 N1000 Sample Beam Spreading\z_incidence"
        # rf"H:\phd stuff\tidy3d\data\09_10_2024 Beam Spreading 0.2 - 0.48\z_incidence",
        # rf"H:\phd stuff\tidy3d\data\09_10_2024 Beam Spreading 0.2 - 0.48 Periodic Conditions\z_incidence"
    ]
data = []
for item in folders:
    folder_path = item
    for i,filename in enumerate(natsorted(os.listdir(folder_path))):
        if i == 1:
            continue
        file=os.path.join(folder_path, filename)
        sim_result = (AM.loadFromFile(key = tidy3dAPI, file_path=file))
        lambdas = 1/sim_result.monitor_lambdas
        data += [{"data":sim_result,"lambdas":lambdas, "filename":filename}]


for i,item in enumerate(data):
    intensity = item["data"].sim_data.get_intensity("time_monitorFieldOut")
    # List of target values
    target_times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,21,22,25,28,30,32,34,35])*1e-12
    # target_times = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8])*1e-12

    # Find the closest indices in intensity["t"] for each value in target_values
    closest_time_indices = [np.argmin(np.abs(intensity["t"].values - target)) for target in target_times]
    item["field_time_out"]=intensity[np.where(np.bool_(intensity["x"]>=-50) & np.bool_(intensity["x"]<=50))[0],np.where(np.bool_(intensity["y"]>=-12.45) & np.bool_(intensity["y"]<=12.45))[0],:,closest_time_indices]
    del intensity

sc = data[0]["field_time_out"].squeeze()
sc = sc/np.max(sc,axis=(0,1))
del data

# Plot
import matplotlib.colors as mcolors
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

colors = [
          (1, 1, 1),  # White
          (0, 0, 1),  # Blue
          (1, 1, 0),  # Yellow
          (0, 1, 0),  # Green
          (1, 0, 0), #Red
          (0, 0, 0), # Black
          ] 
n_bins = 500  # Number of bins for smooth transition
cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors, N=n_bins)
# Create the indices for each dimension
x = sc.x
y =sc.y
z = sc.t*1e12

# Use meshgrid to create the 3D grid of coordinates
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
val_norm = (sc.values.flatten())
val_norm[val_norm<1e-3]=np.nan
# Plot the 3D scatter plot
# ax.scatter(X.flatten(), Y.flatten(),Z.flatten(), c=val, cmap=cmap)
# plt.tight_layout()
# # Show the plot
# plt.show()



import plotly.graph_objs as go
mask = (~np.isnan(val_norm)) & (val_norm != 0)
# Define bins for different opacity levels
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
opacities = [0, 0.4, 0.6, 0.8, 1]  # Define corresponding opacity levels
color_scales = [["white","blue"], ["blue","yellow"], ["yellow","green"], ["green","red"], ["red","black"]]
traces = []



# Loop over each bin and create a separate trace for each opacity level
for i in range(len(bins) - 1):
    # Create a mask for the current bin range
    bin_mask = (val_norm[mask] >= bins[i]) & (val_norm[mask] < bins[i+1])
# Create a Scatter3d trace for the current bin
    trace = go.Scatter3d(
        x=X.flatten()[mask][bin_mask],
        y=Y.flatten()[mask][bin_mask],
        z=Z.flatten()[mask][bin_mask],
        showlegend=False,
        mode='markers',
        marker=dict(
            size=8,
            color=val_norm[mask][bin_mask],  # Use values for color
            colorscale=color_scales[i],  # Color scale
            opacity=opacities[i]  # Use different opacities for each bin

        ),
    )
    
    # Append the trace to the list of traces
    traces.append(trace)


trace = go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='markers',
        showlegend=False,
        marker=dict(
            size=8,
            color=val_norm[mask][bin_mask],  # Use values for color
            colorscale=["white","blue","yellow","green","red","black"],  # Color scale
            showscale=True,
                                 cmin=0,
                                 cmax=1,
               colorbar=dict(              # Add colorbar
                title="Array Values",   # Title for the colorbar
                thickness=15,
                len=0.5,                # Length of the colorbar
                xpad=5,                 # Padding from the right
            ),  # Use different opacities for each bin

        ),
    )

traces.append(trace)

# Create the figure and add all the traces
fig = go.Figure(data=traces)

# Define the layout of the plot
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='t[ps]',
        aspectmode='manual',
        aspectratio=dict(x=2, y=1, z=1)
    ),
   
)

# Show the plot
fig.show()
