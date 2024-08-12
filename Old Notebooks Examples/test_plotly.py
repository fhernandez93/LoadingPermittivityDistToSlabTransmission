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

import plotly.graph_objs as go
import plotly
from IPython.display import display, HTML
plotly.offline.init_notebook_mode()
display(HTML(
    '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
))
tidy3dAPI = os.environ["API_TIDY3D_KEY"]

a=5/3
slices = [0.01,0.03,0.05,0.07,0.09,0.1,0.15,0.3,0.5,0.7,0.9,1.0]

def Td(L,T0,xi,l:float=1.12):
    Db = 1+2*3.25/xi
    Td = (1+3.25)*(l/Db)/xi *np.exp(-L/xi)
    return T0*Td

def Tb(L,l):
    return np.exp(-L/l) 



for path_direction in [
                       "RCP Sample 1 Slices SAL Gap Formation/z_incidence"
                      
                       ]:

      folder_path = f"data/{path_direction}"
      fig = go.Figure()


      
      for i,filename in enumerate(natsorted(os.listdir(folder_path))):
            print(filename)
            
            if not Path(filename).suffix==".txt":
                  continue

            
            if os.path.isfile(os.path.join(folder_path, filename)):
                file=os.path.join(folder_path, filename)
                structure_1 = AM.loadFromFile(key = tidy3dAPI, file_path=file)
                sim_data, sim_data0 = structure_1.sim_data, structure_1.sim_data0
                transmission0 = sim_data0['flux1'].flux
                transmission = sim_data['flux1'].flux
                transmission_normalized = transmission / transmission0
                monitor_lambdas = a/(td.C_0/np.array(np.array(sim_data.simulation.monitors)[0].freqs))
                fig.add_trace(go.Scatter(x=monitor_lambdas, y=np.log10(transmission_normalized),
                                    mode='lines'
                                    ,name=f"L={slices[i]}x18.01a"
                            ))
      fig.update_layout(
                title=dict(text="Bandgap formation", font=dict(size=18)),
                xaxis_title='$\\nu$',
                yaxis_title='T',
                yaxis_gridcolor='lightgrey',  # Set grid color
                plot_bgcolor='rgba(0,0,0,0)'  # Set background color of plot,
                
                )
      #Save the plot in PDF format
      store_path = f"output/anderson/Plots_for_Report_20240613"
      if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f"Folder '{store_path}' created successfully.")
      
      fig.write_image(f"{store_path}/gap_evolution_slices.pdf",format='pdf')
      fig.show()
        


     

               

