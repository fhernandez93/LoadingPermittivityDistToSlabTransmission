import numpy as np
import tidy3d.web as web
import tidy3d as td
from tidy3d.components.data.data_array import SpatialDataArray
import h5py
import matplotlib.pyplot as plt
from pathlib import Path


class loadFromFile:
    """
    Load results from txt file 
    """
    def __init__(self, key:str="", file_path:str = ""):
        if not key:
            raise Exception("No API key was provided")
        else:
            web.configure(key)

        if not file_path:
            raise Exception("No file was provided")
        
        self.list_id = []
        with open(file_path, 'r') as file:
            for line in file:
                self.list_id+=[line.strip()] 

        self.structure_name = Path(file_path).stem

        self.sim_data0 = web.load(self.list_id[0])
        self.sim_data = web.load(self.list_id[1])

        self.cost = web.real_cost(self.list_id[1])
        self.run_time = self.sim_data.simulation.run_time
        self.fwidth=self.sim_data.simulation.sources[0].source_time.fwidth
        self.freq0=self.sim_data.simulation.sources[0].source_time.freq0
        self.run_time = self.sim_data.simulation.run_time*self.fwidth
        self.monitor_lambdas = td.C_0/np.array(np.array(self.sim_data.simulation.monitors)[0].freqs)
        self.final_decay = self.sim_data.final_decay_value
        self.description = Path(file_path).stem

    
    def __str__(self):

        calculated_data_str = ('Simulation Parameters (wavelengths are expressed in um):\n' +
            f'lambda_range: {self.monitor_lambdas[-1]:.3g} - {self.monitor_lambdas[0]:.3g} um \n'+
            f"lambdaw (pulse) {td.C_0/self.fwidth} \n"+
            f"lambda0 {td.C_0/self.freq0} \n"+
            f"Runtime = {self.run_time} \n"+
            f"final decay value = {self.final_decay} \n"+
            f"Cost = {self.cost}"
        )

        return calculated_data_str

        