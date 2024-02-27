import numpy as np
import tidy3d.web as web
import tidy3d as td
from tidy3d.components.data.data_array import SpatialDataArray
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
import os


class loadAndRunStructure:
    """
    This class takes a disordered network permitivity distribution file and calculate the transmission coefficients using Tidy3d
    """

    def __init__(self, key:str="", file_path:str = "", direction:str="z", 
                 lambda_range: list= [], box_size:float = 0, runtime: int = 0, 
                 width:float=0.4, freqs:int=400,permittivity:float=0,
                 min_steps_per_lambda:int = 20):
        if not key:
            raise Exception("No API key was provided")
        else:
            web.configure(key)

        if not file_path:
            raise Exception("No structure was provided")
        
        #import structure permittivity distribution 
        self.file = file_path
        self.structure_name = Path(file_path).stem
        # Load HDF5 file
        with h5py.File(self.file, 'r') as f:
            self.permittivity_raw = np.array(f['epsilon'])
        
        self.min_steps_per_lambda = min_steps_per_lambda
        self.permittivity_value =  np.max(self.permittivity_raw) if permittivity == 0 else permittivity
        
        self.direction = direction
        self.dPML = 1.0
        self.freq_range = td.C_0/np.array(lambda_range)
        self.lambda_range =np.array(lambda_range)
        self.freq0 = np.sum(self.freq_range)/2 #central frequency of the source/monitors
        self.lambda0 = td.C_0 / self.freq0
        self.freqw  = width * (self.freq_range[1] - self.freq_range[0]) #This is the width of the gaussian source      
        # runtime
        self.runtime = runtime
        self.t_stop = self.runtime/self.freqw
        self.Nfreq = freqs
        self.monitor_freqs = np.linspace(self.freq_range[0], self.freq_range[1], self.Nfreq)
        self.monitor_lambdas = td.constants.C_0 / self.monitor_freqs
        self.period = 1.0 # um, period along x- and y- direction
        # Grid size # um
        self.dl = (self.lambda_range[1] / 30) / np.sqrt(self.permittivity_value) # 30 grids per smallest wavelength in medium
        
        # space between slabs and PML
        self.spacing = self.dPML * lambda_range[0]
        self.t_slab=box_size
        self.sim_size = self.Lx, self.Ly, self.Lz = (
                                      self.t_slab+self.spacing*2+2 if direction == "x" else self.t_slab,
                                      self.t_slab+self.spacing*2+2 if direction == "y" else self.t_slab,
                                      self.t_slab+self.spacing*2+2 if direction == "z" else self.t_slab
                                      )
        
        self.sim = self.simulation_definition()
       


    def __str__(self):

        calculated_data_str = ('Simulation Parameters (wavelengths are expressed in um):\n' +
            f'eps: {self.permittivity_value:.3g} \n'+
            f'lambda_range: {self.lambda_range[1]:.3g} - {self.lambda_range[0]:.3g} um \n'+
            f"lambdaw (pulse) {td.C_0/self.freqw} \n"+
            f"lambda0 {self.lambda0} \n"+
            f"Total runtime <= {self.t_stop*1e12} ps \n"+
            f"dl (Cube Size) = {self.dl*1000} nm"
        )

        return calculated_data_str


    def checkConnection():
        return web.test()  
    
    def createSimObjects(self):
        #Defining Source
        source = td.PlaneWave(
            source_time = td.GaussianPulse(
                freq0=self.freq0,
                fwidth=self.freqw
            ),
            size=(0 if self.direction == "x" else td.inf, 
                  0 if self.direction == "y" else td.inf, 
                  0 if self.direction == "z" else td.inf),
            center=((-self.Lx+self.spacing)*0.5-0.8 if self.direction == "x" else 0, 
                    (-self.Ly+self.spacing)*0.5-0.8 if self.direction == "y" else 0, 
                    (-self.Lz+self.spacing)*0.5-0.8 if self.direction == "z" else 0),
            direction='+',
            pol_angle=0,
            name='planewave',
            )
        #Defining monitors
        monitor_1 = td.FluxMonitor(
            center = (
                        self.Lx/2 - self.spacing/2 if self.direction == "x" else 0, 
                        self.Ly/2 - self.spacing/2 if self.direction == "y" else 0, 
                        self.Lz/2 - self.spacing/2 if self.direction == "z" else 0
                        ),
            size = (
                0 if self.direction == "x" else td.inf, 
                0 if self.direction == "y" else td.inf, 
                0 if self.direction == "z" else td.inf, 
                ),
            freqs = self.monitor_freqs,
            name='flux1' #To the right 
        )
        monitor_2 = td.FluxMonitor(
            center = (
                    (-self.Lx+self.spacing)/2 + 1 if self.direction =="x" else 0, 
                    (-self.Ly+self.spacing)/2 + 1 if self.direction =="y" else 0, 
                    (-self.Lz+self.spacing)/2 + 1 if self.direction =="z" else 0
                    ),
            size = (
                0 if self.direction == "x" else td.inf, 
                0 if self.direction == "y" else td.inf, 
                0 if self.direction == "z" else td.inf
                ),
            freqs = self.monitor_freqs,
            name='flux2'#To the left
        )
        #Defining permittivity distribution for structure 
        Nx, Ny, Nz = np.shape(self.permittivity_raw)
        X = np.linspace(-self.t_slab/2,self.t_slab/2, Nx)
        Y = np.linspace(-self.t_slab/2, self.t_slab/2, Ny)
        Z = np.linspace(-self.t_slab/2, self.t_slab/2, Nz)
        coords = dict(x=X, y=Y, z=Z)

        permittivity_data = SpatialDataArray(self.permittivity_raw,coords=coords)
        dielectric = td.CustomMedium(permittivity=permittivity_data)

        #Defining structure 
        slab = td.Structure(
        geometry=td.Box(
            center=(0,  0 ,0),
            size=(
                  self.t_slab if self.direction == "x"  else td.inf, 
                  self.t_slab if self.direction == "y"  else td.inf, 
                  self.t_slab if self.direction == "z"  else td.inf
                  ),
        ),
        medium=dielectric,
        name='slab',
        )

        #Boundary conditions 

        boundaries= td.BoundarySpec(
            x=td.Boundary(plus=td.Absorber(num_layers=50),minus=td.Absorber(num_layers=50)) if self.direction=="x" else td.Boundary.periodic(),
            y=td.Boundary(plus=td.Absorber(num_layers=50),minus=td.Absorber(num_layers=50)) if self.direction=="y" else td.Boundary.periodic(),
            z=td.Boundary(plus=td.Absorber(num_layers=50),minus=td.Absorber(num_layers=50)) if self.direction=="z" else td.Boundary.periodic(),
        )

        return {
                "size":self.sim_size,
                "grid_spec": td.GridSpec.auto(min_steps_per_wvl=self.min_steps_per_lambda,wavelength=self.lambda0,dl_min=self.dl*1e-2), #I'll take this generic one which can be later improved
                "sources": [source],
                "monitors": [monitor_1,monitor_2],
                "run_time": self.t_stop,
                "boundary_spec": boundaries,
                "normalize_index": None,
                "structures": [slab]
                }
    
    def simulation_definition(self):
        definitions = self.createSimObjects()
        sim = td.Simulation(
            center = (0, 0, 0),
            size = definitions['size'],
            grid_spec = definitions['grid_spec'],
            sources = definitions['sources'],
            monitors = definitions['monitors'],
            run_time = definitions['run_time'],
            shutoff = 1e-7, #Simulation stops when field has decayed to this 
            boundary_spec = definitions['boundary_spec'],
            normalize_index = None,
            structures = definitions['structures']

            )
        
        return sim 
    
    def plot_sim_layout(self):
        sim = self.sim
        plt.figure(dpi=200)
        freqs_plot = (self.freq_range[0], self.freq_range[1])
        fig, ax = plt.subplots(1, tight_layout=True, figsize=(8, 4))
        if self.direction == "x":
            sim.plot_eps(z=0, freq=freqs_plot[0], ax=ax)
        elif self.direction == "z":
            sim.plot_eps(x=0, freq=freqs_plot[0], ax=ax)
        elif self.direction == "y":
            sim.plot_eps(z=0, freq=freqs_plot[0], ax=ax)
            
        plt.tight_layout()
        plt.show()

    def estimate_cost(self):
        sim = self.sim
        id =web.upload(sim,task_name="test_net")
        cost = web.estimate_cost(id)
        web.delete(id)
        return cost
    
    def run_sim(self,run_free:bool = True,folder_description:str="",max_grid_size:int = 100,max_time_steps:int=50e3, load:bool=True):
        """
        If run for free is set to True the simulation won't be executed if the predefined max grid size or time step values are surpassed. 
        To fix this, reduce the min_steps_per_wvl on the class definition, decrease run time, or set run_free to False.
        Submits a Simulation to server, starts running, monitors progress, downloads, and loads results as a SimulationData object.
        Pushes taskid into task_name_def.txt
        """

        sim = self.sim

        time_steps = sim.num_time_steps
        grid_size = sim.num_cells*1e-6

        if (time_steps < max_time_steps and grid_size < max_grid_size) or not run_free:

            folder_name = folder_description
            task_name_def = f'{self.structure_name}_size_{str(self.t_slab)}_runtime_{self.runtime}_lambdaRange_{self.lambda_range[0]}-{self.lambda_range[1]}_incidence_{self.direction}'
            #Normalization task
            sim0 = sim.copy(update={'structures':[]})
            id_0 =web.upload(sim0, folder_name=folder_name,task_name=task_name_def+'_0', verbose=False)
            web.start(task_id = id_0)
            web.monitor(task_id=id_0,verbose=False)
            
            id =web.upload(sim, folder_name=folder_name,task_name=task_name_def, verbose=False)
            web.start(task_id = id)
            web.monitor(task_id=id,verbose=False)

            #Store ids in an file 


            ids = id_0 + '\n' + id
            incidence_folder = self.direction+"_incidence"
            file_path = f"data/{folder_name}/{incidence_folder}/{task_name_def}.txt"
            # Check if the folder exists
            if not os.path.exists( f"data/{folder_name}/{incidence_folder}"):
                os.makedirs(f"data/{folder_name}/{incidence_folder}")
                print(f"Folder '{folder_name}/{incidence_folder}' created successfully.")

            # Open file in write mode
            with open(file_path, "w") as file:
                # Write the string to the file
                file.write(ids)

            
        else: 
            raise Exception("Reduce time steps or grid size")
        
        if load:
                sim_data0=web.load(id_0)
                sim_data=web.load(id)
                return (sim_data, sim_data0,task_name_def)
        else:
            return False
        
       