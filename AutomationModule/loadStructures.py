import numpy as np
import tidy3d.web as web
import tidy3d as td
from tidy3d.components.data.data_array import SpatialDataArray
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
import os
from tidy3d.plugins.dispersion import FastDispersionFitter, AdvancedFastFitterParam
import trimesh as tri


class loadAndRunStructure:
    """
    This class takes a disordered network permitivity distribution file and calculate the transmission coefficients using Tidy3d
    File types can be .stl for structures or .h5 for permittivity distributions. 
    If .stl is in place one must specify if we're working with constant index or specify a link with the refractive index dist. 

    flux_monitor records the transmitted flux through the slab
    flux_time_monitor Records time-dependent transmitted flux through the slab
    field_time_monitors record the E-fields throughout simulation volume at t=run_time/2 and t=run_time
    field_time_monitor_output Records E-fields at the output surface at Nt equally spaced times from 0 to run_time
    """

    def __init__(self, key:str="", file_path:str = "", direction:str="z", 
                 lambda_range: list= [], box_size:float = 0, runtime: int = 0, 
                 width:float=0.4, freqs:int=400,permittivity:float=1,
                 min_steps_per_lambda:int = 20, permittivity_dist:str="", scaling:float=1.0,shuoff_condtion:float=1e-7,
                 sim_mode:str = "transmission", subpixel:bool=True,extra_monitors:list=[], extra_sources:list=[]
                 ):
        if not key:
            raise Exception("No API key was provided")
        else:
            web.configure(key)

        if not file_path:
            raise Exception("No structure was provided")
        
        self.file_format = Path(file_path).suffix
        self.permittivity_dist = permittivity_dist
        
        if not  self.file_format in [".h5",".stl"]:
            raise Exception("No .h5 or .stl structure was provided")
        
        
        #import structure or permittivity distribution 
        self.file = file_path
        self.structure_name = Path(file_path).stem
        # Load HDF5 file
        if  self.file_format == ".h5":
            with h5py.File(self.file, 'r') as f:
                self.permittivity_raw = np.array(f['epsilon'])

        
        
        self.min_steps_per_lambda = min_steps_per_lambda
        if self.file_format == ".h5":
            self.permittivity_value =  np.max(self.permittivity_raw) if permittivity == 0 else permittivity 
        else:
            self.permittivity_value = permittivity
        
        self.sim_mode = sim_mode
        self.subpixel = subpixel
        self.extra_monitors = extra_monitors
        self.extra_sources = extra_sources
            
        self.scaling = scaling
        self.direction = direction
        self.dPML = 1.0
        self.lambda_range =np.array(lambda_range)
        self.freq_range = td.C_0/np.array(self.lambda_range)
        self.freq0 = np.sum(self.freq_range)/2 #central frequency of the source/monitors
        self.lambda0 = td.C_0 / self.freq0
        self.freqw  = width * (self.freq_range[1] - self.freq_range[0]) #This is the width of the gaussian source      
        # runtime
        self.shutoff = shuoff_condtion
        self.runtime = runtime
        self.t_stop = self.runtime/self.freqw
        self.Nfreq = freqs
        self.monitor_freqs = np.linspace(self.freq_range[0], self.freq_range[1], self.Nfreq)
        self.monitor_lambdas = td.constants.C_0 / self.monitor_freqs
        self.period = 1.0 # um, period along x- and y- direction
        # Grid size # um
        self.dl = (self.lambda_range[1] / self.min_steps_per_lambda) / np.sqrt(self.permittivity_value) #  grids per smallest wavelength in medium
        
        # space between slabs and PML
        self.spacing = self.dPML * self.lambda_range[0]
        self.t_slab=box_size*scaling
        self.sim_size = self.Lx, self.Ly, self.Lz = (
                                      self.t_slab+self.spacing*2 if direction == "x" else self.t_slab,
                                      self.t_slab+self.spacing*2 if direction == "y" else self.t_slab,
                                      self.t_slab+self.spacing*2 if direction == "z" else self.t_slab
                                      )
        
        self.sim = self.simulation_definition()
       


    def __str__(self):

        calculated_data_str = ('Simulation Parameters (wavelengths are expressed in um):\n' +
                               
            f'Lx: {self.Lx:.3g} Ly: {self.Ly:.3g} Lz: {self.Lz:.3g}\n'+
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
                  0 if self.direction == "z" else td.inf
                  ) if self.sim_mode=="transmission" 
                  else False
                  ,
            center=((-self.Lx*0.5+self.spacing*0.1) if self.direction == "x" else 0, 
                    (-self.Ly*0.5+self.spacing*0.1) if self.direction == "y" else 0, 
                    (-self.Lz*0.5+self.spacing*0.1) if self.direction == "z" else 0),
            direction='+',
            pol_angle=0,
            name='planewave',
            )
        ################Defining monitors###########################################
        monitor_1 = td.FluxMonitor(
            center = (
                        (self.Lx - self.spacing)*0.5 if self.direction == "x" else 0, 
                        (self.Ly - self.spacing)*0.5 if self.direction == "y" else 0, 
                        (self.Lz - self.spacing)*0.5 if self.direction == "z" else 0
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
                    (-self.Lx+self.spacing)*0.5 if self.direction =="x" else 0, 
                    (-self.Ly+self.spacing)*0.5 if self.direction =="y" else 0, 
                    (-self.Lz+self.spacing)*0.5 if self.direction =="z" else 0
                    ),
            size = (
                0 if self.direction == "x" else td.inf, 
                0 if self.direction == "y" else td.inf, 
                0 if self.direction == "z" else td.inf
                ),
            freqs = self.monitor_freqs,
            name='flux2'#To the left
        )
        
        ####################################################################
        #Defining permittivity distribution for structure 

        if self.file_format == ".h5":
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
        
        #Loading stl structure
        if self.file_format == ".stl":
            triangles = tri.load_mesh(self.file)
            
            triangles.remove_degenerate_faces()
            tri.repair.broken_faces(triangles)
            triangles.apply_scale(self.scaling)
            box = td.TriangleMesh.from_trimesh(
                triangles
                )
            #box = td.TriangleMesh.from_stl(
            #        filename=self.file,
            #        scale=self.scaling,  # the units are already microns as desired, but this parameter can be used to change units [default: 1]
            #        origin=(
            #            0,
            #            0,
            #            0,
            #        ),  # this can be used to set a custom origin for the stl solid [default: (0, 0, 0)]
            #        solid_index=None,  # sometimes, there may be more than one solid in the file; use this to select a specific one by index
            #    )
            
            if self.permittivity_dist!="": 
                fitter = FastDispersionFitter.from_url(self.permittivity_dist)
                fitter = fitter.copy(update={"wvl_range": (self.lambda_range[1], self.lambda_range[0])})
                advanced_param = AdvancedFastFitterParam(weights=(1,1))
                medium, rms_error = fitter.fit(max_num_poles=10, advanced_param=advanced_param, tolerance_rms=2e-2)
              
            else: 
                medium = td.Medium(permittivity=self.permittivity_value)
            
            # create a structure composed of the geometry and the medium
            slab = td.Structure(geometry=box, medium=medium)
            


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
            sources = definitions['sources']+self.extra_sources,
            monitors = definitions['monitors']+self.extra_monitors,
            run_time = definitions['run_time'],
            shutoff = self.shutoff, #Simulation stops when field has decayed to this 
            boundary_spec = definitions['boundary_spec'],
            normalize_index = None,
            structures = definitions['structures'],
            subpixel=self.subpixel

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
            task_name_def = f'{self.structure_name}_eps_{self.permittivity_value}_size_{self.t_slab:.3g}_runtime_{self.runtime:.3g}_lambdaRange_{self.lambda_range[0]:.3g}-{self.lambda_range[1]:.3g}_incidence_{self.direction}'
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
        
       