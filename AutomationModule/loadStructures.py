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
    Options for monitors:
    flux_monitor records the transmitted flux through the slab
    flux_time_monitor Records time-dependent transmitted flux through the slab
    field_time_monitors record the E-fields throughout simulation volume at t=run_time/2 and t=run_time
    field_time_monitor_output Records E-fields at the output surface at Nt equally spaced times from 0 to run_time
    Source can be either a planewave, a laser-like beam, or a continuous wave pulse
    """

    def __init__(self, key:str="", file_path:str = "", direction:str="z", 
                 lambda_range: list= [], box_size:float = 0, runtime: int = 0, 
                 width:float=0.4, freqs:int=400,permittivity:float=1, use_permittivity:bool=False,
                 min_steps_per_lambda:int = 20, permittivity_dist:str="", scaling:float=1.0,shuoff_condtion:float=1e-7,
                 sim_mode:str = "transmission", subpixel:bool=True, verbose:bool=False, monitors:list=[], cut_condition:float=1,
                 source:str="planewave", multiplicate_size:bool=False, tight_percentage:float=0.01, multiplication_factor:int = 1,pol_angle:float = 0,
                 ref_only:bool=False, absorbers:int=40, sim_name:str="",runtime_ps:float=0.0
                 ):
        if not key:
            raise Exception("No API key was provided")
        else:
            web.configure(key)

        if not file_path and not ref_only:
            raise Exception("No structure was provided")
        
        self.file_format = Path(file_path).suffix
        self.permittivity_dist = permittivity_dist
        
        if not  self.file_format in [".h5",".stl"] and not ref_only:
            raise Exception("No .h5 or .stl structure was provided")
        
        
        #import structure or permittivity distribution 
        self.file = file_path
        self.structure_name = Path(file_path).stem
        # Load HDF5 file
        if  self.file_format == ".h5":
            with h5py.File(self.file, 'r') as f:
                self.permittivity_raw = np.ceil(np.array(f['epsilon']))
                if cut_condition < 1 :
                    if direction == "x":
                        self.permittivity_raw=(self.permittivity_raw[:,:,:int(np.shape(self.permittivity_raw)[0]*cut_condition-1)])
                    elif direction == "y":
                        self.permittivity_raw=(self.permittivity_raw[:,:,:int(np.shape(self.permittivity_raw)[1]*cut_condition-1)])
                    elif direction == "z":
                        self.permittivity_raw=(self.permittivity_raw[:,:,:int(np.shape(self.permittivity_raw)[2]*cut_condition-1)])

        if use_permittivity:
            self.permittivity_raw[self.permittivity_raw>1] += (permittivity - np.max(self.permittivity_raw))
            self.permittivity_raw[self.permittivity_raw<1] = 1


        self.sim_name = sim_name
        self.absorbers = absorbers
        self.pol_angle = pol_angle
        self.ref_only = ref_only
        self.multiplicate_size = multiplicate_size
        self.multiplication_factor = multiplication_factor
        self.tight_percentage = tight_percentage
        self.source = source
        self.monitors = monitors
        self.min_steps_per_lambda = min_steps_per_lambda
        if self.file_format == ".h5":
            self.permittivity_value =  np.max(self.permittivity_raw) if permittivity == 1 else permittivity 
        else:
            self.permittivity_value = permittivity
        
        self.sim_mode = sim_mode
        self.subpixel = subpixel
        self.verbose = verbose
            
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
        self.t_stop = runtime_ps if runtime_ps>0 else self.runtime/self.freqw
        self.Nfreq = freqs
        self.monitor_freqs = np.linspace(self.freq_range[0], self.freq_range[1], self.Nfreq)
        self.monitor_lambdas = td.constants.C_0 / self.monitor_freqs
        self.period = 1.0 # um, period along x- and y- direction
        # Grid size # um
        self.dl = (self.lambda_range[1] / self.min_steps_per_lambda) / np.sqrt(self.permittivity_value) #  grids per smallest wavelength in medium

         
        # space between slabs and PML
        self.spacing = self.dPML * self.lambda_range[0]
        self.t_slab=box_size*scaling

        #t_slab in all different directions 
        self.t_slab_x = (self.t_slab*cut_condition if direction=="x" else self.t_slab)*(self.multiplication_factor if self.multiplicate_size else 1)
        self.t_slab_y = (self.t_slab*cut_condition if direction=="y" else self.t_slab)*(self.multiplication_factor if self.multiplicate_size else 1)
        self.t_slab_z = self.t_slab*cut_condition if direction=="z" else self.t_slab
       
        self.sim_size = self.Lx, self.Ly, self.Lz = (
                                      (self.t_slab_x+self.spacing*2 if direction == "x" else self.t_slab_x),
                                      (self.t_slab_y+self.spacing*2 if direction == "y" else self.t_slab_y),
                                      self.t_slab_z+self.spacing*2 if direction == "z" else self.t_slab_z
                                      )
        
        self.sim = self.simulation_definition()
       


    def __str__(self):

        calculated_data_str = ('Simulation Parameters (wavelengths are expressed in um):\n' +
                               
            f'Lx: {self.Lx:.3g} Ly: {self.Ly:.3g} Lz: {self.Lz:.3g}\n'+
            f'lambda_range: {self.lambda_range[1]:.3g} - {self.lambda_range[0]:.3g} um \n'+
            f"lambdaw (pulse) {td.C_0/self.freqw} \n"+
            f"lambda0 {self.lambda0} \n"+
            f"Total runtime <= {self.t_stop*1e12} ps \n"+
            f"dl (Cube Size) = {self.dl*1000} nm \n"
            f"Time Steps = {self.sim.num_time_steps}\n"
            f"Grid Points = {self.sim.num_cells*1e-6} million\n"
            f"eps = {self.permittivity_value}"
        )

        return calculated_data_str


    def checkConnection():
        return web.test()  
    
    def createSimObjects(self):
        #Defining Source
        self.source_def = td.PlaneWave(
            source_time = td.GaussianPulse(
                freq0=self.freq0,
                fwidth=self.freqw
            ),
            size= (0 if self.direction == "x" else td.inf, 
                  0 if self.direction == "y" else td.inf, 
                  0 if self.direction == "z" else td.inf
                  ) if self.source == "planewave"
                  else 
                  (
                    0  if self.direction == "x" else self.t_slab_x*self.tight_percentage,
                    0  if self.direction == "y" else self.t_slab_y*self.tight_percentage,
                    0  if self.direction == "z" else self.t_slab_z*self.tight_percentage

                  )
                  ,
            center=((-self.Lx*0.5+self.spacing*0.1) if self.direction == "x" else 0, 
                    (-self.Ly*0.5+self.spacing*0.1) if self.direction == "y" else 0, 
                    (-self.Lz*0.5+self.spacing*0.1) if self.direction == "z" else 0) if self.source == "planewave" 
                    
                    
                    else 

                    (
                        -self.t_slab_x/2-self.spacing/2 if self.direction == "x" else 0,
                        -self.t_slab_y/2-self.spacing/2 if self.direction == "y" else 0,
                        -self.t_slab_z/2-self.spacing/2 if self.direction == "z" else 0
                     
                     
                     )
                    
                    ,
            direction='+',
            pol_angle=self.pol_angle,
            
            name='planewave',
            )
        ################Defining monitors###########################################
        monitors_names = []
        if "flux" in self.monitors:
            self.monitor_1 = td.FluxMonitor(
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
            self.monitor_2 = td.FluxMonitor(
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

            monitors_names += [self.monitor_1,self.monitor_2]


        # Records time-dependent transmitted flux through the slab
        if "time_monitor" in self.monitors:
            self.time_monitorT = td.FluxTimeMonitor(
                    center=[
                        (self.Lx - self.spacing)*0.5 if self.direction == "x" else 0, 
                        (self.Ly - self.spacing)*0.5 if self.direction == "y" else 0, 
                        (self.Lz - self.spacing)*0.5 if self.direction == "z" else 0


                        ],
                    size=[
                        0 if self.direction == "x" else td.inf, 
                        0 if self.direction == "y" else td.inf, 
                        0 if self.direction == "z" else td.inf
                        ],
                    interval = 200,
                    name="time_monitorT",

                )
            monitors_names += [self.time_monitorT]

        # Records E-fields throughout simulation volume at t=run_time/2
        if "field_time_domain" in self.monitors:
            time_monitorH = td.FieldTimeMonitor(
                    center=[0.0, 0.0, 0.0],
                    size=[
                            self.t_slab_x+self.spacing if self.direction == "x" else td.inf,
                            self.t_slab_y+self.spacing if self.direction == "y" else td.inf,
                            self.t_slab_z+self.spacing if self.direction == "z" else td.inf
                        ],
                    start=self.t_stop / 2.0,
                    stop=self.t_stop / 2.0,
                    #interval_space=(10,10,10),
                    fields=["Ex", "Ey", "Ez"],
                    name="time_monitorH",
                    
                )

            # Records E-fields throughout simulation volume at t=run_time
            time_monitorFinal = td.FieldTimeMonitor(
                    center=[0.0, 0.0, 0.0],
                    size=[
                            self.t_slab_x+self.spacing if self.direction == "x" else td.inf,
                            self.t_slab_y+self.spacing if self.direction == "y" else td.inf,
                            self.t_slab_z+self.spacing if self.direction == "z" else td.inf
                        ],
                    start=self.t_stop,
                    stop=self.t_stop,
                    #interval_space=(10,10,10),
                    fields=["Ex", "Ey", "Ez"],
                    name="time_monitorFinal",
                )
            monitors_names+=[time_monitorH,time_monitorFinal]




        #Be carefull as this will record data at every time step for all frequencies 
        if "field_monitor" in self.monitors:

            field_monitor = td.FieldMonitor(
                    center=[0.0, 0.0, 0.0],
                    size=[
                            self.t_slab_x+self.spacing if self.direction == "y" else td.inf,
                            self.t_slab_y+self.spacing if self.direction == "x" else td.inf,
                            self.t_slab_z+self.spacing if self.direction == "z" else td.inf
                        ],
                    fields=["Ex","Ey","Ez"],
                    name="field_monitor",
                    freqs =self.monitor_freqs,
                    interval_space=(5,5,5)
                )
            monitors_names+=[field_monitor]


        if "permittivity_monitor" in self.monitors:
            eps_monitor = td.PermittivityMonitor(
                center=[0.0, 0.0, 0.0],
                size=[
                            self.t_slab_x+self.spacing if self.direction == "x" else td.inf,
                            self.t_slab_y+self.spacing if self.direction == "y" else td.inf,
                            self.t_slab_z+self.spacing if self.direction == "z" else td.inf
                        ],
                freqs=[self.freq0],
                name="eps_monitor",
            )
            monitors_names+=[eps_monitor]


        ####################################################################
        #Defining permittivity distribution for structure 
        if True:
            if self.file_format == ".h5":
                x_size,y_size,z_size = self.t_slab_x/self.multiplication_factor,self.t_slab_y/self.multiplication_factor,self.t_slab_z
                if not self.multiplicate_size:
                    Nx, Ny, Nz = np.shape(self.permittivity_raw)
                    X = np.linspace(-x_size/2,x_size/2, Nx)
                    Y = np.linspace(-y_size/2, y_size/2, Ny)
                    Z = np.linspace(-z_size/2, z_size/2, Nz)
                    coords = dict(x=X, y=Y, z=Z)

                    permittivity_data = SpatialDataArray(self.permittivity_raw,coords=coords)
                    dielectric = td.CustomMedium(permittivity=permittivity_data)

                    #Defining structure 
                    slab = td.Structure(
                    geometry=td.Box(
                        center=(0,  0 ,0),
                        size=(
                              x_size if self.direction == "x"  else td.inf, 
                              y_size if self.direction == "y"  else td.inf, 
                              z_size if self.direction == "z"  else td.inf
                              ),
                    ),
                    medium=dielectric,
                    name='slab',
                    
                    )

                ###This Code creates several cubes and place them together
                else: 
                    slabs = []
                    self.coordinates_slabs = []
                    
                    for i in range(self.multiplication_factor):
                        for j in range(self.multiplication_factor):
                            center_x = (i - ( self.multiplication_factor/ 2) + 0.5) * x_size
                            center_y = (j - ( self.multiplication_factor/ 2) + 0.5) * y_size
                            center_z = 0  # All cubes are centered on the z=0 plane
                            coord_item = {
                                    "X": (center_x - x_size/2, center_x + x_size/2),
                                    "Y": (center_y - y_size/2, center_y + y_size/2),
                                    "Z": (-z_size/2, z_size/2),
                                    "center": (center_x, center_y, center_z)
                                    }
                            
                            self.coordinates_slabs+=[coord_item]
    
                    for i,item in enumerate(self.coordinates_slabs):
                        Nx, Ny, Nz = np.shape(self.permittivity_raw)
                        X = np.linspace(item["X"][0],item["X"][1], Nx)
                        Y = np.linspace(item["Y"][0],item["Y"][1], Ny)
                        Z = np.linspace(-z_size/2, z_size/2, Nz)
                        coords = dict(x=X, y=Y, z=Z)
        
                        permittivity_data = SpatialDataArray(self.permittivity_raw,coords=coords)
                        dielectric = td.CustomMedium(permittivity=permittivity_data)
    
    
                        #Defining structure 
                        slab_i = td.Structure(
                        geometry=td.Box(
                            center= item["center"],
                            size=(
                                  x_size, 
                                  y_size,
                                  z_size
                                  ),
                        ),
                        medium=dielectric,
                        name=f'slab{i}',
                        )
    
                        slabs += [slab_i]
        


                #####This code creates a large permittivity array 
                # else: 
                #     slabs = []
                #     coordinates_slabs = []

                #     ##############Concatenate slabs in chunks#######################
                #     # Create a memory-mapped array with the desired final shape
                #     filename = r'F:\large_permittivity.dat'  # Specify the path for the memory-mapped file
                #     dtype = np.uint8 # Get the data type from the original array
                #     shape = (
                #                 np.shape(self.permittivity_raw)[0]*self.multiplication_factor,
                #                  np.shape(self.permittivity_raw)[1]*self.multiplication_factor,
                #                  np.shape(self.permittivity_raw)[2]
                #             )

                #     if True:
                #         final_permittivity = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
                #         for i in range(self.multiplication_factor): 
                #             for j in range(self.multiplication_factor):
                #                     # Compute the start indices for the current block
                #                     start_x = i * np.shape(self.permittivity_raw)[0]
                #                     start_y = j * np.shape(self.permittivity_raw)[1]
                #                     end_x = (i+1) * np.shape(self.permittivity_raw)[0]
                #                     end_y = (j+1) * np.shape(self.permittivity_raw)[1]
                #                     # Assign the block
                #                     final_permittivity[ 
                #                                             start_x:end_x,
                #                                             start_y:end_y,
                #                                             :
                #                                         ] = self.permittivity_raw


                #                     final_permittivity.flush()
                #     else: 
                #          final_permittivity = np.memmap(filename, dtype=dtype, mode='r', shape=shape)

                #     ################################################################################################

                #     for i in range(self.multiplication_factor):
                #         for j in range(self.multiplication_factor):
                #             center_x = (i - ( self.multiplication_factor/ 2) + 0.5) * x_size
                #             center_y = (j - ( self.multiplication_factor/ 2) + 0.5) * y_size
                #             center_z = 0  # All cubes are centered on the z=0 plane
                #             coord_item = {
                #                     "X": (center_x - x_size/2, center_x + x_size/2),
                #                     "Y": (center_y - y_size/2, center_y + y_size/2),
                #                     "Z": (-z_size/2, z_size/2),
                #                     "center": (center_x, center_y, center_z)
                #                     }

                #             coordinates_slabs+=[coord_item]

                #     X,Y,Z = np.array([]),np.array([]),np.array([])

                #     Nx, Ny, Nz = np.shape(final_permittivity)
                #     X = np.linspace(-self.t_slab_x/2,self.t_slab_x/2, Nx)
                #     Y = np.linspace(-self.t_slab_y/2, self.t_slab_y/2, Ny)
                #     Z = np.linspace(-self.t_slab_z/2, self.t_slab_z/2, Nz)
                #     coords = dict(x=X, y=Y, z=Z)

                #     #Defining structure 
                #     coords = dict(x=X, y=Y, z=Z)
                #     permittivity_data = SpatialDataArray(final_permittivity,coords=coords)
                #     self.dielectric = td.CustomMedium(permittivity=permittivity_data)


                #     slab_i = td.Structure(
                #     geometry=td.Box(
                #         center= (0,0,0),
                #         size=(
                #               self.t_slab_x if self.direction == "x"  else td.inf, 
                #               self.t_slab_y  if self.direction == "y"  else td.inf, 
                #               self.t_slab_z if self.direction == "z"  else td.inf
                #               ),
                #     ),
                #     medium=self.dielectric,
                #     name=f'slab{0}',
                #     )

                #     slabs += [slab_i]


            #Loading stl structure
            if self.file_format == ".stl":
                triangles = tri.load_mesh(self.file)
                triangles.remove_degenerate_faces()
                tri.repair.broken_faces(triangles)
                triangles.apply_scale(self.scaling)
                tri.repair.broken_faces(triangles)
                box = td.TriangleMesh.from_trimesh(
                    triangles
                    )


                if self.permittivity_dist!="": 
                    fitter = FastDispersionFitter.from_url(self.permittivity_dist)
                    fitter = fitter.copy(update={"wvl_range": (self.lambda_range[1], self.lambda_range[0])})
                    advanced_param = AdvancedFastFitterParam(weights=(1,1))
                    medium, rms_error = fitter.fit(max_num_poles=10, advanced_param=advanced_param, tolerance_rms=2e-2)

                else: 
                    medium = td.Medium(permittivity=self.permittivity_value)

                # create a structure composed of the geometry and the medium
                slab = td.Structure(geometry=box, medium=medium)
        else:
            slab = ""   
            slabs = []


        #Boundary conditions 

        boundaries= td.BoundarySpec(
            x=td.Boundary(plus=td.Absorber(num_layers=self.absorbers),minus=td.Absorber(num_layers=self.absorbers)) if self.direction=="x" else td.Boundary.periodic(),
            y=td.Boundary(plus=td.Absorber(num_layers=self.absorbers),minus=td.Absorber(num_layers=self.absorbers)) if self.direction=="y" else td.Boundary.periodic(),
            z=td.Boundary(plus=td.Absorber(num_layers=self.absorbers),minus=td.Absorber(num_layers=self.absorbers)) if self.direction=="z" else td.Boundary.periodic(),
        )

        #Mesh override structure 
        mesh_override = td.MeshOverrideStructure(
        geometry=td.Box(center=(0,0,0), size=(
                      self.t_slab_x if self.direction == "x"  else td.inf, 
                      self.t_slab_y if self.direction == "y"  else td.inf, 
                      self.t_slab_z if self.direction == "z"  else td.inf
                      )),
            dl=( (self.lambda_range[1] / (self.min_steps_per_lambda)) / np.sqrt(self.permittivity_value) #  grids per smallest wavelength in medium
            ,)*3
        )

        return {
                "size":self.sim_size,
                "grid_spec": td.GridSpec.auto(min_steps_per_wvl=self.min_steps_per_lambda,wavelength=self.lambda0,
                                            dl_min=self.dl,
                                            max_scale=1.2,), 
                "sources": [self.source_def],
                "monitors": monitors_names,
                "run_time": self.t_stop,
                "boundary_spec": boundaries,
                "normalize_index": None,
                "structures": [] if self.ref_only else ([slab] if not self.multiplicate_size else slabs)
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
            shutoff = self.shutoff, #Simulation stops when field has decayed to this 
            boundary_spec = definitions['boundary_spec'],
            normalize_index = None,
            structures = definitions['structures'],
            subpixel=self.subpixel

            )
        
        if "time_monitorFieldOut" in self.monitors:
            time_monitorFieldOut = td.FieldTimeMonitor(
                center = (
                            (self.t_slab_x)*0.5 if self.direction == "x" else 0, 
                            (self.t_slab_y)*0.5 if self.direction == "y" else 0, 
                            (self.t_slab_z)*0.5 if self.direction == "z" else 0
                            ),
                size = (
                    0 if self.direction == "x" else td.inf, 
                    0 if self.direction == "y" else td.inf, 
                    0 if self.direction == "z" else td.inf
                    ),
                    start=0,
                    stop=self.t_stop,
                    interval=int(sim.num_time_steps/100),
                    fields=["Ex", "Ey", "Ez"],
                    name="time_monitorFieldOut",
                    
                )
            
            sim = sim.copy(update={"monitors":list(sim.monitors)+[time_monitorFieldOut]})

        if "time_monitorFieldLateral" in self.monitors:
            time_monitorFieldLateral = td.FieldTimeMonitor(
                center = (
                           0,0,0
                            ),
                size = (
                    0 if self.direction == "z" else self.Lx, 
                    0 if self.direction == "y" else self.Ly, 
                    0 if self.direction == "x" else self.Lz
                    ),
                    start=0,
                    stop=20e-12,
                    interval=200,
                    fields=["Ex", "Ey", "Ez"],
                    name="time_monitorFieldLateral",
                    
                )
            
            sim = sim.copy(update={"monitors":list(sim.monitors)+[time_monitorFieldLateral]})

        if "freq_monitorFieldOut" in self.monitors:
            freq_monitorFieldOut = td.FieldMonitor(
                center = (
                            (self.t_slab_x)*0.5 if self.direction == "x" else 0, 
                            (self.t_slab_y)*0.5 if self.direction == "y" else 0, 
                            (self.t_slab_z)*0.5 if self.direction == "z" else 0
                            ),
                size = (
                    0 if self.direction == "x" else td.inf, 
                    0 if self.direction == "y" else td.inf, 
                    0 if self.direction == "z" else td.inf
                    ),
                    fields=["Ex","Ey","Ez"],
                    freqs =self.monitor_freqs,
                    name="freq_monitorFieldOut",
                    
                )
            
            sim = sim.copy(update={"monitors":list(sim.monitors)+[freq_monitorFieldOut]})

        if "time_monitorFieldCenter" in self.monitors:
            time_monitorFieldCenter = td.FieldTimeMonitor(
                center = [0,0,0],
                size = [
                    0 if self.direction == "x" else td.inf, 
                    0 if self.direction == "y" else td.inf, 
                    0 if self.direction == "z" else td.inf
                    ],
                    start=0,
                    stop=self.t_stop,
                    interval=int(sim.num_time_steps/100),
                    fields=["Ex", "Ey", "Ez"],
                    name="time_monitorFieldCenter"
                    
                )
            
            sim = sim.copy(update={"monitors":list(sim.monitors)+[time_monitorFieldCenter]})

        
        return sim 
    
    
    def plot_sim_layout(self):
        sim = self.sim
        plt.figure(dpi=200)
        freqs_plot = (self.freq_range[0], self.freq_range[1])
        fig, ax = plt.subplots(1, tight_layout=True, figsize=(16, 8))
        if self.direction == "x":
            sim.plot_eps(z=0, freq=freqs_plot[0], ax=ax)
        elif self.direction == "z":
            sim.plot_eps(x=0, freq=freqs_plot[0], ax=ax)
            plt.show()
            plt.figure(dpi=250)
            freqs_plot = (self.freq_range[0], self.freq_range[1])
            fig, ax = plt.subplots(1, tight_layout=True, figsize=(16, 8))
            sim.plot_eps(z=0, freq=freqs_plot[0], ax=ax)
            plt.show()
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
    
    def run_sim(self,run_free:bool = True,folder_description:str="",max_grid_size:int = 100,max_time_steps:int=50e3, 
                load:bool=True, run:bool=True,add_ref:bool=True):
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

            size = self.t_slab_x if self.direction == "x" else self.t_slab
            size = self.t_slab_y if self.direction == "y" else self.t_slab
            size = self.t_slab_z if self.direction == "z" else self.t_slab

            folder_name = folder_description
            task_name_def = f'{self.structure_name}_eps_{self.permittivity_value}_size_{size:.3g}_runtime_{self.runtime:.3g}_lambdaRange_{self.lambda_range[0]:.3g}-{self.lambda_range[1]:.3g}_incidence_{self.direction}' if not self.sim_name else self.sim_name 
            #Normalization task
            if add_ref:
                sim0 = sim.copy(update={'structures':[]})
                id_0 =web.upload(sim0, folder_name=folder_name,task_name=task_name_def+'_0', verbose=self.verbose)
                if run:
                    web.start(task_id = id_0)
                    web.monitor(task_id=id_0,verbose=self.verbose)

            
            id =web.upload(sim, folder_name=folder_name,task_name=task_name_def, verbose=self.verbose)
            if run:
                web.start(task_id = id)
                web.monitor(task_id=id,verbose=self.verbose)

            #Store ids in an file 

            if run:
                ids = (id_0 if add_ref else "") + '\n' + id
                incidence_folder = self.direction+"_incidence"
                file_path = rf"H:\phd stuff\tidy3d\data\{folder_name}\{incidence_folder}\{task_name_def}.txt"
                # Check if the folder exists
                if not os.path.exists( rf"H:\phd stuff\tidy3d\data\{folder_name}\{incidence_folder}"):
                    os.makedirs(rf"H:\phd stuff\tidy3d\data\{folder_name}\{incidence_folder}")
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
        
       