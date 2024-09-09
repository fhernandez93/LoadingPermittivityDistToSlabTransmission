import h5py
import xarray as xr
import imageio
import matplotlib.pyplot as plt
import os
import numpy as np
import tidy3d as td

def _write_dict_to_hdf5(data, hdf_group):
    for key, value in data.items():
        if isinstance(value, dict):
            # If the value is a nested dictionary, create a group and call recursively
            subgroup = hdf_group.create_group(key)
            _write_dict_to_hdf5(value, subgroup)
        else:
            # If the value is not a dictionary, store it as a dataset
            hdf_group.create_dataset(key, data=value)
def create_hdf5_from_dict(data, filename):
    """ Data must have a structure like this, and all items are arrays 
     data = {
                        'transmission_right':transmission_flux_right,
                        'transmission_left':transmission_flux_left,
                        'transmission0':transmission0,
                        'lambdas':monitor_lambdas,
                        'decay':structure_1.final_decay
                  }
    
    """
    with h5py.File(filename, 'w') as hdf_file:
        # Recursively traverse the dictionary and write data to the HDF5 file
        _write_dict_to_hdf5(data, hdf_file)





def create_movie(field_time_out, monitor_lambdas,name='',type='t',log=False,path="",frames_per_second=1,rem_frames=False, normalize=True, a=1):
    frames = []
    max_values = field_time_out.max(axis=(0,1,2)) if normalize else 1

    field_time_out = field_time_out/max_values
    field_time_out['x'] = field_time_out['x']/a
    field_time_out['y'] = field_time_out['y']/a
    field_time_out['z'] = field_time_out['z']/a


    if log:
        field_log = np.log10((field_time_out))
        folder_pics = "logPics"
        
    else:
        field_log = field_time_out
        folder_pics = "linPics"

    if not os.path.exists(f'{path}/{folder_pics}'):
            os.makedirs(f'{path}/{folder_pics}')
            print(f"Folder {path}/{folder_pics} created successfully.")

    for i, element in enumerate(field_time_out['f'][()]):
        if os.path.isfile(f'{path}/{folder_pics}/frame_{i}.png'):
            frames.append(f'{path}/{folder_pics}/frame_{i}.png')
            continue
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            if type=="t":
                pcolormesh = (field_log).isel(t=i).squeeze().plot.pcolormesh(ax=ax,cmap="magma")
               
            else:
                pcolormesh = (field_log).isel(f=i).squeeze().plot.pcolormesh(ax=ax,cmap="magma")

            ax.set_aspect('equal', adjustable='box')
            try:
                plt.title(f'Time: {str(np.array(field_time_out['t'][()][i])*1e12)} ps')
            except:
                plt.title(f'$\\nu$: {(a/np.array(td.C_0/field_time_out['f'][()][i])):.4g}')


            # Save the frame
            plt.xlim(-9,9)
            plt.ylim(-9,9)
            plt.savefig(f'{path}/{folder_pics}/frame_{i}.png')
            plt.close(fig)
            frames.append(f'{path}/{folder_pics}/frame_{i}.png')
        except:
            break
        

    name_movie = f'{path}/{name}.mp4' if name else f'output/anderson/d(t) analysis/Diameter d(t) at output of the structure Range - {monitor_lambdas[0]:.3g} - {monitor_lambdas[-1]:.3g}.mp4'
    with imageio.get_writer(name_movie, fps=frames_per_second) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    # Optionally, remove the individual frames if no longer needed
    if rem_frames:
        for frame in frames:
            os.remove(frame)
    
    return False