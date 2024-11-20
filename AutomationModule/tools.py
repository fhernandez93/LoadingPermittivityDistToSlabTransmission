import h5py
import xarray as xr
import imageio
import matplotlib.pyplot as plt
import os
import numpy as np
import tidy3d as td
import matplotlib
import matplotlib.patches as patches
import matplotlib.colors as mcolors


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





def create_movie(sim_result, monitor_lambdas,name='',type='t',log=False,path="",frames_per_second=1,rem_frames=False, normalize=True, a=1, movie=False):
    # Define the colors: Blue -> White -> Yellow -> Green -> Red
    colors = [
              (1, 1, 1),  # White
              (0, 0, 1),  # Blue
              (1, 1, 0),  # Yellow
              (0, 1, 0),  # Green
              (1, 0, 0), #Red
              (0, 0, 0), # Black
              ]  
    
    # Create a custom colormap
    n_bins = 500  # Number of bins for smooth transition
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors, N=n_bins)
    field_time_out = sim_result.get_intensity("time_monitorFieldOut")
    fig1, ax_1 = plt.subplots()
    ax1=sim_result.simulation.plot_structures_eps(freq=monitor_lambdas[0], cbar=False, z=0, ax=ax_1,
                reverse=False)

    # Create a new figure for plotting the shapes
    patches_list = ax1.patches
    plt.close(fig1)
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

    for i, element in enumerate(range(np.shape(field_time_out)[-1])):
        if i%5 != 0:
            continue
        if os.path.isfile(f'{path}/{folder_pics}/frame_{i}.png'):
            frames.append(f'{path}/{folder_pics}/frame_{i}.png')
            continue
        fig, ax = plt.subplots(figsize=(5, 7))
        if type=="t":
            pcolormesh = (field_log).isel(t=i).squeeze().plot.pcolormesh(ax=ax,cmap=cmap)
           
        else:
            pcolormesh = (field_log).isel(f=i).squeeze().plot.pcolormesh(ax=ax,cmap=cmap)

        ax.set_aspect('equal', adjustable='box')
        try:
            plt.title(f'Time: {(np.array(field_time_out['t'][()][i])*1e12):.4g} ps')
        except:
            plt.title(f'$\\nu$: {(a/np.array(td.C_0/field_time_out['f'][()][i])):.4g}')
        ax.set_ylabel(rf"x",fontsize=16)
        ax.set_xlabel(rf"y",fontsize=16)
        ax.tick_params(which='major', labelsize=14)
        
        for patch in patches_list:
            path_patch = patch.get_path()  
            new_patch = patches.PathPatch(path_patch, edgecolor= (0,0,0, 0.08), facecolor='none')
            t2 = matplotlib.transforms.Affine2D().rotate_deg(90) + ax.transData
            new_patch.set_transform(t2)
            ax.add_patch(new_patch)
        # Save the frame
        plt.savefig(f'{path}/{folder_pics}/frame_{i}.pdf')
        plt.close(fig)
        frames.append(f'{path}/{folder_pics}/frame_{i}.png')
        

        
    if movie:
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