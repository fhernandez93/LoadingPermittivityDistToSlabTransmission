
#function to save the output of the sampling stored in a dictionary
def save_sampling(file,dictionary):
    import numpy as np
    import h5py
    with h5py.File(file, 'w') as f:
        for key in dictionary.keys():
            data=dictionary[key]
            #checking if the value is numpy array:
            if type(data) is np.ndarray:
                #checking whether the entries are complex or not
                if (data.dtype=='complex128') or (data.dtype=='complex64'):
                    #if so writing real and imagiinary parts in separate sets
                    dset = f.create_dataset('re_'+key, data = data.real)
                    dset = f.create_dataset('im_'+key, data = data.imag)
                else:
                    dset = f.create_dataset(key, data = data)
            else: #if not we just print
                dset = f.create_dataset(key, data = data)
    return



#function to read a full sampling from a hdf5 file
def read_sampling(file):
    import numpy as np
    import h5py
    #creating the dictionary
    out_data={}
    with h5py.File(file, 'r') as f:
        for key in f.keys():
            print('key=',key)
            #out_data[key]=f[key]
            #out_data[key]=f.get(key).value
            out_data[key]=f[key][()]
    return out_data 

  
