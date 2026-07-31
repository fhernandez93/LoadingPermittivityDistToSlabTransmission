"""HDF5 dict I/O used by `Computing_nDOS.ipynb` and `PBG_boundaries.ipynb`.

Same behaviour as `IO_helper.save_sampling` / `read_sampling`, under the names the
notebooks import (`IO.read_dict_hdf5` / `IO.save_dict_hdf5`). Complex arrays are split
into `re_<key>` / `im_<key>` on write and recombined on read.
"""

import h5py
import numpy as np


def save_dict_hdf5(file, dictionary):
    """Write a flat dictionary to `file`, one dataset per key."""
    with h5py.File(file, "w") as f:
        for key, data in dictionary.items():
            if isinstance(data, np.ndarray) and np.iscomplexobj(data):
                f.create_dataset("re_" + key, data=data.real)
                f.create_dataset("im_" + key, data=data.imag)
            else:
                f.create_dataset(key, data=data)


def read_dict_hdf5(file, verbose=False):
    """Read every dataset of `file` into a flat dictionary."""
    out_data = {}
    with h5py.File(file, "r") as f:
        for key in f.keys():
            if verbose:
                print("key=", key)
            out_data[key] = f[key][()]

    for key in [k for k in out_data if k.startswith("re_")]:
        name = key[3:]
        if "im_" + name in out_data:
            out_data[name] = out_data.pop(key) + 1j * out_data.pop("im_" + name)

    for key, value in out_data.items():  # h5py returns bytes for strings
        if isinstance(value, bytes):
            out_data[key] = value.decode()
        elif isinstance(value, np.ndarray) and value.dtype == object:
            out_data[key] = np.array([v.decode() if isinstance(v, bytes) else v for v in value.ravel()]
                                     ).reshape(value.shape)

    return out_data


# aliases matching IO_helper.py
save_sampling = save_dict_hdf5
read_sampling = read_dict_hdf5
