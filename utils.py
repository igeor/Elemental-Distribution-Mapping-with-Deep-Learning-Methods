import torch 
import numpy as np 
import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import pandas as pd 

def open_spectra_image(file_path, hdf5_location='Experiments/__unnamed__/data'):
    """ Open a spectral image file.
    Args:
        file_path (str): The path to the spectral image file.
        hdf5_location (str): The location of the spectral image in the HDF5 file.
    Returns:
        np.ndarray: The spectral image.
    """
    # Open spectral image file
    f = h5py.File(file_path, 'r')
    spec_image = f[hdf5_location][()] # numpy.ndarray 
    f.close()
    
    return spec_image


def open_target_image(file_path, 
        target_elems=['S_K','K_K','Ca_K','Cr_K','Mn_K','Fe_K','Cu_K','Zn_K','Sr_K','Au_L','Hg_L','Pb_L'], 
        sep='  ', engine='python'):
    
    ## Open target image file (elemental_maps)
    df = pd.read_csv(file_path , sep=sep, engine=engine)
    target_image = np.array(df[target_elems])
    h, w = df['row'].iloc[-1] + 1, df['column'].iloc[-1] + 1
    target_image = target_image.reshape(h, w, len(target_elems))
    return target_image


def wav_to_kev(spectrum_range, a=0.010001, b=0.95529552):
    kev_axis = [(i * a - b) for i in spectrum_range]
    return np.array(kev_axis)