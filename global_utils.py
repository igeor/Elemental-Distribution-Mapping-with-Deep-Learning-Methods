import torch 
import numpy as np 
import h5py
import pandas as pd 
import matplotlib.pyplot as plt

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