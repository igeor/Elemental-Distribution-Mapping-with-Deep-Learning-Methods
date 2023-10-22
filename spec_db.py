import os
import glob
import torch
import numpy as np 
from scipy.signal import find_peaks

# Get the list of all pure elements files
list_of_files = glob.glob('./data/h5/pure/*.txt')

# Load all pure elements
pure_elements = {}
for file in list_of_files:
    # keep only the name of the element
    element = file.split('/')[-1].split('.')[0]
    pure_elements[element] = np.loadtxt(file)

# Normalize all pure elements
for element in pure_elements:
    pure_element =  pure_elements[element] / np.max(pure_elements[element])
    pure_spectrum = torch.tensor(pure_element).float()
    pure_elements[element] = pure_spectrum.clone().detach()
