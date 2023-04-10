import torch 
import numpy as np 
import h5py
import pandas as pd 
import matplotlib.pyplot as plt

# %%
image_names=['gogo', 'saintjohn', 'dionisios', 'fanourios', 'odigitria', 'minos']
PATH = '/home/igeor/MSC-THESIS/data/h5'

image_filenames = [PATH + '/spec/' + x + '.hdf5' for x in image_names]
image_filenames

import utils

spectra_image = utils.open_spectra_image(image_filenames[0])



