import torch 
import numpy as np 
import pandas as pd 
import h5py
from torch.utils.data.dataset import Dataset

class Pixel2PixelDataset(Dataset):

    def __init__(self, dir_name, image_names=['minos'], sample_step = 1, device='cuda', band_range=(80, 2128), 
        target_elems=['S_K','K_K','Ca_K','Cr_K','Mn_K','Fe_K','Cu_K','Zn_K','Sr_K','Au_L','Hg_L','Pb_L'] ): 

        self.dir_name = dir_name
        self.image_names = image_names
        self.sample_step = sample_step
        self.device = device
        self.band_min, self.band_max = band_range
        self.target_elems = target_elems
        self.X_train = []; self.y_train = []

        self.train_spec_files = [self.dir_name + '/spec/' + x + '.hdf5' for x in self.image_names]
        self.train_elem_files = [self.dir_name + '/elem_maps/' + x +'.dat' for x in self.image_names]

        ## Initialize Train and Validation datasets
        for i_spec, spec_file in enumerate(self.train_spec_files):

            ## Open spectral image file
            f = h5py.File(spec_file, 'r')
            spec_image = f['Experiments/__unnamed__/data'][()] # numpy.ndarray 
            f.close()
            spec_image = spec_image.reshape(
                spec_image.shape[0]*spec_image.shape[1], spec_image.shape[2]
            )
            
            ## Open target image file (elemental_maps)
            target_file = spec_file.replace("spec", "elem_maps")
            target_file = target_file.replace("hdf5", "dat")
            df = pd.read_csv(target_file , sep='  ', engine='python')
            target_image = np.array(df[self.target_elems])

            ## Keep a subset of spectra dataset (default: 10%)
            for idx, spectra in enumerate(spec_image):
                if(idx % self.sample_step == 0):
                    x = np.expand_dims(spectra, axis=0).astype(np.float32)
                    x = torch.tensor(x)
                    self.X_train.append(x)
                    y = torch.tensor(target_image[idx].astype(np.float32))
                    self.y_train.append(y)

    def __getitem__(self, index):
        x_in = self.X_train[index][:,self.band_min:self.band_max]
        y_in = self.y_train[index]
        return x_in.to(self.device), y_in.to(self.device)

    def __len__(self):
        return len(self.X_train)


if __name__ == "__main__":
    directory = '/home/igeor/MSC-THESIS/data/h5'
    dataset = Pixel2PixelDataset(directory, sample_step=1, device='cuda', band_range=(80, 2128), 
                                 image_names=['gogo', 'saintjohn', 'dionisios', 'fanourios', 'odigitria', 'minos'])
    print(len(dataset))
    