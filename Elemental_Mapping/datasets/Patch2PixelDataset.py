import torch 
import numpy as np 
import pandas as pd 
import h5py
import random 
import glob
import os 
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.preprocessing import StandardScaler
import pickle

class Patch2PixelDataset(Dataset):

    def __init__(self, dir_name, patch_size=(5, 5), stride=3, sample=0.1, device='cuda', band_range=(0, 4096), transpose=True,
            return_index=False, image_names=['gogo', 'saintjohn', 'dionisios', 'fanourios', 'odigitria'], 
            target_elems=['S_K','K_K','Ca_K','Cr_K','Mn_K','Fe_K','Cu_K','Zn_K','Sr_K','Au_L','Hg_L','Pb_L'], apply_scaling=False): 

        self.dir_name = dir_name
        self.pw, self.ph = patch_size
        self.stride = stride
        self.sample = sample
        self.device = device
        self.band_min, self.band_max = band_range
        self.c = abs(self.band_max - self.band_min)
        self.transpose = transpose
        self.image_names = image_names
        self.target_elems = target_elems
        self.return_index = return_index
        self.apply_scaling = apply_scaling
        self.spec_file_paths = [self.dir_name + '/spec/' + x + '.hdf5' for x in self.image_names]
        self.target_file_paths = [self.dir_name + '/elem_maps/' + x +'.dat' for x in self.image_names]

        self.spec_images = []
        self.target_images = []
        self.image_indices = []
        ## Generate indices for all the patches
        for i, spec_image_path in enumerate(self.spec_file_paths):

            ## Open spectral image file
            f = h5py.File(spec_image_path, 'r')
            spec_image = f['Experiments/__unnamed__/data'][()] # numpy.ndarray 
            f.close()
            spec_image = np.swapaxes(spec_image, 1, 2)
            spec_image = np.swapaxes(spec_image, 0, 1)
            spec_image = spec_image[self.band_min:self.band_max, :, :]
            self.spec_images += [ spec_image ]
            
            ## Open target image file (elemental_maps)
            spec_image_path = spec_image_path.replace("spec", "elem_maps")
            spec_image_path = spec_image_path.replace("hdf5", "dat")
            df = pd.read_csv(spec_image_path , sep='  ', engine='python')
            w, h = df['row'].iloc[-1] + 1, df['column'].iloc[-1] + 1
            target_image = np.array(df[self.target_elems])
            target_image = np.swapaxes(target_image, 0, 1)
            target_image = np.reshape(target_image, (target_image.shape[0], w, h))
            self.target_images += [ target_image ]

            ## Extract patches and store as indices
            for x in range(0, w-self.pw+1, self.stride):
                for y in range(0, h-self.ph+1, self.stride):
                    self.image_indices.append((i, x, y))
        
        if self.apply_scaling:
            try:
                with open('./x_scaler.pkl', 'rb') as file:
                    self.x_scaler = pickle.load(file)
                with open('./y_scaler.pkl', 'rb') as file:
                    self.y_scaler = pickle.load(file)
            except:
                # Create a StandardScaler object to normalize the Input Data
                flattened_spec_images = [ image.reshape(self.c, -1) for image in self.spec_images ]
                stacked_flat_spec_images = np.concatenate(flattened_spec_images, axis=1)
                self.x_scaler = StandardScaler()
                self.x_scaler.fit(stacked_flat_spec_images.T)
                # Create a StandardScaler object to normalize the Target Data
                flattened_target_images = [ image.reshape(len(self.target_elems), -1) for image in self.target_images ]
                stacked_flat_target_images = np.concatenate(flattened_target_images, axis=1)
                self.y_scaler = StandardScaler()
                self.y_scaler.fit(stacked_flat_target_images.T)
                # Save the scaler to a file
                with open('x_scaler.pkl', 'wb') as file:
                    pickle.dump(self.x_scaler, file)
                with open('y_scaler.pkl', 'wb') as file:
                    pickle.dump(self.y_scaler, file)


    def __getitem__(self, index):
        img_idx, x, y = self.image_indices[index]
        target_index_w = x+self.pw//2
        target_index_h = y+self.ph//2
        ## Load spectra image and target image
        x_in = self.spec_images[img_idx][:, x:x+self.pw, y:y+self.ph]
        y_in = self.target_images[img_idx][:, target_index_w, target_index_h]
        ## Convert type to float32
        x_in = x_in.astype(np.float32)
        y_in = y_in.astype(np.float32)
        
        ## Apply the scaling 
        if self.apply_scaling:
            x_in_scaled = self.x_scaler.transform(x_in.reshape(self.c, -1).T).T
            x_in = x_in_scaled.reshape(self.c, self.pw, self.ph)
            y_in_scaled = self.y_scaler.transform(y_in.reshape(1, -1))

        ## Transform data to tensors and Add to device
        x_in = torch.tensor(x_in).to(self.device) 
        y_in = torch.tensor(y_in).to(self.device) 
        if self.transpose:
            ## Apply random horizontal flipping
            if random.random() > 0.5: x_in = TF.hflip(x_in)
            ## Apply random vertical flipping
            if random.random() > 0.5: x_in = TF.vflip(x_in)
        ## View spectra image as a single representation
        x_in = torch.unsqueeze(x_in, dim=0)
        
        if self.return_index:
            return (target_index_w, target_index_h), (x_in, y_in)  
        return x_in, y_in

    def __len__(self):
        return len(self.image_indices)


    
if __name__ == "__main__":
    dir = "/home/igeor/MSC-THESIS/data/h5"
    dset = Patch2PixelDataset(dir, image_names=['gogo', 'saintjohn', 'dionisios', 'fanourios', 'odigitria', 'minos'], band_range=(80, 2128), apply_scaling=True)
    x, y = dset.__getitem__(0)
    print(x.shape, y.shape, len(dset))