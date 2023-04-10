import torch 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import h5py
import random 
import glob
import os 
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class Patch2PatchDataset(Dataset):

    def __init__(self, dir_name, patch_size=(5, 5), stride=(3, 3), sample=0.1, device='cuda', band_range=(0, 4096), flip=True,
            image_names=['gogo', 'saintjohn', 'dionisios', 'fanourios', 'odigitria'], return_indices=False): 

        self.dir_name = dir_name
        self.pw, self.ph = patch_size
        self.stride_w, self.stride_h = stride
        self.sample = sample
        self.device = device
        self.band_min, self.band_max = band_range
        self.c = abs(self.band_max - self.band_min)
        self.flip = flip
        self.image_names = image_names
        self.target_elems = ['S_K','K_K','Ca_K','Cr_K','Mn_K','Fe_K','Cu_K','Zn_K','Sr_K','Au_L','Hg_L','Pb_L']
        self.return_indices = return_indices

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
            for x in range(0, w-self.pw+1, self.stride_w):
                for y in range(0, h-self.ph+1, self.stride_h):
                    self.image_indices.append((i, x, y))

    def __getitem__(self, index):
        img_idx, x, y = self.image_indices[index]
        ## Load spectra image and target image
        x_in = self.spec_images[img_idx][:, x:x+self.pw, y:y+self.ph]
        y_in = self.target_images[img_idx][:, x:x+self.pw, y:y+self.ph]
        ## Convert type to float32
        x_in = x_in.astype(np.float32)
        y_in = y_in.astype(np.float32)
        ## Transform data to tensors and Add to device
        x_in = torch.tensor(x_in).to(self.device) 
        y_in = torch.tensor(y_in).to(self.device) 
        if self.flip:
            ## Apply random horizontal flipping
            if random.random() > 0.5:
                x_in = TF.hflip(x_in)
                y_in = TF.hflip(y_in)
            ## Apply random vertical flipping
            if random.random() > 0.5:
                x_in = TF.vflip(x_in)
                y_in = TF.vflip(y_in)
        ## View spectra image as a single representation
        x_in = torch.unsqueeze(x_in, dim=0)
        if self.return_indices:
            return (x, y), (x_in, y_in)
        return x_in, y_in

    def __len__(self):
        return len(self.image_indices)


if __name__ == "__main__":
    dir = "/home/igeor/MSC-THESIS/data/h5"
    dset = Patch2PatchDataset(dir, patch_size=(64, 64), stride=64, band_range=(80, 2128))
    x, y = dset.__getitem__(0)
    x = torch.squeeze(x)
    y = torch.squeeze(y)
    x = torch.sum(x, dim=0)
    y = y[-1]
    print(x.shape, y.shape)
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    out = np.concatenate((x,y), axis=-1)
    plt.imsave('out.png', out)