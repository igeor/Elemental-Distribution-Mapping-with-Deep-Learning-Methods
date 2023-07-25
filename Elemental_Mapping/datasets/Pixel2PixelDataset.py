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
        self.X = []; self.y = []

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
            spec_image = spec_image[:,self.band_min:self.band_max]
            
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
                    self.X.append(x)
                    y = torch.tensor(target_image[idx].astype(np.float32))
                    self.y.append(y)

    def __getitem__(self, index):
        x_in, y_in = self.X[index], self.y[index]
        return x_in.to(self.device), y_in.to(self.device)

    def __len__(self):
        return len(self.X)


if __name__ == "__main__":
    directory = '/home/igeor/MSC-THESIS/data/h5'
    dataset = Pixel2PixelDataset(directory, sample_step=10, device='cuda', band_range=(80, 2128), 
                                 image_names=['gogo', 'saintjohn', 'dionisios', 'fanourios', 'odigitria', 'minos'])
    
    from torch import nn 
    fcn = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Linear(64, 12)
    ).to('cuda')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(fcn.parameters(), lr=0.01)

    for epoch in range(100):
        epoch_loss = 0.0
        for x, y in dataloader:
            y_hat = fcn(x)
            loss = nn.MSELoss()(y_hat, y)
            epoch_loss += loss.item() / len(dataloader)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch: {epoch} | Loss: {epoch_loss}')
