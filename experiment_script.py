device = 'cuda'

# %%
import torch 
import matplotlib.pyplot as plt

# %% [markdown]
# **PROBLEM**: An element map is an image showing the spatial distribution of elements in a sample. Because it is acquired from a polished section, it is a 2D section through the unknown sample. Element maps are extremely useful for displaying element distributions in textural context, particularly for showing compositional zonation.

# %%
from Elemental_Mapping.datasets.Pixel2PixelDataset import Pixel2PixelDataset

# %%
training_images = ['gogo', 'dionisios', 'fanourios', 'odigitria', 'minos']
test_images = ['saintjohn']

# %%
band_range = (80, 2128)

dataset = Pixel2PixelDataset(
    '/home/igeor/MSC-THESIS/data/h5',
    image_names=training_images, 
    sample_step = 1, 
    device='cuda', 
    band_range=(80, 2128), 
    target_elems=['S_K','K_K','Ca_K','Cr_K','Mn_K','Fe_K','Cu_K','Zn_K','Sr_K','Au_L','Hg_L','Pb_L'])

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

# %%
# Define the Fully Connected Model
from Elemental_Mapping.models.FullyConnectedModel import FullyConnectedModel

in_features = band_range[1]-band_range[0]
out_features = len(dataset.target_elems)
fcn = FullyConnectedModel(in_features=in_features, out_features=out_features, hidden_dims=[512, 64])
fcn.to(device)

# %%
# Define Criterion and optimizer
from Elemental_Mapping.loss_functions.AdaptiveL1Loss import AdaptiveL1Loss
import torch.optim as optim
    
criterion = AdaptiveL1Loss()
fcn_optimizer = optim.Adam(fcn.parameters(), lr=0.001)

# %%
num_epochs = 100

for epoch in range(num_epochs):
    train_loss = fcn.train(train_loader, fcn_optimizer, criterion, epochs=1, device=device)
    val_loss, pred = fcn.eval(val_loader, criterion, device=device)
    print(f'Epoch: {epoch}, Train Loss: {round(train_loss, 4)}, Val Loss: {round(val_loss, 4)}')
    # Save model 
    torch.save(fcn.state_dict(), f'fcn.pt')

# %%



