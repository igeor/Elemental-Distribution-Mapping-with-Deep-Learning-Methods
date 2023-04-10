import torch 
from torch import nn 

# Fully connected network
# Input: 2D tensor of size (batch_size, 4096)
# Output: 2D tensor of size (batch_size, 12) values in [0, inf]
class FullyConnectedModel(nn.Module):
    def __init__(self, in_features=4096, out_features=12, hidden_dims=[512, 64]):
        super(FullyConnectedModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dims = hidden_dims
        
        self.layers = nn.ModuleList()
        self.layers += [nn.Linear(self.in_features, self.hidden_dims[0])]
        for i in range(len(self.hidden_dims) - 1):
            self.layers += [nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1])]
        self.layers += [nn.Linear(self.hidden_dims[-1], self.out_features)]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.relu(x)
            if self.state == 'train':
                x = self.dropout(x)
        return x

    def state(self, state):
        if state in ['train', 'eval', 'test']:
            self.state = state
    
    def train(self, dataloader, optimizer, criterion, epochs=1000, 
              device='cuda', experiment_alias='default'):
        
        self.state = 'train'
        train_loss = 0.0
        for epoch in range(epochs):
            for i, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = self(x)
                loss = criterion(y_hat, y)
                train_loss += loss.item() / (len(dataloader) * x.shape[0])
                loss.backward()
                optimizer.step()
                        
        return train_loss
    
    def eval(self, dataloader, criterion, device='cuda'):
        self.state = 'eval'
        eval_loss = 0.0
        y_hats = []
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            y_hat = self(x)
            y_hats += [y_hat]
            eval_loss += criterion(y_hat, y) / (len(dataloader) * x.shape[0])
        
        y_hats = torch.cat(y_hats, dim=0)
        return eval_loss.item(), y_hats
    



if "__main__" == __name__:
    device = 'cuda'

    import torch 
    import matplotlib.pyplot as plt

    from Elemental_Mapping.datasets.Pixel2PixelDataset import Pixel2PixelDataset

    training_images = ['gogo', 'dionisios', 'fanourios', 'odigitria', 'minos']
    test_images = ['saintjohn']

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

    # Define the Fully Connected Model
    from Elemental_Mapping.models.FullyConnectedModel import FullyConnectedModel

    in_features = band_range[1]-band_range[0]
    out_features = len(dataset.target_elems)
    fcn = FullyConnectedModel(in_features=in_features, out_features=out_features, hidden_dims=[512, 64])
    fcn.to(device)

    # Define Criterion and optimizer
    from Elemental_Mapping.loss_functions.AdaptiveL1Loss import AdaptiveL1Loss
    import torch.optim as optim
        
    criterion = AdaptiveL1Loss()
    fcn_optimizer = optim.Adam(fcn.parameters(), lr=0.001)

    num_epochs = 100

    for epoch in range(num_epochs):
        train_loss = fcn.train(train_loader, fcn_optimizer, criterion, epochs=1, device=device)
        val_loss, pred = fcn.eval(val_loader, criterion, device=device)
        print(f'Epoch: {epoch}, Train Loss: {round(train_loss, 4)}, Val Loss: {round(val_loss, 4)}')
        # Save model 
        torch.save(fcn.state_dict(), f'fcn.pt')

