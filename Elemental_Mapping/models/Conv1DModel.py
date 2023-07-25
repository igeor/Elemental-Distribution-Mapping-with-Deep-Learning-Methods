import torch
import torch.nn as nn 

class Conv1DModel(nn.Module):
    def __init__(self, in_features=2048, hidden_dims=[64, 64], out_features=12, 
                prior_layer=None, iis=True, flatten_dims=8192, dropout=0.0):
        """
        Convolutional 1D model for elemental mapping
        Args:
            in_features: number of input features
            hidden_dims: list of hidden dimensions
            out_features: number of output features
            prior_layer: prior layer to concatenate to the input
            iis: (include input spectrum) if True, 
                concatenate the input spectrum to the output of the prior layer
            flatten_dims: number of dimensions to flatten the output of the convolutional layers
            dropout: dropout rate
        """
        super(Conv1DModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.iis = iis
        self.hidden_dims = hidden_dims
        self.flatten_dims = flatten_dims
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool1d(2)
        self.state = 'test'

        self.in_channels = 1
        self.prior_layer = prior_layer
        if self.prior_layer is not None:
            self.in_channels += self.prior_layer.w.shape[0] 

        # Initialize a Modulelist with len(hidden_dims) layers
        self.conv_layers = nn.ModuleList([nn.Conv1d(self.in_channels, self.hidden_dims[0], 5, stride=2, padding=2)])
        # Add the rest of the layers
        for i in range(len(self.hidden_dims)-1):
            self.conv_layers.append(nn.Conv1d(self.hidden_dims[i], self.hidden_dims[i+1], 3, stride=2, padding=2))

        self.l1 = nn.Linear(self.flatten_dims, self.out_features)
        # self.l2 = nn.Linear(self.flatten_dims//2, self.out_features)
        
        self.alias = f'CNN1D_{self.in_channels}_{self.in_features}_{self.hidden_dims}_{self.out_features}_dr{dropout}_{prior_layer}'

    def set_state(self, state):
        if self.state in ['train', 'eval', 'test']:
            self.state = state

    def forward(self, x):
        x_in = x
        if self.prior_layer is not None:
            x = self.prior_layer(x)
            if self.iis:
                x = torch.cat([x, x_in], dim=1)
        # Forward pass through the convolutional layers
        for _, conv in enumerate(self.conv_layers):
            x = self.maxpool(self.relu(conv(x)))

        x = torch.flatten(x, start_dim=1)
        if self.state == 'train':
            x = self.drop(self.relu(self.l1(x)))
        else:
            x = self.relu(self.l1(x))

        return x

    def train(self, dataloader, optimizer, criterion, epochs=1000, 
              device='cuda', experiment_alias='default'):
        
        self.set_state('train')
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
        self.set_state('eval')
        eval_loss = 0.0
        y_hats = []
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                y_hat = self(x)
            y_hats += [y_hat]
            eval_loss += criterion(y_hat, y) / (len(dataloader) * x.shape[0])
        
        y_hats = torch.cat(y_hats, dim=0)
        return eval_loss.item(), y_hats

if __name__ == "__main__":
    device = 'cuda'
    x = torch.rand(32, 1, 2048).to(device)
    model = Conv1DModel(hidden_dims=[64,64,64,128], out_features=1).to(device)
    
    with torch.no_grad():
        y = model(x)
    