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
    