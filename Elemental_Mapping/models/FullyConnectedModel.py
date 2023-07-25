import torch 
from torch import nn 

# Fully connected network
class FullyConnectedModel(nn.Module):
    def __init__(self, in_features=4096, out_features=12, hidden_dims=[512, 64], prior_layer=None, dropout=0.1):
        super(FullyConnectedModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dims = hidden_dims

        self.layers = nn.ModuleList()

        if prior_layer is not None:
            self.prior_layer = prior_layer
            self.layers += [self.prior_layer]
            self.in_features = self.prior_layer.out_features

        self.layers += [nn.Linear(self.in_features, self.hidden_dims[0])]
        for i in range(len(self.hidden_dims) - 1):
            self.layers += [nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1])]
        self.layers += [nn.Linear(self.hidden_dims[-1], self.out_features)]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        self.alias = f'FCN_{self.in_features}_{self.hidden_dims}_{self.out_features}_dr{dropout}_{prior_layer}'
        
    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
            if self.state == 'train':
                x = self.dropout(x)
        return x

    def set_state(self, state):
        if state in ['train', 'eval', 'test']:
            self.state = state
    
    def train(self, dataloader, optimizer, criterion, epochs=1, device='cuda'):
        self.set_state('train')
        train_loss = 0.0
        for epoch in range(epochs):
            for x, y in dataloader:
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
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                y_hat = self(x)
            y_hats += [y_hat]

            # Secure that y_hat and y have the same shape
            # otherwise squeeze() the tensor with the most dimensions
            if y_hat.shape != y.shape:
                if len(y_hat.shape) > len(y.shape): y_hat = y_hat.squeeze()
                else: y = y.squeeze()
            eval_loss += criterion(y_hat, y) / (len(dataloader) * x.shape[0])
        
        y_hats = torch.cat(y_hats, dim=0)
        return eval_loss.item(), y_hats
    