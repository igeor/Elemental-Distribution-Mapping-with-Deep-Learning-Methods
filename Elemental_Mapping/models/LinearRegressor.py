import torch 
from torch import nn
import numpy as np 

class LinearRegressor(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer = nn.Linear(self.in_features, self.out_features)
        
    def forward(self, x):
        x = self.layer(x)
        x = x.squeeze(1)
        return x

    def train(self, dataloader, optimizer, criterion, epochs=1000, device='cuda'):
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
    
    def predict(self, inputs):
        inputs = torch.from_numpy(inputs)
        self.cpu()
        with torch.no_grad():
            outputs = self.forward(inputs)
        return outputs.cpu().numpy()
