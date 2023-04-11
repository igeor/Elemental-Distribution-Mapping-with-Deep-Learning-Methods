import torch
import torch.nn as nn 

class Conv1DModel(nn.Module):
    def __init__(self, in_features=2048, hidden_dims=[64, 64], out_features=12):
        super(Conv1DModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dims = hidden_dims
        self.conv1 = nn.Conv1d(1, 64, 5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(64, 64, 3, stride=2, padding=2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.maxpool = nn.MaxPool1d(2)
        self.fcn = nn.Sequential( nn.Linear(8192, 512), nn.ReLU(), nn.Linear(512, self.out_features), nn.ReLU() )
        
    def set_state(self, state):
        if state in ['train', 'eval', 'test']:
            self.state = state

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        if self.state == 'train': x = self.drop(x)
        x = self.maxpool(self.relu(self.conv2(x)))
        if self.state == 'train': x = self.drop(x)
        x = torch.flatten(x, start_dim=1) 
        return self.fcn(x)

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
    model = Conv1DModel(out_features=1, device='cuda')
    print(model(x).shape)
    print(sum(p.numel() for p in model.parameters()))