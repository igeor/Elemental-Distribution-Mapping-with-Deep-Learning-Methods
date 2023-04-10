import torch 
import torch.nn as nn 

class AdaptiveMSELoss(torch.nn.Module):
    def __init__(self):
        super(AdaptiveMSELoss, self).__init__()

    def forward(self, y_pred, y_real):
        return torch.sum(torch.div(torch.pow(y_pred - y_real, 2), torch.sqrt(y_real) + 1))
