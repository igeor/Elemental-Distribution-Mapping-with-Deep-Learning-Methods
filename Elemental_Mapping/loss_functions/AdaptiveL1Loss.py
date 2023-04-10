import torch 
import torch.nn as nn 

class AdaptiveL1Loss(torch.nn.Module):
    def __init__(self):
        super(AdaptiveL1Loss, self).__init__()

    def forward(self, y_pred, y_real):
        y_pred = y_pred.view(-1, y_pred.shape[-1])
        y_real = y_real.view(-1, y_pred.shape[-1])
        return torch.sum(torch.div(torch.abs(y_pred - y_real), torch.sqrt(y_real) + 1))
