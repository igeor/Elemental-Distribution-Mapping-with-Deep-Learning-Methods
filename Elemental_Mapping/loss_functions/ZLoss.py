import torch 
import torch.nn as nn 

class ZLoss(torch.nn.Module):
    def __init__(self):
        super(ZLoss, self).__init__()

    def forward(self, predict, target):
        diff = (target - predict)
        mean = diff.mean()
        std = diff.std()
        z_score = (diff - mean) / std
        return z_score.abs().mean()