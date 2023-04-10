import torch 
import torch.nn as nn 

class CombinedLoss(torch.nn.Module):
    def __init__(self, criterion1, criterion2):
        super(CombinedLoss, self).__init__()
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def forward(self, predict, target):
        criterion1Loss = self.criterion1(predict, target) 
        criterion2Loss = self.criterion2(predict, target)
        return criterion1Loss + criterion2Loss