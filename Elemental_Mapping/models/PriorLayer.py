import torch 
from torch import nn 

class PriorLayer(nn.Module):
    def __init__(self, w, s=None, bias=False, apply_sum=False, requres_grad=False, device='cuda'):
        super(PriorLayer, self).__init__()
        self.device = device
        self.requires_grad = requres_grad
        self.w = nn.Parameter(w.to(self.device), requires_grad=requres_grad)
        self.s = s
        self.bias = bias
        self.apply_sum = apply_sum
        self.out_features = w.shape[0]
        
        if self.bias:
            self.b = nn.Parameter(torch.zeros(self.out_features).to(self.device), requires_grad=True)
        
    def __str__(self):
        return f"prior_sum{self.apply_sum}_bias{self.bias}_act{self.s is not None}_grad{self.requires_grad}"
    
    def forward(self, x):
        if self.apply_sum:
            x = torch.sum(x * self.w.unsqueeze(0), dim=-1)
        else:
            x = x * self.w.unsqueeze(0)
            if self.bias:
                x += self.b.unsqueeze(0)
        if self.s is not None: x = self.s(x)
        return x 

