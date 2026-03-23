import torch
from torch import nn

from layers.ema import EMA


class DECOMP(nn.Module):    
    def __init__(self, alpha):
        super(DECOMP, self).__init__()
        self.ma = EMA(alpha)
        

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average