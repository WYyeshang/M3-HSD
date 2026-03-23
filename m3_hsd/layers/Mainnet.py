import torch
import torch.nn as nn
from layers.layersnet import EMADecomp, MLPStream, CNNStream
import matplotlib.pyplot as plt 
import os

class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.pred_len = configs.pred_len
        
        alpha_low = getattr(configs, 'ema_alpha_low', getattr(configs, 'alpha', 0.1))

        alpha_mid = getattr(configs, 'ema_alpha_mid', getattr(configs, 'beta', 0.5))
        self.decomp = EMADecomp(alpha_low=alpha_low, alpha_mid=alpha_mid)
        
        self.net_low = MLPStream(configs.seq_len, configs.pred_len)
        
        k_mid = getattr(configs, 'mid_kernel_size', 5) 
        
        self.net_mid = CNNStream(
            configs.seq_len, configs.pred_len, 
            configs.patch_len, configs.stride, configs.padding_patch,
            kernel_size=k_mid 
        )
        
        k_high = getattr(configs, 'high_kernel_size', 3)
        
        self.net_high = CNNStream(
            configs.seq_len, configs.pred_len, 
            configs.patch_len, configs.stride, configs.padding_patch,
            kernel_size=k_high 
        )
        
        self.fusion = nn.Linear(configs.pred_len * 3, configs.pred_len)

    def forward(self, x):
        
        x_in = x.unsqueeze(-1) 
        
        x_high, x_mid, x_low = self.decomp(x_in)

        if self.training and torch.rand(1).item() < 0.001: 
            self.debug_plot(x_in, x_low, x_mid, x_high)
        
        x_high = x_high.squeeze(-1)
        x_mid  = x_mid.squeeze(-1)
        x_low  = x_low.squeeze(-1)
        
        pred_low = self.net_low(x_low)
        pred_mid = self.net_mid(x_mid)
        pred_high = self.net_high(x_high)
        
        combined = torch.cat([pred_high, pred_mid, pred_low], dim=1)
        output = self.fusion(combined)
        
        return output
    
    