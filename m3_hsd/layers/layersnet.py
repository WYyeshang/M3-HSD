import torch
import torch.nn as nn

class EMA(nn.Module):
    def __init__(self, alpha):
        super(EMA, self).__init__()
        self.alpha = alpha

    def forward(self, x):       
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.float32), dims=(0,))        
        weights = torch.pow((1 - self.alpha), powers).to(x.device)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x

class EMADecomp(nn.Module):
    def __init__(self, alpha_low=0.1, alpha_mid=0.5):
        super(EMADecomp, self).__init__()
        self.ema_low = EMA(alpha_low)
        self.ema_mid = EMA(alpha_mid)

    def forward(self, x):        
        x_low = self.ema_low(x)          # Macro
        res = x - x_low
        x_mid = self.ema_mid(res)        # Meso
        x_high = res - x_mid             # Micro
        return x_high, x_mid, x_low

class MLPStream(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(MLPStream, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(seq_len, pred_len * 4),
            nn.AvgPool1d(kernel_size=2), 
            nn.LayerNorm(pred_len * 2),
            nn.Linear(pred_len * 2, pred_len),
            nn.AvgPool1d(kernel_size=2), 
            nn.LayerNorm(pred_len // 2),
            nn.Linear(pred_len // 2, pred_len)
        )

    def forward(self, x):        
        x = x.unsqueeze(1)
        x = self.fc_seq[0](x.squeeze(1)) 
        x = self.fc_seq[1](x.unsqueeze(1)).squeeze(1) 
        x = self.fc_seq[2](x) 
        x = self.fc_seq[3](x) 
        x = self.fc_seq[4](x.unsqueeze(1)).squeeze(1) 
        x = self.fc_seq[5](x) 
        x = self.fc_seq[6](x) 
        return x

class CNNStream(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, kernel_size=3):
        super(CNNStream, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1
        
        self.fc1 = nn.Linear(patch_len, patch_len * patch_len)
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        self.conv1 = nn.Conv1d(
            in_channels=self.patch_num, 
            out_channels=self.patch_num, 
            kernel_size=kernel_size,      
            padding=(kernel_size - 1)//2, 
            groups=self.patch_num
        )
        self.bn2 = nn.BatchNorm1d(self.patch_num)
        self.fc2 = nn.Linear(patch_len * patch_len, patch_len)
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.bn3 = nn.BatchNorm1d(self.patch_num)
        self.flatten = nn.Flatten(start_dim=-2)
        self.head = nn.Sequential(
            nn.Linear(self.patch_num * patch_len, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len)
        )
        self.gelu = nn.GELU()

    def forward(self, x):        
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x.unsqueeze(1)).squeeze(1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)       
        res = x                 
        x = self.fc1(x)
        x = self.gelu(self.bn1(x)) 
        x = self.gelu(self.bn2(self.conv1(x)))        
        x = self.fc2(x)        
        x = x + res                
        x = self.gelu(self.bn3(self.conv2(x)))
        x = self.flatten(x)
        
        return self.head(x)