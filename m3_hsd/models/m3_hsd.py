import torch
import torch.nn as nn
from layers.Mainnet import Network
from layers.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # 1. 归一化层
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        
        # 2. 核心网络 (直接调用 Network)
        self.network = Network(configs)

    def forecast(self, x_enc):
        # x_enc: [Batch, Seq_Len, Channel]
        
        # 1. RevIN 归一化
        x_enc = self.revin_layer(x_enc, 'norm')
        
        # 2. Channel Independence 操作
        # [Batch, Seq_Len, Channel] -> [Batch, Channel, Seq_Len]
        x_enc = x_enc.permute(0, 2, 1)
        
        B = x_enc.shape[0]
        C = x_enc.shape[1]
        L = x_enc.shape[2]
        
        # Flatten Batch & Channel: [B * C, L]
        x_enc = x_enc.reshape(B * C, L)
        
        # 3. 进入 Network (包含分解、处理、融合)
        # 输入: [B*C, L], 输出: [B*C, Pred_Len]
        dec_out = self.network(x_enc)
        
        # 4. 恢复维度
        # [B*C, Pred_Len] -> [B, C, Pred_Len]
        dec_out = dec_out.reshape(B, C, self.pred_len)
        
        # [B, C, Pred_Len] -> [B, Pred_Len, C]
        dec_out = dec_out.permute(0, 2, 1)
        
        # 5. RevIN 反归一化
        dec_out = self.revin_layer(dec_out, 'denorm')
        
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc)
        else:
            raise ValueError('Only forecast implemented')