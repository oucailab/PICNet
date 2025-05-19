import time
import pywt
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable, gradcheck

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, *filters):
        x = x.contiguous()
        ctx.save_for_backward(*filters)
        ctx.shape = x.shape
        dim = x.shape[0]
        # print(" x.shape : ", x.shape)
        # print("dim : " , dim)
        outputs = []
        for filt in filters:
            # Expand filter to match input channels and use 3D convolution
            conv_out = F.conv3d(x, filt.expand(dim, -1, -1, -1, -1).to(torch.float), stride=2, groups=dim)
            outputs.append(conv_out)
        x = torch.cat(outputs, dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            # print("ctx.shape : ", ctx.shape)
            B, D, H, W = ctx.shape
            C = 1
    
            # 确保 dx 的形状正确
            # print("dx.shape : ", dx.shape)
            dx = dx.view(B, 8, -1, D//2, H//2, W//2)
            dx = dx.transpose(1, 2).reshape(B, -1, D//2, H//2, W//2).to(torch.float)

            # print("dx.shape: ", dx.shape)

            dx = dx.repeat(1, 16, 1, 1, 1)  # 从 8 通道扩展到 128 通道
            # print("dx.shape: ", dx.shape)
    
            # 定义新的滤波器形状
            # 确保 filters 是一个 tensor
            if isinstance(filters, tuple):
                filters = torch.cat(filters, dim=0)
            elif isinstance(filters, list):
                filters = torch.stack(filters, dim=0)
            filters = filters.repeat(16, 1, 1, 2, 1).to(torch.float)

            # print("filters.shape: ", filters.shape)
            # 执行转置卷积
            dx = F.conv_transpose3d(dx, filters, stride=(1, 2, 2), groups=32)

            # 在时间维度上进行平均池化以降低时间维度
            dx = F.avg_pool3d(dx, kernel_size=(16, 1, 1), stride=(16, 1, 1))
    
            # 确保 dx 的形状与前向传播的输入张量 x 的形状一致
            # print("dx.shape : ", dx.shape)
            dx = dx.view(B, D, H, W)
    
        return (dx,) + (None,) * len(filters)

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        x = x.unsqueeze(1)
        # print("x.shape : ", x.shape)
        B, C, D, H, W = x.shape  # Assuming x.shape is [64, 1, 512, 3, 3]
        # print("x.shape : ", x.shape)
        # Adjust the reshaping to maintain correct dimensions
        x = x.view(B, 8, -1, H, W)  # No need to transpose if dimensions are correct
        # Ensure filters are in the correct shape
        filters = filters.to(torch.float)
        # print("x.shape : ", x.shape)
        # print("filters : ", filters.shape)
        # Set groups=1 to match input channels
        x = F.conv_transpose3d(x, filters, stride=2, groups=1)
        return x
        
    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            # print("dx.shape", dx.shape)
            # Adjust dx shape to match the expected input for conv3d
            # dx = dx.unsqueeze(2)  # Now dx.shape is [64, 512, 1, 3, 3]
            
            # Perform the backward convolution
            dx = F.conv3d(dx, filters, stride=2, groups=1)
            dx = dx.view(B, C, H, W)
            
        return dx, None

class IDWT_3D(nn.Module):
    def __init__(self, wave):
        super(IDWT_3D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        N = rec_lo.numel()  # Length of the filter
        filters = []
        
        # Define the combinations for LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
        combinations = [
            ('L', 'L', 'L'),
            ('L', 'L', 'H'),
            ('L', 'H', 'L'),
            ('L', 'H', 'H'),
            ('H', 'L', 'L'),
            ('H', 'L', 'H'),
            ('H', 'H', 'L'),
            ('H', 'H', 'H')
        ]
        
        for combo in combinations:
            lo = rec_lo if combo[0] == 'L' else rec_hi
            hi = rec_lo if combo[1] == 'L' else rec_hi
            dep = rec_lo if combo[2] == 'L' else rec_hi
            
            # Reshape filters for outer product
            lo = lo.view(1, 1, N, 1, 1)
            hi = hi.view(1, 1, 1, N, 1)
            dep = dep.view(1, 1, 1, 1, N)
            
            filt = lo * hi * dep  # Outer product to get 3D filter
            filters.append(filt)
        
        # Concatenate all filters along the first dimension
        self.register_buffer('filters', torch.cat(filters, dim=0).to(torch.float))
    
    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_3D(nn.Module):
    def __init__(self, wave):
        super(DWT_3D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        
        # Reshape the 1D filters to 3D tensors
        lo_x = dec_lo.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, N)
        lo_y = dec_lo.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, N)
        lo_z = dec_lo.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, N)
        
        hi_x = dec_hi.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, N)
        hi_y = dec_hi.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, N)
        hi_z = dec_hi.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, N)
        
        # Generate 3D filters
        w_lll = lo_x * lo_y * lo_z                # Shape: (1, 1, N)
        w_llh = lo_x * lo_y * hi_z                # Shape: (1, 1, N)
        w_lhl = lo_x * hi_y * lo_z                # Shape: (1, 1, N)
        w_lhh = lo_x * hi_y * hi_z                # Shape: (1, 1, N)
        w_hll = hi_x * lo_y * lo_z                # Shape: (1, 1, N)
        w_hlh = hi_x * lo_y * hi_z                # Shape: (1, 1, N)
        w_hhl = hi_x * hi_y * lo_z                # Shape: (1, 1, N)
        w_hhh = hi_x * hi_y * hi_z                # Shape: (1, 1, N)
        
        # Expand dimensions to match 3D convolution requirements
        self.register_buffer('w_lll', w_lll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_llh', w_llh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lhl', w_lhl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lhh', w_lhh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hll', w_hll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hlh', w_hlh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hhl', w_hhl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hhh', w_hhh.unsqueeze(0).unsqueeze(0))
    
    def forward(self, x):
        return DWT_Function.apply(x, 
                                  self.w_lll, self.w_llh, 
                                  self.w_lhl, self.w_lhh, 
                                  self.w_hll, self.w_hlh, 
                                  self.w_hhl, self.w_hhh)