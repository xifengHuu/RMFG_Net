import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class QKVBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(QKVBlock, self).__init__()
        self.query_conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.key_conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.value_conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.query_conv(x), self.key_conv(x), self.value_conv(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, x, out_shapes):
        B, C, T, H, W = x.shape
        x = x.view(B * C, T, H, W)  # must be 4-D for bilinear
        x = F.interpolate(x, size=(out_shapes[-2], out_shapes[-1]),
                          mode="bilinear", align_corners=False)
        x = self.conv(x.view(B, C, T, out_shapes[-2], out_shapes[-1])).view(B, self.out_channels, -1)
        return x


class RMFG(nn.Module):
    def __init__(self, x2_channels, x3_channels, x4_channels):
        super(RMFG, self).__init__()
        # print(self.__class__.__name__, x2_channels, x3_channels, x4_channels)
        self.tmp_channels = [16, 24, 32]
        self.X2_QKV = QKVBlock(x2_channels, self.tmp_channels[0])
        self.X3_QKV = QKVBlock(x3_channels, self.tmp_channels[1])
        self.X4_QKV = QKVBlock(x4_channels, self.tmp_channels[2])

        self.X4_Q = UpConvBlock(self.tmp_channels[2], self.tmp_channels[1])
        self.T1_Conv = nn.Conv3d(self.tmp_channels[1], self.tmp_channels[2], 1, bias=False)
        self.Output1_Conv = nn.Conv3d(self.tmp_channels[2] * 2, self.tmp_channels[2], 1, bias=False)

        self.X3_Q = UpConvBlock(self.tmp_channels[1], self.tmp_channels[0])
        self.T3_Conv = nn.Conv3d(self.tmp_channels[0], self.tmp_channels[1], 1, bias=False)
        self.Output2_Conv = nn.Conv3d(self.tmp_channels[1] * 2, self.tmp_channels[1], 1, bias=False)

        self.Output3_Conv = nn.Conv3d(self.tmp_channels[2], x4_channels, 1, bias=False)

    def transformer_impl(self, Q, K, V):
        # ( Q * K ) * V
        B, CQ, MQ = Q.shape  # B C T*H*W
        B, MQ, CQ = K.shape
        B, CQ, MQ = V.shape

        P = torch.bmm(Q, K)
        if isinstance(CQ, torch.Tensor):
            P = P / CQ.sqrt()
        else:
            P = P / math.sqrt(CQ)
        P = torch.softmax(P, dim=1)

        M = torch.bmm(P, V)

        return M, P

    def forward(self, x2, x3, x4):
        x2_ = x2.permute(0, 2, 1, 3, 4).contiguous()  # B C T H W
        x2_query, x2_key, x2_value = self.X2_QKV(x2_)

        x3_ = x3.permute(0, 2, 1, 3, 4).contiguous()  # B C T H W
        x3_query, x3_key, x3_value = self.X3_QKV(x3_)

        x4_ = x4.permute(0, 2, 1, 3, 4).contiguous()  # B C T H W
        x4_query, x4_key, x4_value = self.X4_QKV(x4_)

        x4_query_up = self.X4_Q(x4_query, x3_.shape)
        x3_key_t = x3_key.view(*x3_key.shape[:2], -1).permute(0, 2, 1).contiguous()
        x3_value_t = x3_value.view(*x3_value.shape[:2], -1)
        t1, _ = self.transformer_impl(x4_query_up, x3_key_t, x3_value_t)          # B C T*H*W
        
        x4_query_t = x4_query.view(*x4_query.shape[:2], -1)
        t1 = self.T1_Conv(t1.view(x3_key.shape)).view(*x4_query.shape[:2], -1)      # B C T H W
        x4_key_t = x4_key.view(*x4_key.shape[:2], -1).permute(0, 2, 1).contiguous()
        t2, _ = self.transformer_impl(x4_query_t, x4_key_t, t1)                   # B C T H W
 
        t2 = t2.view(*t2.shape[:2], *x3_key.shape[2:])                       # B C T H W

        out1 = torch.cat([t2, x3_], dim=1)
        out1 = self.Output1_Conv(out1).permute(0, 2, 1, 3, 4).contiguous()   # B T C H W
        
        x3_query_up = self.X3_Q(x3_query, x2_.shape)
        x2_key_t = x2_key.view(*x2_key.shape[:2], -1).permute(0, 2, 1).contiguous()
        x2_value_t = x2_value.view(*x2_value.shape[:2], -1)
        t3, _ = self.transformer_impl(x3_query_up, x2_key_t, x2_value_t)           # B C T*H*W
        
        x3_query_t = x3_query.view(*x3_query.shape[:2], -1)
        t3 = self.T3_Conv(t3.view(*x2_key.shape)).view(*x3_query.shape[:2], -1)      # B C T H W
        x3_key_t = x3_key.view(*x3_key.shape[:2], -1).permute(0, 2, 1).contiguous()
        t4, _ = self.transformer_impl(x3_query_t, x3_key_t, t3)                    # B C T H W

        t4 = t4.view(*t4.shape[:2], *x2_key.shape[2:])                        # B C T H W
        out2 = torch.cat([t4, x2_], dim=1)
        out2 = self.Output2_Conv(out2).permute(0, 2, 1, 3, 4).contiguous()   # B T C H W
        return out2, out1
