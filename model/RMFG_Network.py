import os
from re import S
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from model.LightRFB import LightRFB
from model.Res2Net import res2net50_v1b_26w_4s
from model.RMFG_Module import RMFG


class BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)  # Conv + ReLU
        residual = self.conv2(residual) # Conv
        out = x + residual
        return out


class GFNetModule(nn.Module):
    def __init__(self, x2_channels, x3_channels, x4_channels, out_channels):
        super(GFNetModule, self).__init__()
        # print(self.__class__.__name__, x2_channels, x3_channels, x4_channels, out_channels)
        self.conv1 = nn.Conv2d(x2_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(x3_channels, out_channels, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(x4_channels, out_channels, kernel_size=1, stride=1)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.refine = nn.Sequential(
                                        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.PReLU()
                                   )
    
    def forward(self, x1, x2, x3):
        x1n = self.conv1(x1)
        g1 = self.conv2(x1n)
        g1 = torch.sigmoid(g1)

        x2n = self.conv3(x2)
        g2 = self.conv4(x2n)
        g2 = torch.sigmoid(g2)

        x3n = self.conv5(x3)
        g3 = self.conv6(x3n)
        g3 = torch.sigmoid(g3)

        x1gff = (1+g1)*x1n + (1-g1)*(g2*x2n + g3*x3n)
        x2gff = (1+g2)*x2n + (1-g2)*(g1*x1n + g3*x3n)
        x3gff = (1+g3)*x3n + (1-g3)*(g2*x2n + g1*x1n)

        out = self.refine(x1gff + x2gff + x3gff)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class RMFGNet(nn.Module):
    def __init__(self):
        super(RMFGNet, self).__init__()
        self.tmp_channels = [24, 32, 40]
        self.backbone = res2net50_v1b_26w_4s(pretrained=True)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.X2_RFB = LightRFB(channels_in=512, channels_mid=128, channels_out=self.tmp_channels[0])
        self.X3_RFB = LightRFB(channels_in=1024, channels_mid=128, channels_out=self.tmp_channels[1])
        self.X4_RFB = LightRFB(channels_in=2048, channels_mid=256, channels_out=self.tmp_channels[2])

        self.X2_BR = BoundaryRefineModule(self.tmp_channels[0])
        self.X3_BR = BoundaryRefineModule(self.tmp_channels[1])
        self.X4_BR = BoundaryRefineModule(self.tmp_channels[2])

        self.ca2 = ChannelAttention(self.tmp_channels[0])
        self.ca3 = ChannelAttention(self.tmp_channels[1])
        self.ca4 = ChannelAttention(self.tmp_channels[2])
        self.sa = SpatialAttention()
        
        self.RMFG = RMFG(*self.tmp_channels)

        self.gate_fuse = GFNetModule(*self.tmp_channels, 16)
        self.OutNet = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(16, 1, kernel_size=1, bias=False))

    def load_backbone(self, pretrained_dict, logger):
        model_dict = self.state_dict()
        logger.info("load_state_dict!!!")
        for k, v in pretrained_dict.items():
            if (k in model_dict):
                logger.info("load:%s" % k)
            else:
                logger.info("jump over:%s" % k)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def pretrain(self, x):
        origin_shape = x.shape
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2_feature = self.backbone.layer2(x1)
        x3_feature = self.backbone.layer3(x2_feature)
        x4_feature = self.backbone.layer4(x3_feature)

        x2_feature = self.X2_RFB(x2_feature)
        x3_feature = self.X3_RFB(x3_feature)
        x4_feature = self.X4_RFB(x4_feature)

        x2_feature = self.X2_BR(x2_feature)
        x3_feature = self.X3_BR(x3_feature)
        x4_feature = self.X4_BR(x4_feature)
        
        x3_feature = F.interpolate(x3_feature, size=(x2_feature.shape[-2], x2_feature.shape[-1]),
                                        mode="bilinear",
                                        align_corners=False)
        x4_feature = F.interpolate(x4_feature, size=(x2_feature.shape[-2], x2_feature.shape[-1]),
                                        mode="bilinear",
                                        align_corners=False)

        out = self.gate_fuse(x2_feature.clone(), x3_feature.clone(), x4_feature.clone())
        out = F.interpolate(self.OutNet(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                            align_corners=False)

        out = torch.sigmoid(out)
        return out

    def finetune(self, x):
        origin_shape = x.shape
        x = x.view(-1, *origin_shape[2:])                     # (B*T, C, H, W) 
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.backbone.maxpool(x)
        x1 = self.backbone.layer1(x)
        x2_feature = self.backbone.layer2(x1)          # (12, 512, 64, 52)
        x3_feature = self.backbone.layer3(x2_feature)  # (12, 1024, 32, 26)
        x4_feature = self.backbone.layer4(x3_feature)  # (12, 2048, 16, 13)

        x2_feature = self.X2_RFB(x2_feature)                    # (12, 24, 64, 52)
        x3_feature = self.X3_RFB(x3_feature)                    # (12, 32, 32, 26)
        x4_feature = self.X4_RFB(x4_feature)                    # (12, 40, 16, 13)

        x2_feature = self.X2_BR(x2_feature)
        x3_feature = self.X3_BR(x3_feature)
        x4_feature = self.X4_BR(x4_feature)

        x2_feature = self.ca2(x2_feature) * x2_feature
        x2_feature = self.sa(x2_feature) * x2_feature
        x3_feature = self.ca3(x3_feature) * x3_feature
        x3_feature = self.sa(x3_feature) * x3_feature
        x4_feature = self.ca4(x4_feature) * x4_feature
        x4_feature = self.sa(x4_feature) * x4_feature

        x2_feature = x2_feature.view(*origin_shape[:2], *x2_feature.shape[1:])  # (4, 3, 24, 64, 52)
        x3_feature = x3_feature.view(*origin_shape[:2], *x3_feature.shape[1:])  # (4, 3, 32, 32, 26)
        x4_feature = x4_feature.view(*origin_shape[:2], *x4_feature.shape[1:])  # (4, 3, 40, 16, 13)
    
        x2_feature, x3_feature = self.RMFG(x2_feature, x3_feature, x4_feature)

        x2_feature = x2_feature.view(-1, *x2_feature.shape[2:]) # (12, 24, 64, 52)
        x3_feature = x3_feature.view(-1, *x3_feature.shape[2:]) # (12, 32, 32, 26)
        x4_feature = x4_feature.view(-1, *x4_feature.shape[2:]) # (12, 40, 16, 13)

        x3_feature = F.interpolate(x3_feature, size=(x2_feature.shape[-2], x2_feature.shape[-1]),
                                        mode="bilinear",
                                        align_corners=False)
        x4_feature = F.interpolate(x4_feature, size=(x2_feature.shape[-2], x2_feature.shape[-1]),
                                        mode="bilinear",
                                        align_corners=False)

        out = self.gate_fuse(x2_feature.clone(), x3_feature.clone(), x4_feature.clone())
        out = F.interpolate(self.OutNet(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                            align_corners=False)
        out = torch.sigmoid(out)
        return out

    def forward(self, x):
        out = []
        if len(x.shape) == 4:  # (B, C, H, W)
            out = self.pretrain(x)
        elif len(x.shape) == 5:  # (B, T, C, H, W)
            out = self.finetune(x)
        else:
            print("x shape only support for 4-D in pretrain or 5-D in finetune")
        return out


if __name__ == "__main__":
    a = torch.randn(4, 3, 512, 416).cuda()
    mobile = RMFGNet().cuda()
    print(mobile(a).shape)
