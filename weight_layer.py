import conv
import BN
import torch
import torch.nn as nn


class weight_layer(nn.Module):
    def __init__(self,in_channel, out_channel,stride=1):
        super().__init__()
        self.conv =nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return nn.functional.relu(self.bn(self.conv(x)))