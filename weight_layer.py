import conv
import BN
import torch

class weight_layer:
    def __init__(self,in_channel, out_channel,stride=1):
        self.conv =conv.Conv2d(in_channel,out_channel,stride=stride)
        self.bn = BN.BN()

    def forward(self, x):
        output =self.conv.forward(x,padding=1)
        output = self.bn.norm(output)
        output = torch.relu(output)
        return output