from weight_layer import weight_layer 
from conv import Conv2d
import torch.nn as nn


class residualBlock(nn.Module):
    def __init__(self,in_ch, out_ch, stride=1):
        super().__init__()
        self.wl1 = weight_layer(in_ch,out_ch,stride=stride)
        self.wl2 = weight_layer(out_ch,out_ch)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1,stride=stride)
        else:
            self.shortcut = None 


    def forward(self,x):
        out=self.wl1.forward(x)
        out=self.wl2.forward(out)

        if self.shortcut is not None :
            x = self.shortcut.forward(x)
            
        out = out +x
        return out