import torch
import torch.nn as nn
from residualBlock import residualBlock


class ResNet(nn.Module):
    def __init__(self, n=3, num_classes = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.ModuleList([residualBlock(16,16) for _ in range(n)])
        self.layer2= nn.ModuleList([residualBlock(in_ch = 16 , out_ch=32,stride=2)]+ [residualBlock(32,32) for _ in range(n-1)])
        self.layer3= nn.ModuleList([residualBlock(in_ch = 32, out_ch=64,stride=2)]+ [residualBlock(64,64) for _ in range(n-1)])
        self.fc = torch.nn.Linear(64,num_classes)

    def forward(self,x):
        out = x
        out = torch.relu(self.bn1(self.conv1(x)))
        for block in self.layer1:
            out = block.forward(out)
        for block in self.layer2:
            out = block.forward(out)
        for block in self.layer3:
            out = block.forward(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out,(1,1) )
        
        out =out.reshape(out.shape[0],-1)

        out = self.fc(out)
        return out