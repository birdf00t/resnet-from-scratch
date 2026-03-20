from residualBlock import residualBlock
class ResNet:
    def __init__(self, n=3, num_classes = 10):
        self.layer1 = [residualBlock(in_ch = 3, out_ch=16)] + [residualBlock(16,16) for _ in range(n-1)]
        self.layer2= [residualBlock(in_ch = 16 , out_ch=32,stride=2)]+ [residualBlock(32,32) for _ in range(n-1)]
        self.layer3= [residualBlock(in_ch = 32, out_ch=64,stride=2)]+ [residualBlock(64,64) for _ in range(n-1)]

    def forward(self,x):
        out = x
        for block in self.layer1:
            out = block.forward(out)
        for block in self.layer2:
            out = block.forward(out)
        for block in self.layer3:
            out = block.forward(out)
        return out