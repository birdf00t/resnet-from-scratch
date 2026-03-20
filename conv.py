import torch
import torch.nn.functional as F

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels=in_channels
        self.out_channels = out_channels
        self.weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        

    def forward(self,x, padding=0):
        x = self.pad2d(x,padding=padding)
        self.height_out = int((x.shape[2] - self.kernel_size)/self.stride + 1)
        self.width_out = int((x.shape[3] - self.kernel_size)/self.stride + 1)
        return self.im2col(x,self.stride)


    def pad2d(self,x,padding):
        x= F.pad(x, (padding, padding, padding, padding))
        return x
    
    def im2col(self, x, stride):
        height_range = torch.arange(0,self.height_out*stride,stride)
        width_range = torch.arange(0,self.width_out*stride,stride)
        h, w = torch.meshgrid(height_range, width_range, indexing='ij')

        ki = torch.arange(self.kernel_size)
        h_idx =h.unsqueeze(-1) +ki

        kj = torch.arange(self.kernel_size)
        w_idx = w.unsqueeze(-1) +kj
        patches =x[:, :, h_idx.unsqueeze(-1), w_idx.unsqueeze(-2)]
        batch_size = patches.shape[0]
        slide_h = patches.shape[2]
        slide_w =patches.shape[3]
        result = patches.reshape(batch_size,slide_h*slide_w,(self.kernel_size**2)*self.in_channels)
        w = self.weight.reshape(self.out_channels, (self.kernel_size**2)*self.in_channels)
        final = result @ w.T
        final = final.permute(0,2,1)
        final = final.reshape(batch_size,self.out_channels, self.height_out, self.width_out)
        return final