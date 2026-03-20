import torch

class BN:
    def __init__(self, gamma =1, beta =0):
        self.gamma = gamma
        self.beta = beta

    def norm(self, x):
        mean = torch.mean(x, dim=(0,2,3),keepdim=True)
        var = torch.var(x,dim=(0,2,3), keepdim=True)
        x_norm = (x-mean) /torch.sqrt(var+1e-5)
        
        output_x = self.gamma*x_norm+self.beta
        return output_x