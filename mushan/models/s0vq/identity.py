import torch
from vector_quantize_pytorch import VectorQuantize, ResidualVQ, GroupedResidualVQ
from mushan.io import *
from torch import nn
from einops import rearrange
from mushan.io import *

class GRVQS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vqs = nn.ModuleDict()

        for para in config.train.model_list:
            self.vqs['_'.join([str(i) for i in para])] = GRVQ(config, para=para)

    def forward(self, x):
        res = {}
        for key, vq in self.vqs.items():
            res[key] = vq(x)
        return res

    
    @torch.no_grad()
    def infer(self, x, key=None):
        q, c, _ = self.vqs[key](x)
        return q, c

    def keys(self,):
        return self.vqs.keys()

class GRVQ(nn.Module):
    def __init__(self, config, para):
        super().__init__()
        print(para)
        self.vq = GroupedResidualVQ(
                    dim = para[-1],
                    num_quantizers = para[1],
                    groups = para[0],
                    codebook_dim = para[-1],
                    codebook_size = para[2],     # codebook size
                    use_cosine_sim = False,
                    decay = 0.8,             
                    commitment_weight = 1., )
        self.in_layers = nn.Conv1d(in_channels=config.model.s0.in_dim, out_channels=para[-1], kernel_size=3, stride=1, padding=1)
        self.out_layers = nn.Conv1d(in_channels=para[-1], out_channels=config.model.s0.in_dim, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.in_layers(x)
        x = x.transpose(1,2)
        x, indices, commit_loss = self.vq(x)
        x = x.transpose(1,2)
        x = self.out_layers(x)
        return x, indices, commit_loss

    @torch.no_grad()
    def infer(self, x, key=None):
        x = self.in_layers(x)
        x = x.transpose(1,2)
        x, indices, commit_loss = self.vq(x)
        x = x.transpose(1,2)
        x = self.out_layers(x)
        return x, indices, commit_loss

class RVQS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vqs = nn.ModuleDict()

        for para in config.train.model_list:
            self.vqs['_'.join([str(i) for i in para])] = RVQ(config, para=para)

    def forward(self, x):
        res = {}
        for key, vq in self.vqs.items():
            res[key] = vq(x)
        return res

    
    @torch.no_grad()
    def infer(self, x, key=None):
        q, c, _ = self.vqs[key](x)
        return q, c

    def keys(self,):
        return self.vqs.keys()

class RVQ(nn.Module):
    def __init__(self, config, para):
        super().__init__()
        print(para)
        self.vq = ResidualVQ(
                    dim = para[-1],
                    num_quantizers = para[0],
                    codebook_dim = para[-1],
                    codebook_size = para[1],     # codebook size
                    use_cosine_sim = False,
                    decay = 0.8,             
                    commitment_weight = 1., )
        self.in_layers = nn.Conv1d(in_channels=config.model.s0.in_dim, out_channels=para[-1], kernel_size=3, stride=1, padding=1)
        self.out_layers = nn.Conv1d(in_channels=para[-1], out_channels=config.model.s0.in_dim, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.in_layers(x)
        x = x.transpose(1,2)
        x, indices, commit_loss = self.vq(x)
        x = x.transpose(1,2)
        x = self.out_layers(x)
        return x, indices, commit_loss

    @torch.no_grad()
    def infer(self, x, key=None):
        x = self.in_layers(x)
        x = x.transpose(1,2)
        x, indices, commit_loss = self.vq(x)
        x = x.transpose(1,2)
        x = self.out_layers(x)
        return x, indices, commit_loss


