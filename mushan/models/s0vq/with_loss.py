import torch
from vector_quantize_pytorch import VectorQuantize, ResidualVQ, GroupedResidualVQ
from mushan.io import *
from torch import nn
from einops import rearrange

class VQS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vqs = nn.ModuleDict()
        for para in config.train.model_list:
            self.vqs['_'.join([str(i) for i in para])] = VectorQuantize(
                                                        dim = config.model.s0.in_dim,
                                                        codebook_dim = para[0],
                                                        codebook_size = para[1],     # codebook size
                                                        use_cosine_sim = False,
                                                        decay = 0.8,             
                                                        commitment_weight = 1.   # the weight on the commitment loss
                                                                            )

    def forward(self, x, key=None):
        if key == None:
            res = {}
            for key, vq in self.vqs.items():
                res[key] = vq(x)
            return res
        else:
            return self.vqs[key](x)
    
    @torch.no_grad()
    def infer(self, x, key=None):
        q, c, _ = self.vqs[key](x)
        return q, c

    def keys(self,):
        return self.vqs.keys()


class RVQS(nn.Module):
    def __init__(self, config):
        
        super().__init__()
        self.vqs = nn.ModuleDict()
        for para in config.train.model_list:
            self.vqs['_'.join([str(i) for i in para])] = ResidualVQ(
                                                        dim = config.model.s0.in_dim,
                                                        num_quantizers = para[0],
                                                        codebook_size = para[1],     # codebook size
                                                        stochastic_sample_codes = True,
                                                        sample_codebook_temp = 0.1,         # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
                                                        shared_codebook = False
                                                    )

                # Debug: Print registered parameters
        for name, param in self.named_parameters():
            print(f"Parameter name: {name}, size: {param.size()}")

    def forward(self, x, key=None):
        if key == None:
            res = {}
            for key, vq in self.vqs.items():
                res[key] = vq(x)
            return res
        else:
            return self.vqs[key](x)
    
    @torch.no_grad()
    def infer(self, x, key=None):
        q, c, _ = self.vqs[key](x)
        return q, c

    def keys(self,):
        return self.vqs.keys()



class GRVQS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vqs = nn.ModuleDict()
        for para in config.train.model_list:
            print(para)
            self.vqs['_'.join([str(i) for i in para])] = GRVQ(
                dim = config.model.s0.in_dim,
                num_quantizers = para[0],  # specify number of quantizers
                groups = para[2],
                codebook_size = para[1],    # codebook size
            )

    def forward(self, x, key=None):
        if key == None:
            res = {}
            for key, vq in self.vqs.items():
                res[key] = vq(x)
            return res
        else:
            return self.vqs[key](x)
    
    @torch.no_grad()
    def infer(self, x, key=None):
        q, c, _ = self.vqs[key](x)
        return q, c

    def keys(self,):
        return self.vqs.keys()


class GRVQ(nn.Module):
    def __init__(self, dim, num_quantizers, groups, codebook_size):
        super().__init__()
        
        self.vq = GroupedResidualVQ(
                                    dim = dim,
                                    num_quantizers = num_quantizers,  # specify number of quantizers
                                    groups = groups,
                                    codebook_size = codebook_size,    # codebook size
                                )

        self.conv_encoder = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)

        self.pre = nn.Linear(dim, dim)
        self.post = nn.Linear(dim, dim)

        self.conv_decoder = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_encoder(x)
        x = rearrange(x, 'b d l -> b l d')

        x = self.pre(x)
        q, c, _ = self.vq(x)
        q = self.post(q)
        
        q = rearrange(q, 'b l d -> b d l')
        x = self.conv_decoder(q)
        return q, c, _

