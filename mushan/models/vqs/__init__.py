import torch
from vector_quantize_pytorch import VectorQuantize, ResidualVQ, GroupedResidualVQ
from mushan.io import *
from torch import nn

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
            self.vqs['_'.join([str(i) for i in para])] = GroupedResidualVQ(
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


class MHVQS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vqs = nn.ModuleDict()
        for para in config.train.model_list:
            self.vqs['_'.join([str(i) for i in para])] = VectorQuantize(
                                                                    dim = config.model.s0.in_dim,
                                                                    codebook_size = para[0],  # specify number of quantizers
                                                                    heads = para[1],
                                                                    codebook_dim = para[2],
                                                                    separate_codebook_per_head = True    # codebook size
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