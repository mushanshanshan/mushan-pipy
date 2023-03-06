import torch
from vector_quantize_pytorch import VectorQuantize, ResidualVQ, GroupedResidualVQ
from torch import nn
from einops import rearrange

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

    @torch.no_grad()
    def from_idx(self, x, key=None):
        q, x = self.vqs[key].from_idx(x)
        return q, x

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
        
    @torch.no_grad()
    def from_idx(self, k):
        q = self.vq.get_output_from_indices(k)
        q = self.post(q)
        
        q = rearrange(q, 'b l d -> b d l')
        x = self.conv_decoder(q)
        return q, x