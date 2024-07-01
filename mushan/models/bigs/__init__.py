from .model import BigVSAN
import json
import os
import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict

def get_model(config, ckp):
    with open(config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    state_dict_g = load_checkpoint(ckp)
    generator = BigVSAN(h)
    generator.load_state_dict(state_dict_g['generator'])
    
    generator.eval()
    generator.remove_weight_norm()
    return generator