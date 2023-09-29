from .bigv import BigVGAN as Generator

import gdown
import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write

from pathlib import Path

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


class bigv():
    def __init__(self, device='cpu'):
        self.config_fpath = str(Path(__file__).resolve().parent.joinpath("config.json"))
        self.checkpoint_fpath = str(Path(__file__).resolve().parent.joinpath("g_05000000"))
        self.device = device
        
        if not os.path.exists(self.checkpoint_fpath) or not os.path.exists(self.config_fpath):
            gdown.download(url="https://drive.google.com/file/d/1njxNw1YgQdYRhkCMMAjWeiXPFVcETe-c/view?usp=sharing", output=self.config_fpath, quiet=False, fuzzy=True)
            gdown.download(url="https://drive.google.com/file/d/1OQm6SiZOcKFzg4xQBqIKN23JYntPozUZ/view?usp=sharing", output=self.checkpoint_fpath, quiet=False, fuzzy=True)
        
        with open(self.config_fpath) as f:
            data = f.read()


        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        torch.manual_seed(self.h.seed)
        
        if self.device == 'cuda':
            torch.cuda.manual_seed(self.h.seed)
        
        self.generator = Generator(self.h).to(self.device)

        state_dict_g = load_checkpoint(self.checkpoint_fpath, self.device)
        self.generator.load_state_dict(state_dict_g['generator'])
        self.MAX_WAV_VALUE = 32768.0
        self.generator.eval()
        self.generator.remove_weight_norm()
        
    def infer(self, x):
        with torch.no_grad():
            # load the mel spectrogram in .npy format
            x = torch.FloatTensor(x).to(self.device)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)

            y_g_hat = self.generator(x)

            audio = y_g_hat.squeeze()
            audio = audio * self.MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

        return audio
        

