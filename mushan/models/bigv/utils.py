from .bigv import BigVGAN as Generator

import gdown
import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
from pathlib import Path
import torchaudio


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

def mel_spectrogram(y, 
                    n_fft = 1024, 
                    num_mels = 100, 
                    sampling_rate = 24000, 
                    hop_size = 256, 
                    win_size = 1024, 
                    fmin = 0, 
                    fmax = 12000, 
                    center=False):
    
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def bigv_mel(audio,
            n_fft = 1024, 
            num_mels = 100, 
            sampling_rate = 24000, 
            hop_size = 256, 
            win_size = 1024, 
            fmin = 0, 
            fmax = 12000, ):
    if isinstance(audio, str):
        audio, sampling_rate = torchaudio.load(audio)
    else:
        sampling_rate = 24000
    
    audio = audio.numpy().squeeze(0)
    audio = normalize(audio) * 0.95
    audio = torch.FloatTensor(audio).cpu()
    audio = audio.unsqueeze(0)
    mel = mel_spectrogram(audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False).squeeze(0).cpu()
    return mel


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
            # load the mel spectrogram in .npy formati
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(self.device)
            else:
                x = x.to(self.device)
                
            if len(x.shape) == 2:
                x = x.unsqueeze(0)

            y_g_hat = self.generator(x)

            audio = y_g_hat.squeeze()
            audio = audio * self.MAX_WAV_VALUE
            audio = audio.cpu().type(torch.short)

        return audio
        

