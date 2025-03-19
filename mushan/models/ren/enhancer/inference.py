import logging
from functools import cache
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from ..inference import inference
from .download import download
from .enhancer import Enhancer
from .hparams import HParams

hp = HParams(
    fg_dir=Path('data/fg'),
    bg_dir=Path('data/bg'),
    rir_dir=Path('data/rir'),
    load_fg_only=False,
    praat_augment_prob=0.2,
    wav_rate=44100,
    n_fft=2048,
    win_size=2048,
    hop_size=420,
    num_mels=128,
    stft_magnitude_min=0.0001,
    preemphasis=0.97,
    mix_alpha_range=[0.2, 0.8],
    nj=64,
    training_seconds=3.0,
    batch_size_per_gpu=32,
    min_lr=1e-5,
    max_lr=1e-4,
    warmup_steps=1000,
    max_steps=1_000_000,
    gradient_clipping=1.0,
    cfm_solver_method='midpoint',
    cfm_solver_nfe=64,
    cfm_time_mapping_divisor=4,
    univnet_nc=96,
    lcfm_latent_dim=64,
    lcfm_training_mode='cfm',
    lcfm_z_scale=6.0,
    vocoder_extra_dim=32,
    gan_training_start_step=None,
    enhancer_stage1_run_dir=None,
    denoiser_run_dir=None
)


class EnhWrapper:
    def __init__(self, device):
        self.device = device
        self.hp = hp
        self.net = Enhancer(self.hp)

    def _config(self, state_dict, nfe=32, solver="midpoint", lambd=0.5, tau=0.5):
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.net.to(self.device)
        self.net.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
        
    
    @torch.inference_mode()
    def infer_path_wo_resample(self, path):
        dwav, ori_sr = torchaudio.load(path)
        dwav = dwav.to(self.device)
        dwav = dwav.mean(dim=0).to(self.device)
        wav, new_sr = inference(model=self.net, dwav=dwav, sr=ori_sr, device=self.device)
    
        return wav, new_sr
    
    @torch.inference_mode()
    def infer_wave(self, dwav, ori_sr):
        dwav = dwav.mean(dim=0).to(self.device)
        wav, new_sr = inference(model=self.net, dwav=dwav, sr=ori_sr, device=self.device)
        resampler = T.Resample(new_sr, ori_sr, dtype=wav.dtype)
        wav = resampler(wav.cpu())
        
        return wav

    @torch.inference_mode()
    def infer_path(self, path):
        dwav, ori_sr = torchaudio.load(path)
        dwav = dwav.mean(dim=0).to(self.device)
        wav, new_sr = inference(model=self.net, dwav=dwav, sr=ori_sr, device=self.device)
        resampler = T.Resample(new_sr, ori_sr, dtype=wav.dtype)
        wav = resampler(wav.cpu())
        
        return wav