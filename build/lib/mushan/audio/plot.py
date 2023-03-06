'''
Date: 2022-12-12 21:00:22
LastEditors: mushan wwd137669793@gmail.com
LastEditTime: 2023-01-28 23:50:50
FilePath: /mushan-pip/mushan/audio/plot.py
'''
from ..audio.spec_process import get_feature
import librosa
import numpy as np
from pathlib import Path
import torch

def plot_detail_info_from_file(audio, sr=24000, n_fft=1024, hop_size=256, win_size=1024, n_mel=100, pitc_scale=1, fmin=0, fmax=12000, pitch_mel=True, durations=None, level='f'):
    import matplotlib.pylab as plt
    
    naudio, spec, melspec, F0, eng = get_feature(audio, sr=sr, n_fft=n_fft, hop_size=hop_size, win_size=win_size, n_mel=n_mel, fmin=fmin, fmax=fmax,need_f0=True, need_energy=True, need_mel=True)
    
    fig, ax = plt.subplots(figsize=(20, 4))
    im = ax.imshow(melspec, aspect="auto", origin="lower",
                   interpolation='none')

    freqs = librosa.core.mel_frequencies(fmin=0.0, fmax=8000, n_mels=80)
    freqs = torch.from_numpy(freqs)
    
    if pitch_mel:
        F0_mel = [np.argmin(abs(freqs - p))*pitc_scale for p in F0]
    else:
        F0_mel = F0 * pitc_scale
    
    plt.plot([i for i in range(len(F0_mel))], F0_mel, color='r', linewidth=2.5)

    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    plt.show()
    
    
def plot_mel_from_data(audio, sr=24000, n_fft=1024, hop_size=256, win_size=1024, n_mel=100, fmin=0, fmax=12000):
    import matplotlib.pylab as plt
    

    _, _, melspec, _, _ = get_feature(audio, sr=sr, n_fft=n_fft, hop_size=hop_size, win_size=win_size, n_mel=n_mel, fmin=fmin, fmax=fmax,)
        
    fig, ax = plt.subplots(figsize=(20, 4))
    im = ax.imshow(melspec, aspect="auto", origin="lower",
                   interpolation='none')

    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    plt.show()