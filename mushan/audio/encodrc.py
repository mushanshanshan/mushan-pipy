from encodec import EncodecModel
from encodec.utils import convert_audio
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union
from IPython.display import Audio
import numpy as np
import torch
import torchaudio

class AudioEnc():

    def __init__(self, target_bandwidth = 12.0, device = -1):
        
        assert target_bandwidth in [1.5, 3., 6., 12., 24.]
        
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(target_bandwidth)
        self.remove_encodec_weight_norm(model)

        if device < 0:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{device}")

        self.device = device
        self.codec = model.to(device)
        self.sample_rate = model.sample_rate
        self.frame_shift = 320 / self.sample_rate
        self.channels = model.channels
        
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(wav.to(self.device))

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(frames)
    
    def compute_num_frames(self, duration, frame_shift, sampling_rate: int,) -> int:

        num_samples = round(duration * sampling_rate)
        window_hop = round(frame_shift * sampling_rate)
        num_frames = int((num_samples + window_hop // 2) // window_hop)
        return num_frames
    
    def remove_encodec_weight_norm(self, model):
        from encodec.modules import SConv1d
        from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock
        from torch.nn.utils import remove_weight_norm

        encoder = model.encoder.model
        for key in encoder._modules:
            if isinstance(encoder._modules[key], SEANetResnetBlock):
                remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
                block_modules = encoder._modules[key].block._modules
                for skey in block_modules:
                    if isinstance(block_modules[skey], SConv1d):
                        remove_weight_norm(block_modules[skey].conv.conv)
            elif isinstance(encoder._modules[key], SConv1d):
                remove_weight_norm(encoder._modules[key].conv.conv)

        decoder = model.decoder.model
        for key in decoder._modules:
            if isinstance(decoder._modules[key], SEANetResnetBlock):
                remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
                block_modules = decoder._modules[key].block._modules
                for skey in block_modules:
                    if isinstance(block_modules[skey], SConv1d):
                        remove_weight_norm(block_modules[skey].conv.conv)
            elif isinstance(decoder._modules[key], SConvTranspose1d):
                remove_weight_norm(decoder._modules[key].convtr.convtr)
            elif isinstance(decoder._modules[key], SConv1d):
                remove_weight_norm(decoder._modules[key].conv.conv)
                
    def extract_from_path(self, wav_path) -> np.ndarray:
        wave, sr = torchaudio.load(wav_path)
        return self.extract_from_wave(wave, sr)

    def extract_from_wave(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if sampling_rate != self.sample_rate:
            samples = convert_audio(
                samples,
                sampling_rate,
                self.sample_rate,
                self.channels,
            )
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(0)
        else:
            raise ValueError()

        device = self.device
        encoded_frames = self.encode(samples.detach().to(device))
        codes = encoded_frames[0][0]  # [B, n_q, T]
        if True:
            duration = round(samples.shape[-1] / sampling_rate, ndigits=12)
            expected_num_frames = self.compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            assert abs(codes.shape[-1] - expected_num_frames) <= 1
            codes = codes[..., :expected_num_frames]
        return codes.cpu().squeeze(0).permute(1, 0)