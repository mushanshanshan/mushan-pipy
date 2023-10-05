import lightning_module
import torch
import torchaudio
import unittest
from torch.utils.data import Dataset, DataLoader
import tqdm
import gdown
from pathlib import Path
import os

class Dataset(Dataset):
    def __init__(self, wavlist):
        self.wavlist = wavlist
        _, self.sr = torchaudio.load(self.wavlist[0])

    def __len__(self):
        return len(self.wavlist)

    def __getitem__(self, idx):
        fname = self.wavlist[idx]
        wav, _ = torchaudio.load(fname)
        sample = {
            "fname": fname,
            "wav": wav}
        return sample
    
    def collate_fn(self, batch):
        max_len = max([x["wav"].shape[1] for x in batch])
        names = []
        out = []
        # Performing repeat padding
        for t in batch:
            names.append(t["fname"])
            wav = t["wav"]
            amount_to_pad = max_len - wav.shape[1]
            padding_tensor = wav.repeat(1,1+amount_to_pad//wav.size(1))
            out.append(torch.cat((wav,padding_tensor[:,:amount_to_pad]),dim=1))
        return names, torch.stack(out, dim=0)

class Score:
    """Predicting score for each audio clip."""

    def __init__(
        self,
        input_sample_rate: int = 16000,
        device: str = "cpu"):
        """
        Args:
            ckpt_path: path to pretrained checkpoint of UTMOS strong learner.
            input_sample_rate: sampling rate of input audio tensor. The input audio tensor
                is automatically downsampled to 16kHz.
        """
        print(f"Using device: {device}")
        
        self.ckpt_path = str(Path(__file__).resolve().parent.joinpath("g_05000000"))
        if not os.path.exists(self.ckpt_path):
            gdown.download(url="https://drive.google.com/file/d/1XiLi8V4t40gde1oeL0uh7CTIiqupNJHm/view?usp=drive_link", output=self.ckpt_path, quiet=False, fuzzy=True)
        
        self.device = device
        self.model = lightning_module.BaselineLightningModule.load_from_checkpoint(
            self.ckpt_path, map_location=self.device).eval().to(self.device)
        self.in_sr = input_sample_rate
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=input_sample_rate,
            new_freq=16000,
            resampling_method="sinc_interpolation",
            lowpass_filter_width=6,
            dtype=torch.float32,
        ).to(device)
    
    def score(self, wavs: torch.tensor) -> torch.tensor:
        """
        Args:
            wavs: audio waveform to be evaluated. When len(wavs) == 1 or 2,
                the model processes the input as a single audio clip. The model
                performs batch processing when len(wavs) == 3. 
        """
        if len(wavs.shape) == 1:
            out_wavs = wavs.unsqueeze(0).unsqueeze(0)
        elif len(wavs.shape) == 2:
            out_wavs = wavs.unsqueeze(0)
        elif len(wavs.shape) == 3:
            out_wavs = wavs
        else:
            raise ValueError('Dimension of input tensor needs to be <= 3.')
        if self.in_sr != 16000:
            out_wavs = self.resampler(out_wavs)
        bs = out_wavs.shape[0]
        batch = {
            'wav': out_wavs,
            'domains': torch.zeros(bs, dtype=torch.int).to(self.device),
            'judge_id': torch.ones(bs, dtype=torch.int).to(self.device)*288
        }
        with torch.no_grad():
            output = self.model(batch)
        
        return output.mean(dim=1).squeeze(1).cpu().detach().numpy()*2 + 3
    
    def pred(self, wav_list, batch_size = 10):

        dataset = Dataset(wav_list)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            num_workers=4)
        
        set_res = {}
        for names, batch in tqdm.tqdm(loader):
            scores = self.score(batch.to(self.device))
            ds = {names[i] : scores[i] for i in range(len(names))}
            set_res.update(ds)
            
        return set_res