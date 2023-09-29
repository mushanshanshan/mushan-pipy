from glob import glob
import librosa   
import whisper
import torch


class log_mel_spec_dataset(torch.utils.data.Dataset):

    def __init__(self, wav_list, sr=22050, device='cuda'):
        self.sr = sr
        self.device = device
        self.wav_list = wav_list
        print(len(self.wav_list))

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_path = self.wav_list[idx]
        audio, s = librosa.load(wav_path, sr=self.sr)
        assert s == self.sr
        audio = torch.from_numpy(audio)
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return mel, wav_path