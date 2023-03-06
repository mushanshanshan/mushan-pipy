from glob import glob
import librosa   
from tqdm import tqdm
import os
import random
import numpy as np
import torch
import math
from librosa.util import normalize
import torch.utils.data
import torchaudio
from collections import defaultdict
from loguru import logger
from mushan.text.eng.front_end import Frontend as ENFrontend
from mushan.text.chs.front_end import Frontend as CNFrontend
from mushan.models.bigv.utils import bigv_mel

    

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

    
class SimpleAudioSpecLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, config, optinal = {}):
        self.audiopaths_sid_text = []
        self.rank = config.dist.rank
        
        self.optinal = optinal
        for i in config.train.train_filelists:
            temp = load_filepaths_and_text(i)
            self.audiopaths_sid_text += temp
            if self.rank == '0':
                logger.info(f"Added train file : {i}, length : {len(temp)}")
    
        self.text_cleaners = config.data.text_cleaners
        self.max_wav_value = config.data.max_wav_value
        self.sampling_rate = config.data.sampling_rate
        self.filter_length = config.data.filter_length
        self.hop_length = config.data.hop_length
        self.win_length = config.data.win_length
        self.sampling_rate = config.data.sampling_rate

        self.cleaned_text = config.data.cleaned_text

        self.add_blank = config.data.add_blank
        self.min_text_len = config.data.min_text_len
        self.max_text_len = config.data.max_text_len
        
        self.min_audio_len = config.data.min_audio_len
        self.max_audio_len = config.data.max_audio_len
        
        if config.data.language == 'ch':
            self.language = 'ch'
            self.frontend = CNFrontend()
        else:
            self.language = 'en'
            self.frontend = ENFrontend()
            
        self.symbols_len = self.frontend.symbols_len
        self.loop_idx = []

        

        self._filter()
        
        random.seed(1234)
        
        
    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        audio_lengths = []
        text_lengths = []
        missing_file = []
        idx_loop_new = []

        logger.info(f"Rank {self.rank} start processing filelists...")
        for i in tqdm(range(len(self.audiopaths_sid_text)), disable=self.rank != 0):
            try:
                audiopath, spk, dur, ori_text, pho = self.audiopaths_sid_text[i]
                dur = float(dur)
                if dur > self.min_audio_len and dur < self.max_audio_len:
                    audiopaths_sid_text_new.append(audiopath)
                    for time in range(max(1, math.floor(dur))):
                        idx_loop_new.append(len(audiopaths_sid_text_new)-1)
                    
            except Exception as e:
                print(e)
                # exit(1)
        

        if self.rank == 0 and len(missing_file) > 0:
            logger.error(f"Missing data index: {missing_file}")
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.loop_idx = idx_loop_new
        random.shuffle(self.loop_idx)
        if self.rank == 0:
            logger.info(f"Avaliable data length: {len(self.loop_idx)}")
            
    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath = audiopath_sid_text

        # wav = self.get_audio(audiopath)
        # if "clip_wave" in self.optinal.keys():
        #     wav, _ = self.clip_spec(wav, self.optinal['clip_wave'])
        # spec = bigv_mel(wav)
        
        wav = torch.zeros(1, 5)
        spec = self.get_mel_spec(audiopath)
        spec, _ = self.clip_spec(spec, self.optinal['clip_spec'])
            
        return wav, spec
    
    def get_audio(self, filename, length = 16384):
        audio, sampling_rate = torchaudio.load(filename)
        audio = audio.numpy().squeeze(0)
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio)
        start_idx = random.randint(0, audio.shape[0] - length - 1)
        audio = audio[start_idx: start_idx + length].clone()
        audio = audio.unsqueeze(0)
        return audio
    

    
    def get_mel_spec(self, filename):
        spec_filename = filename.replace("/wave/", "/feature/mel_spec/").replace(".flac", f".mel")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename, map_location='cpu')
            assert spec.shape[0] == 100
        else:
            raise FileNotFoundError
        return spec

    def clip_spec(self, spec, length):
        if spec.shape[1] < length:
            p = torch.zeros(spec.shape[0], length, dtype = spec.dtype, device = spec.device)
            p[:, :spec.shape[1]] = spec
            return p, False
        else:
            start_idx = random.randint(0, spec.shape[1] - length - 1)
            spec = spec[:, start_idx: start_idx + length].clone()
            return spec, True

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[self.loop_idx[index]])

    def __len__(self):
        return len(self.loop_idx)
    

class SimpleAudioSpecCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, config=None, optional = {}, return_ids=False, tacotron=False):
        
        self.return_ids = return_ids
        self.optional = optional


    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[0].size(1) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))



        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        spec_padded.zero_()
        wav_padded.zero_()

        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[0]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            

        return wav_padded, wav_lengths, spec_padded, spec_lengths
