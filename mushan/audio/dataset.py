from glob import glob
import librosa   
from tqdm import tqdm
import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio
from collections import defaultdict
from loguru import logger
from mushan.text.eng.front_end import Frontend as ENFrontend
from mushan.text.chs.front_end import Frontend as CNFrontend


    

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

    
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, config, tag='train'):
        self.audiopaths_sid_text = []
        self.rank = config.dist.rank
        
        if tag == 'train':
            for i in config.train.train_filelists:
                temp = load_filepaths_and_text(i)
                self.audiopaths_sid_text += temp
                if self.rank == '0':
                    logger.info(f"Added train file : {i}, length : {len(temp)}")
        elif tag == 'eval':
            for i in config.train.eval_filelists:
                temp = load_filepaths_and_text(i)
                self.audiopaths_sid_text += temp
                if self.rank == '0':
                    logger.info(f"Added train file : {i}, length : {len(temp)}")
        elif tag == 'tune':
            for i in config.train.tune_filelists:
                temp = load_filepaths_and_text(i)
                self.audiopaths_sid_text += temp
                if self.rank == '0':
                    logger.info(f"Added train file : {i}, length : {len(temp)}")
        
        self.ref_dict = defaultdict(list)

        
        self.need_audio = 'audio' in config.data.data_list
        self.need_spec = 'spec' in config.data.data_list
        self.need_ref = 'ref' in config.data.data_list
        self.need_text = 'text' in config.data.data_list
        self.need_vc = 'vc' in config.data.data_list
        self.need_f0 = 'f0' in config.data.data_list
        self.need_energy = 'energy' in config.data.data_list
        self.need_mel = 'mel' in config.data.data_list
            
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
        
        self.spec_suffix = config.data.spec_suffix
        self.f0_suffix = config.data.f0_suffix
        self.energy_suffix = config.data.energy_suffix
        
        self.quantize_f0 = config.data.quantize_f0
        self.quantize_f0_bond = torch.Tensor(config.bins.f0_log)
        
        self.quantize_energy = config.data.quantize_energy
        self.quantize_energy_bond = torch.Tensor(config.bins.energy)
        
        if config.data.language == 'ch':
            self.language = 'ch'
            self.frontend = CNFrontend()
        else:
            self.language = 'en'
            self.frontend = ENFrontend()
            
        self.symbols_len = self.frontend.symbols_len
        
        # if tag == 'train' and config.train.test:
        #     random.shuffle(self.audiopaths_sid_text)
        #     self.audiopaths_sid_text = random.sample(self.audiopaths_sid_text, 256)
        
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

        logger.info(f"Rank {self.rank} start processing filelists...")
        for i in tqdm(range(len(self.audiopaths_sid_text)), disable=self.rank != 0):
            try:
                audiopath, spk, dur, ori_text, pho = self.audiopaths_sid_text[i]
                dur = float(dur)
                
                if dur > self.min_audio_len and dur < self.max_audio_len:
                    audiopaths_sid_text_new.append([audiopath, spk, dur, ori_text, pho])
                    audio_lengths.append(dur)
                    text_lengths.append(len(pho))
                    self.ref_dict[spk].append(audiopath)
            except Exception as e:
                print(e)
                # exit(1)
        
        if os.environ['PYTORCH_RANK'] == '0' and len(missing_file) > 0:
            logger.error(f"Missing data index: {missing_file}")
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.audio_lengths = audio_lengths
        self.text_lengths = text_lengths
        if self.rank == 0:
            logger.info(f"Avaliable data length: {len(self.audiopaths_sid_text)}")

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, spk,  dur, ori_text, text = audiopath_sid_text
        #print(audiopath)
        
        if self.need_audio:
            wav = self.get_audio(audiopath)
        else:
            wav = torch.zeros(1,5)
            
        if self.need_spec:
            spec = self.get_spec(audiopath)
        else:
            spec = torch.zeros(1,1)

        
        if self.need_text:
            text = self.get_text(text)
        else:
            text = torch.zeros(1)
        
        if self.need_ref:
            ref = self.get_ref_spec(spk)
        else:
            ref = torch.zeros((1,1))
            
        if self.need_f0:
            f0 = self.get_f0(audiopath)
            assert spec.shape[-1] == f0.shape[-1], f"spec.shape={spec.shape}, f0.shape={f0.shape}"
        else:
            f0 = torch.zeros(1)
            
        if self.need_energy:
            energy = self.get_energy(audiopath)
            assert spec.shape[-1] == energy.shape[-1], f"spec.shape={spec.shape}, energy.shape={energy.shape}"
        else:
            energy = torch.zeros(1)
            
        return (text, spec, wav, ref, f0, energy)

    def get_ref_spec(self, spk):
        try:
            ref = random.choice(self.ref_dict[spk])
            ref = self.get_spec(ref)
            if ref.shape[-1] > 500:
                start = random.randint(0, ref.shape[-1] - 500)
                end = start + 500
                ref = ref[:,start:end]
        except Exception as e:
            logger.error(f"Ref Error: {spk}, {e}")
            print((f"Ref Error: {spk}, {e}"))
            ref = torch.zeros(513,500)
        return ref
    
    
    def get_f0(self, filename):
        f0 = torch.load(filename.replace(".wav", f".{self.f0_suffix}"))
        nan = f0.isnan()
        f0[nan] = 0
        if self.quantize_f0:
            f0 = torch.bucketize(f0, self.quantize_f0_bond)
            f0[nan] = -1
            f0 += 1
        else:
            f0 /= 2100
        return f0
    
    def get_energy(self, filename):
        energy = torch.load(filename.replace(".wav", f".{self.energy_suffix}"))
        if self.quantize_energy:
            energy = torch.bucketize(energy, self.quantize_energy_bond)
        return energy
    
    def get_audio(self, filename):
        audio_norm, sampling_rate = torchaudio.load(filename)
        return audio_norm

    def get_spec(self, filename):

        if ".wav" in filename:
            spec_filename = filename.replace(".wav", ".linear")
        else:
            if self.spec_suffix == "linear":
                spec_filename = filename.replace("/wave/", "/feature/linear_spec/").replace(".flac", f".{self.spec_suffix}")
                if os.path.exists(spec_filename):
                    spec = torch.load(spec_filename)
                    assert spec.shape[0] == 513
                else:
                    raise FileNotFoundError
            elif self.spec_suffix == "mel":
                spec_filename = filename.replace("/wave/", "/feature/mel_spec/").replace(".flac", f".{self.spec_suffix}")
                if os.path.exists(spec_filename):
                    spec = torch.load(spec_filename)
                    assert spec.shape[0] == 100
                else:
                    raise FileNotFoundError
            elif self.spec_suffix == "codec":
                spec_filename = filename.replace("/wave/", "/feature/codec_24/").replace(".flac", f".{self.spec_suffix}")
                if os.path.exists(spec_filename):
                    spec = torch.load(spec_filename).squeeze(0)
                    assert spec.shape[0] == 128
                else:
                    raise FileNotFoundError
            else:
                raise NotImplementedError
        return spec

    def get_text(self, text):
        
        if self.language == 'en':
            if self.cleaned_text:
                text_norm = self.frontend.pho_to_idx(text)
            else:
                raise NotImplemented
        elif self.language == 'ch':
            text_norm = self.frontend.pho_to_ids(text, add_blank=True)
            text_norm = text_norm['phone_ids'][0]
            text_norm = torch.LongTensor(text_norm)
        else:
            raise NotImplemented
        
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)
    

class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, config=None,return_ids=False, tacotron=False, div2=False):
        
        self.return_ids = return_ids
        self.div2 = div2
        if config != None:
            self.quantize_f0 = config.data.quantize_f0
            self.quantize_energy = config.data.quantize_energy
        else:
            self.quantize_f0 = True
            self.quantize_energy = True
            
        self.tacotron = tacotron

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

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        max_ref_len = max([x[3].size(1) for x in batch])
        
        if self.div2:
            while max_spec_len % 4 != 0:
                max_spec_len += 1

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        ref_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        ref_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_ref_len)
        if self.quantize_f0:
            f0_padded = torch.LongTensor(len(batch), max_spec_len)
        elif self.tacotron:
            f0_padded = torch.FloatTensor(len(batch), max_spec_len)
        else:
            f0_padded = torch.FloatTensor(len(batch), max_spec_len)
            
        if self.quantize_energy:
            energy_padded = torch.LongTensor(len(batch), max_spec_len)
        else:
            energy_padded = torch.FloatTensor(len(batch), max_spec_len)

        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        ref_padded.zero_()
        f0_padded.zero_()
        energy_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            ref = row[3]
            ref_padded[i, :, :ref.size(1)] = ref
            ref_lengths[i] = ref.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            
            if self.tacotron:
                f0_padded[i, spec.size(1)-1:] = 1
            else:
                f0 = row[4]
                f0_padded[i, :f0.size(0)] = f0
            
            energy = row[5]
            energy_padded[i, :energy.size(0)] = energy


        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ref_padded, ref_lengths, f0_padded, energy_padded


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, gpu_mem_limite=-1, gpu_mem_pred=None, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.audio_lengths = dataset.audio_lengths
        self.text_lengths = dataset.text_lengths
        self.rank = dataset.rank

        self.batch_size = batch_size

        self.min_len, self.max_len = boundaries
        
        
        self.gpu_mem_limite = gpu_mem_limite
        if self.gpu_mem_limite > 0:
            self.pred_mem = [gpu_mem_pred(self.audio_lengths[i], self.text_lengths[i]) for i in range(len(self.audio_lengths))]
        else:
            self.pred_mem = None
        
        self.boundaries = list(range(self.min_len, self.max_len+1))
            
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.audio_lengths)):
            length = self.audio_lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                try:
                    buckets[idx_bucket].append(i)
                except:
                    print(length)

        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
            
        if self.rank == 0:
            logger.info("****************** Buckets *********************")
            for i in range(len(buckets)):
                logger.info(f"[{self.boundaries[i], self.boundaries[i+1]}]:{len(buckets[i])}")
            logger.info("************************************************")

           
        
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # # add extra samples to make it evenly divisible
            # rem = num_samples_bucket - len_bucket
            # ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            if self.gpu_mem_limite < 0:
                # batching
                for j in range(len(ids_bucket) // self.batch_size):
                    batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
                    if len(batch) == self.batch_size:
                        batches.append(batch)
            else:
                cur_men = self.gpu_mem_limite
                batch = []
                for i in ids_bucket:
                    pred_mem = self.pred_mem[bucket[i]]
                    if len(batch) < 2 or cur_men - pred_mem > 0:
                        batch.append(bucket[i])
                        cur_men -= pred_mem
                    else:
                        batches.append(batch)
                        batch = [bucket[i]]
                        cur_men = self.gpu_mem_limite - pred_mem
                
                # Drop last
                if len(batch) > 1:
                    batches.append(batch)

        
        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
        
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
