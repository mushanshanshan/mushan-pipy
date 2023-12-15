from glob import glob
import librosa   
from tqdm import tqdm
import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio
import copy
from collections import defaultdict
from loguru import logger
from mushan.io import from_pickle
from mushan.text.eng.front_end import Frontend as ENFrontend
from mushan.text.chs.front_end import Frontend as CNFrontend
from mushan.models.bigv.utils import bigv_mel

class TorchStandardScaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, keepdim=True)
        assert (self.std != 0).all()
        
    def normalize(self, x):
        return (x - self.mean) / self.std

    def reverse_normalize(self, x):
        return x * self.std + self.mean
    

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

    def __init__(self, config, optinal = {}, tag='train', debug = False):
        self.audiopaths_sid_text = []
        self.rank = config.dist.rank
        self.config = config
        self.debug = debug
        self.optinal = optinal
        
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
        
        self.data_list = config.data.data_list
            
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
        
        
        self.dataset_lang_map = {
            'cmhq': 'english',
            'lib': 'english',
            'mls': 'english',
            'vctk': 'english',
            'dutch': 'dutch',
            'french': 'french',
            'german': 'german',
            'italian': 'italian',
            'polish': 'polish',
            'portuguese': 'portuguese',
            'spanish': 'spanish',
        }
        
        self.language_map = {
            'english': 0,
            'dutch': 1,
            'french': 2,
            'german': 3,
            'italian': 4,
            'polish': 5,
            'portuguese': 6,
            'spanish': 7,
        }
        
        if config.data.language == 'ch':
            self.language = 'ch'
            self.frontend = CNFrontend()
        else:
            self.language = 'en'
            self.frontend = ENFrontend()
            
        self.symbols_len = self.frontend.symbols_len
            
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

        
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
        
        if self.rank == 0:
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
        
        if self.rank == 0 and len(missing_file) > 0:
            logger.error(f"Missing data index: {missing_file}")
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.audio_lengths = audio_lengths
        self.text_lengths = text_lengths
        if self.rank == 0:
            logger.info(f"Avaliable data length: {len(self.audiopaths_sid_text)}")
            
    def get_audiopath(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        
        return {"audio_path": audiopath}
            
    def get_hubert(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        hu_filename = audiopath.replace("/wave/", "/feature/hubert/").replace(".flac", ".code")
        assert os.path.exists(hu_filename)
        
        hu = torch.load(hu_filename, map_location=torch.device('cpu'))
        hu, dur = torch.unique_consecutive(hu, return_counts=True)
        
        return {"hubert_code": hu, "hubert_dur": dur}
            
    def get_language_idx(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        lang_idx = -1
        
        for k,v in self.dataset_lang_map.items():
            if k in audiopath:
                lang_idx = self.language_map[v]
                break
        
        assert lang_idx != -1
        return {"language_idx": lang_idx}
            
    def get_mel_spec(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        spec_filename = audiopath.replace("/wave/", "/feature/mel_spec/").replace(".flac", f".mel")
        assert os.path.exists(spec_filename)
        
        spec = torch.load(spec_filename, map_location='cpu')
        return {"mel_spec": spec}
    
    def get_mel_ref(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        assert len(self.ref_dict[spk]) > 0

        ref = random.choice(self.ref_dict[spk])
        spec_filename = ref.replace("/wave/", "/feature/mel_spec/").replace(".flac", f".mel")
        assert os.path.exists(spec_filename)
        
        spec = torch.load(spec_filename, map_location='cpu')
        return {"mel_ref": spec}
    
    def get_phoneme(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        text_norm = self.frontend.pho_to_idx(text)
        return {"phoneme": text_norm}
        
            
    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        res = {}
        
        for key in self.data_list:
            try:
                func = getattr(self, f"get_{key}")
                res.update(func(audiopath_sid_text))
            except AttributeError:
                raise NotImplementedError("Class `{}` does not implement `{}`".format(self.__class__.__name__, key))
        
        return res

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)
    

class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, config=None, optional = {}, return_ids=False, tacotron=False):
        
        self.return_ids = return_ids
        self.data_list = config.data.data_list
        self.optional = optional
        
    
    def get_language_idx(self, batch, ids_sorted_decreasing):
        lang_idx = torch.LongTensor(len(batch))
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            lang_idx[i] = row['language_idx']

        return {"language_idx": lang_idx}
        
        
    def get_mel_spec(self, batch, ids_sorted_decreasing):
        max_spec_len = max([x['mel_spec'].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0]['mel_spec'].size(0), max_spec_len)
        spec_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            spec = row['mel_spec']
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

        return {"mel": spec_padded, 
                "mel_length": spec_lengths}
    
    def get_mel_ref(self, batch, ids_sorted_decreasing):
        max_ref_len = max([x["mel_ref"].size(1) for x in batch])
        ref_padded = torch.FloatTensor(len(batch), batch[0]['mel_ref'].size(0), max_ref_len)
        ref_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            ref = row["mel_ref"]
            ref_padded[i, :, :ref.size(1)] = ref
            
        return {"mel_ref": ref_padded}
    
    def get_hubert(self, batch, ids_sorted_decreasing):
        max_hu_len = max([len(x['hubert_code']) for x in batch])
        
        hu_lengths = torch.LongTensor(len(batch))
        hu_padded = torch.LongTensor(len(batch), max_hu_len)
        hu_padded.zero_()
        hu_dur_padded = torch.LongTensor(len(batch), max_hu_len)
        hu_dur_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            hu = row['hubert_code']
            hu_dur = row['hubert_dur']
            hu_padded[i, :hu.size(0)] = hu
            hu_dur_padded[i, :hu.size(0)] = hu_dur
            hu_lengths[i] = hu.size(0)

        return {"hubert": hu_padded, 
                "hubert_dur": hu_dur_padded,
                "hubert_length": hu_lengths}
    
    
    def get_phoneme(self, batch, ids_sorted_decreasing):
        max_pho_len = max([len(x['phoneme']) for x in batch])
        pho_lengths = torch.LongTensor(len(batch))
        pho_padded = torch.LongTensor(len(batch), max_pho_len)
        pho_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            pho = row['phoneme']
            pho_padded[i, :pho.size(0)] = pho
            pho_lengths[i] = pho.size(0)

        return {"phoneme": pho_padded, 
                "phoneme_length": pho_lengths}

    def __call__(self, batch):

        res = {}
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x['mel_spec'].size(1) for x in batch]),
            dim=0, descending=True)
        
        for key in self.data_list:
            try:
                func = getattr(self, f"get_{key}")
                res.update(func(batch, ids_sorted_decreasing))
            except AttributeError:
                raise NotImplementedError("Class `{}` does not implement `{}`".format(self.__class__.__name__, key))
        

        return res


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.audio_lengths
        self.batch_size = batch_size
        
        self.min_len, self.max_len = boundaries
        self.boundaries = list(range(self.min_len, self.max_len+1))
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
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
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              if len(batch) > 1:
                batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size


# class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
#     """
#     Maintain similar input lengths in a batch.
#     Length groups are specified by boundaries.
#     Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
#     It removes samples which are not included in the boundaries.
#     Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
#     """

#     def __init__(self, dataset, batch_size, boundaries, gpu_mem_limite=-1, gpu_mem_pred=None, num_replicas=None, rank=None, shuffle=True):
#         super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
#         self.audio_lengths = dataset.audio_lengths
#         self.text_lengths = dataset.text_lengths
#         self.rank = dataset.rank

#         self.batch_size = batch_size

#         self.min_len, self.max_len = boundaries
        
        
#         self.gpu_mem_limite = gpu_mem_limite
#         if self.gpu_mem_limite > 0:
#             self.pred_mem = [gpu_mem_pred(self.audio_lengths[i], self.text_lengths[i]) for i in range(len(self.audio_lengths))]
#         else:
#             self.pred_mem = None
        
#         self.boundaries = list(range(self.min_len, self.max_len+1))
            
#         self.buckets, self.num_samples_per_bucket = self._create_buckets()
        
#         self.total_size = sum(self.num_samples_per_bucket)
#         self.num_samples = self.total_size // self.num_replicas

#     def _create_buckets(self):
#         buckets = [[] for _ in range(len(self.boundaries) - 1)]
#         for i in range(len(self.audio_lengths)):
#             length = self.audio_lengths[i]
#             idx_bucket = self._bisect(length)
#             if idx_bucket != -1:
#                 try:
#                     buckets[idx_bucket].append(i)
#                 except:
#                     print(length)

#         for i in range(len(buckets) - 1, -1, -1):
#             if len(buckets[i]) == 0:
#                 buckets.pop(i)
#                 self.boundaries.pop(i + 1)

#         num_samples_per_bucket = []
#         for i in range(len(buckets)):
#             len_bucket = len(buckets[i])
#             total_batch_size = self.num_replicas * self.batch_size
#             rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
#             num_samples_per_bucket.append(len_bucket + rem)
            
#         if self.rank == 0:
#             logger.info("****************** Buckets *********************")
#             for i in range(len(buckets)):
#                 logger.info(f"[{self.boundaries[i], self.boundaries[i+1]}]:{len(buckets[i])}")
#             logger.info("************************************************")

           
        
#         return buckets, num_samples_per_bucket

#     def __iter__(self):
#         # deterministically shuffle based on epoch
#         g = torch.Generator()
#         g.manual_seed(self.epoch)

#         indices = []
#         if self.shuffle:
#             for bucket in self.buckets:
#                 indices.append(torch.randperm(len(bucket), generator=g).tolist())
#         else:
#             for bucket in self.buckets:
#                 indices.append(list(range(len(bucket))))

#         batches = []
        
#         for i in range(len(self.buckets)):
#             bucket = self.buckets[i]
#             len_bucket = len(bucket)
#             ids_bucket = indices[i]
#             num_samples_bucket = self.num_samples_per_bucket[i]

#             # # add extra samples to make it evenly divisible
#             # rem = num_samples_bucket - len_bucket
#             # ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

#             # subsample
#             ids_bucket = ids_bucket[self.rank::self.num_replicas]

#             if self.gpu_mem_limite < 0:
#                 # batching
#                 for j in range(len(ids_bucket) // self.batch_size):
#                     batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
#                     if len(batch) == self.batch_size:
#                         batches.append(batch)
#             else:
#                 cur_men = self.gpu_mem_limite
#                 batch = []
#                 for i in ids_bucket:
#                     pred_mem = self.pred_mem[bucket[i]]
#                     if len(batch) < 2 or cur_men - pred_mem > 0:
#                         batch.append(bucket[i])
#                         cur_men -= pred_mem
#                     else:
#                         batches.append(batch)
#                         batch = [bucket[i]]
#                         cur_men = self.gpu_mem_limite - pred_mem
                
#                 # Drop last
#                 if len(batch) > 1:
#                     batches.append(batch)

        
#         if self.shuffle:
#             batch_ids = torch.randperm(len(batches), generator=g).tolist()
#             batches = [batches[i] for i in batch_ids]
#         self.batches = batches
        
#         return iter(self.batches)

#     def _bisect(self, x, lo=0, hi=None):
#         if hi is None:
#             hi = len(self.boundaries) - 1

#         if hi > lo:
#             mid = (hi + lo) // 2
#             if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
#                 return mid
#             elif x <= self.boundaries[mid]:
#                 return self._bisect(x, lo, mid)
#             else:
#                 return self._bisect(x, mid + 1, hi)
#         else:
#             return -1

#     def __len__(self):
#         return self.num_samples // self.batch_size
