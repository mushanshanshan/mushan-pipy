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
import pickle
import types 
import lmdb
import itertools
import inspect
from collections import defaultdict

from loguru import logger
from mushan.io import from_pickle, to_pickle
from mushan.text.eng.front_end import Frontend as ENFrontend
from mushan.text.chs.front_end import Frontend as CNFrontend
from mushan.models.bigv.utils import bigv_mel
from mushan.audio.hifi_mel import mel_spectrogram as hifi_mel_spectrogram
from mushan.audio.lang_info import *
from mushan.audio.codec_mel import mel_spectrogram as codec_mel_spectrogram
from librosa.util import normalize
from einops import rearrange, repeat, reduce
from transformers import AutoFeatureExtractor, AutoTokenizer, HfArgumentParser
from mushan.audio.dataset_funcs import get_funcs, collect_funcs
from mushan.models import get_dac

def build_black_list(filename, key, target_dir = None):
    if target_dir != None:
        key_black_list_path = target_dir
    else:
        key_black_list_path = f"/home/{os.getlogin()}/data/filelists/blacklists/{key}.pk"
    if os.path.exists(key_black_list_path):
        blist = from_pickle(key_black_list_path)
    else:
        blist = []
    blist += [audiopath.split("/")[-1].split(".")[0] for audiopath in filename]
    to_pickle(key_black_list_path, blist)

    return os.path.exists(key_black_list_path)


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

    def __init__(self, config, optional={}, tag='train', debug=False, username=None):
        self.audiopaths_sid_text = []
        self.rank = config.dist.rank
        self.config = config
        self.debug = debug
        self.optional = optional
    

        
        func_counter = 0
        for func_name, func in inspect.getmembers(get_funcs, inspect.isfunction):
            if func.__module__ == get_funcs.__name__ and func_name.startswith("get_"):
                setattr(self, func_name, func.__get__(self, self.__class__))
                func_counter += 1

        if self.rank == 0:
            logger.info(f"Import {func_counter} GET functions from external.")
        
        
        if username == None:
            self.username = os.getlogin()
        else:
            self.username = username

        if tag == 'train':
            for i in config.train.train_filelists:
                if "*" in i:
                    fls_list = glob(i)
                    if self.rank == '0':
                        logger.info(f"Got {len(fls_list)} filelists from {i}.")
                    for _f in fls_list:
                        temp = load_filepaths_and_text(_f)
                        self.audiopaths_sid_text += temp
                else:
                    temp = load_filepaths_and_text(i)
                    self.audiopaths_sid_text += temp
                if self.rank == '0':
                    logger.info(
                        f"Added train file : {i}, length : {len(temp)}")
        elif tag == 'eval':
            for i in config.train.eval_filelists:
                temp = load_filepaths_and_text(i)
                self.audiopaths_sid_text += temp
                if self.rank == '0':
                    logger.info(
                        f"Added train file : {i}, length : {len(temp)}")
        elif tag == 'tune':
            for i in config.train.tune_filelists:
                temp = load_filepaths_and_text(i)
                self.audiopaths_sid_text += temp
                if self.rank == '0':
                    logger.info(
                        f"Added train file : {i}, length : {len(temp)}")

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

        self.need_text = any("text" in i for i in self.data_list)
        self.need_phoneme = any("phoneme" in i for i in self.data_list)
        
        
        if 'text_to_id_map' in self.optional.keys():
            self.text_to_id_map = self.optional['text_to_id_map']
        else:
            self.text_to_id_map = None

        self.language_map = {}
        self.language_map.update(mls_language_map)
        self.language_map.update(fleurs_language_map)
        self.language_map.update(mmstts_language_map)
        self.language_code = language_code
        
        self.language_class = max(list(self.language_map.values()))
        
        self.ori_text_dict = {}
        self.text_dict = {}

        if config.data.language == 'ch':
            self.language = 'ch'
            self.frontend = CNFrontend()
        else:
            self.language = 'en'
            self.frontend = ENFrontend()

        self.symbols_len = self.frontend.symbols_len

        for key in self.data_list:
            try:
                func = getattr(self, f"pre_{key}")
                func()
            except AttributeError:
                pass
        
        self.temp_arg = None
        random.seed(1234)
        self._blacklist_filter()
        if "dnsmos_filter" in self.optional.keys():
            self._dnsmos_filter()
        self._filter()
        self._language_balance()
        self._duration_balance()
        self._repeat()
        self._pre_nar_language_ref()

        self.tokenizer = {}
        
        if 'dac_tokenizer' in self.optional.keys():
            self.tokenizer['dac_tokenizer'] = get_dac(model_type=self.optional["dac_tokenizer"])
            
        if 'prepare_tokenizer' in self.optional.keys():
            self.pre_trans_input_text_token()
        
        if 'language_balance_sampler' in self.optional.keys():
            self._pre_language_balance_sampler()
        else:
            random.shuffle(self.audiopaths_sid_text)
            self.audio_lengths = [float(i[2]) for i in self.audiopaths_sid_text]
            self.dataset_length = len(self.audiopaths_sid_text)
        
        if self.rank == 0:
            logger.info("=" * 40)
            logger.info("Finished dataset init!")
            logger.info("=" * 40)
        
    def _blacklist_filter(self):
        blacklist = set()
        new_audiopaths_sid_text = []
        
        remove_counter = 0
        
        for key in self.data_list:
            filter_func = getattr(self, f"filter_{key}", None)
            if callable(filter_func):
                filter_funcs.append(filter_func)
            key_black_list_path = f"/home/{self.username}/data/filelists/blacklists/{key}.pk"
            if os.path.exists(key_black_list_path):
                cur_blacklist = from_pickle(key_black_list_path)
                blacklist = blacklist | set(cur_blacklist)
                
        for i in self.audiopaths_sid_text:
            if len(i) == 5:
                audiopath, _, dur, _, _ = i
            elif len(i) == 4:
                audiopath, _, dur, _ = i
                
            if audiopath.split("/")[-1].split(".")[0] in blacklist:
                remove_counter += 1
                continue
            
            new_audiopaths_sid_text.append(i)
        
        if self.rank == 0:
            logger.info("=" * 40)
            logger.info(f"Remove {remove_counter} from {len(self.audiopaths_sid_text)} files by blacklist filter.")
            logger.info(f"Now remain {len(new_audiopaths_sid_text)} files.")
            logger.info("=" * 40)
            
        self.audiopaths_sid_text = new_audiopaths_sid_text
        
    
    def _dnsmos_filter(self):
        
        mos = from_pickle(f"/home/{self.username}/data/filelists/blacklists/dnsmos.pk")
        
        if self.optional["dnsmos_filter"] == 0.05:
            sig_th = 2.90
            bak_th = 2.53
            over_th = 2.17
        elif self.optional["dnsmos_filter"] == 0.1:
            sig_th = 3.08
            bak_th = 2.86
            over_th = 2.38
        elif self.optional["dnsmos_filter"] == 0.2:
            sig_th = 3.24
            bak_th = 3.24
            over_th = 2.62
        else:
            raise Exception(f"Get unexcepted dnsmos_filter threshold: {self.optional['dnsmos_filter']}, need to be: 0.05 / 0.1 / 0.2")
        
        ori_dur = 0
        val_dur = 0
        new_audiopaths_sid_text = []
        miss = 0
        
        
        for i in self.audiopaths_sid_text:
            if len(i) == 5:
                audiopath, _, dur, _, _ = i
            elif len(i) == 4:
                audiopath, _, dur, _ = i
                
            dur = float(dur)
            ori_dur += dur
            try:
                sig, bak, o = mos[audiopath.split("/")[-1].split(".")[0]]
            except KeyError:
                # if self.debug:
                #     raise Exception(f"Key error: {audiopath}")
                miss += 1
                continue

            if sig > sig_th and bak > bak_th and o > over_th:
                val_dur += dur
                new_audiopaths_sid_text.append(i)
        
        if self.rank == 0:
            logger.info("=" * 40)
            logger.info(f"Using DNSMOS filter: {miss} files missing dnsmos scores")
            logger.info(f"before filter: {ori_dur / 60 / 60} hours; after filter: {val_dur / 60 / 60} hours")
            logger.info("=" * 40)
    
    def _intersperse(self, seq, item):
        result = [item] * (len(seq) * 2 + 1)
        result[1::2] = seq
        return result
        
    def _text_to_idx(self, text, add_blank=False, inter_item = 0):
        assert self.text_to_id_map != None
        sequence = []
        for symbol in pho:
            symbol_id = self.text_to_id_map[symbol]
            sequence += [symbol_id]
        
        if add_blank:
            sequence = self._intersperse(sequence, inter_item)
        
        sequence = torch.LongTensor(sequence)
        return sequence
    
    def _language_balance(self):
        if 'max_language_dur' not in self.optional.keys() or 'min_language_dur' not in self.optional.keys():
            return
        
        if 'max_language_ratio' not in self.optional.keys():
            max_language_ratio = 5
        else:
            max_language_ratio = self.optional['max_language_ratio']
            
        language_counter = defaultdict(int)
        language_set = defaultdict(list)
        for i in self.audiopaths_sid_text:
            lang = self.get_language_idx(i)['language_name']
            language_counter[lang] += i[2]
            language_set[lang].append(i)
                
        new_language_counter = defaultdict(int)
        new_audiopaths_sid_text = []
        
        for lang, fls in language_set.items():
            length = len(fls)
            i = 0
            while True:
                cur = i % length
                i += 1
                
                new_audiopaths_sid_text.append(fls[cur])
                new_language_counter[lang] += fls[cur][2]
                
                if new_language_counter[lang] > self.optional['max_language_dur'] * 60 * 60:
                    break
                if i > length and new_language_counter[lang] > self.optional['min_language_dur'] * 60 * 60:
                    break
                if i / length > max_language_ratio:
                    break
                
        self.audiopaths_sid_text = new_audiopaths_sid_text
        
        if self.rank == 0:
            logger.info("=" * 40)
            logger.info("Using language balancer, after balance:")
            for k, v in new_language_counter.items():
                logger.info(f"{k}: {(language_counter[k] / 60 / 60):.2f} -> {(v / 60 / 60):.2f} hours| ratio: {(v / language_counter[k]):.2f}")
        
                
    def _duration_balance(self):
        if 'audio_dur_balance_step_size' not in self.optional.keys():
            return
        
        if self.rank == 0:
            logger.info("=" * 40)
            logger.info(f"Using audio length balancer, ratio: {str(self.optional['audio_dur_balance_step_size'])}")
            ori_language_counter = defaultdict(int)
            new_language_counter = defaultdict(int)
            for i in self.audiopaths_sid_text:
                lang = self.get_language_idx(i)['language_name']
                ori_language_counter[lang] += i[2]
            
        new_audiopaths_sid_text = []
        
        for i in self.audiopaths_sid_text:
            dur = i[2]
            sample_times = max(1, int(dur // self.optional['audio_dur_balance_step_size']))
                
            for _ in range(sample_times):
                new_audiopaths_sid_text.append(i)
        
        
        self.audiopaths_sid_text = new_audiopaths_sid_text
        for i in self.audiopaths_sid_text:
            lang = self.get_language_idx(i)['language_name']
            if self.rank == 0:
                new_language_counter[lang] += i[2]
            
        if self.rank == 0:
            for k, v in new_language_counter.items():
                logger.info(f"{k}: {(ori_language_counter[k] / 60 / 60):.2f} -> {(v / 60 / 60):.2f} hours| ratio: {(v / ori_language_counter[k]):.2f}")

    
    def _pre_language_balance_sampler(self):
        
        self.lang_audiopaths_sid_text = defaultdict(list)
        self.lang_sample_length = {}

        for i in self.audiopaths_sid_text:
            lang = self.get_language_idx(i)['language_name']
            self.lang_audiopaths_sid_text[lang].append(i)
        
        self.lang_list = list(self.lang_audiopaths_sid_text.keys())
        self.audiopaths_sid_text = []
        
        for k, v in self.lang_audiopaths_sid_text.items():
            self.lang_sample_length[k] = len(v)
            random.shuffle(self.lang_audiopaths_sid_text[k])
            
        self.max_sample_per_lang = max(list(self.lang_sample_length.values()))
        self.dataset_length = self.max_sample_per_lang * len(list(self.lang_sample_length.keys()))
        self.lang_sample_counter = {k: 0 for k in self.lang_list}
        self.audio_lengths = []
        
        
        for i in range(self.dataset_length):
            lang = self.lang_list[i % len(self.lang_list)]
            idx = self.lang_sample_counter[lang] % self.lang_sample_length[lang]
            self.lang_sample_counter[lang] += 1
            self.audiopaths_sid_text.append(self.lang_audiopaths_sid_text[lang][idx])
            self.audio_lengths.append(self.audiopaths_sid_text[-1][2])
            
        
        if self.rank == 0:
            logger.info("=" * 40)
            logger.info(f"Using language balance sampler, it may cause potential issue with the time bucket sampler, Pls use the default sampler!")
            logger.info("Language data distribution:")
            for lang in self.lang_list:
                logger.info(f"{lang}: {self.lang_sample_length[lang]} -> {self.max_sample_per_lang}")
            logger.info(f"Total dataset length: {sum(list(self.lang_sample_length.values()))} -> {self.dataset_length}")
        
        

    def _repeat(self):
        """
        repeat filelist
        """
        if "repeat" in self.optional.keys():
            ori_length = len(self.audiopaths_sid_text)
            self.audiopaths_sid_text = self.audiopaths_sid_text * \
                self.optional['repeat']
            self.text_lengths = self.text_lengths * self.optional['repeat']
            if self.rank == 0:
                logger.info(
                    f"Repeat dataset from {ori_length} to {len(self.audiopaths_sid_text)}")
                

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        text_lengths = []
        missing_file = []
        filter_funcs = []
        total_dur = 0
        
        ori_data_length = len(self.audiopaths_sid_text)
        
        dataset_counter = defaultdict(int)

        if self.rank == 0:
            logger.info(f"Rank {self.rank} start processing filelists...")
        for i in tqdm(range(len(self.audiopaths_sid_text)), disable=self.rank != 0):
            try:
                if len(self.audiopaths_sid_text[i]) == 5:
                    audiopath, spk, dur, ori_text, pho = self.audiopaths_sid_text[i]
                elif len(self.audiopaths_sid_text[i]) == 4:
                    audiopath, spk, dur, ori_text = self.audiopaths_sid_text[i]
                    pho = '_'
                else:
                    continue
                dur = float(dur)

                val = True
                val = val and dur > self.min_audio_len and dur < self.max_audio_len
                
                if pho != '_':
                    val = val and len(pho) > self.min_text_len and len(
                        pho) < self.max_text_len
                else:
                    val = val and len(ori_text) > self.min_text_len and len(
                        ori_text) < self.max_text_len
                    
                for filter_func in filter_funcs:
                    val = val and filter_func(
                        audiopath, spk, dur, ori_text, pho)

                if val:
                    dataset_counter[audiopath.split("/")[5]] += 1
                    if not self.need_text:
                        ori_text = " "
                    if not self.need_phoneme:
                        pho = " "

                    self.ori_text_dict[audiopath] = ori_text
                    self.text_dict[audiopath] = pho
                    
                    ori_text = ""
                    pho = ""
                    
                    audiopaths_sid_text_new.append(
                        [audiopath, spk, dur, ori_text, pho])
                    total_dur += dur
                    text_lengths.append(len(pho))

                    self.ref_dict[spk].append(audiopath)

            except Exception as e:
                print(e)
                # exit(1)

        if self.rank == 0 and len(missing_file) > 0:
            logger.error(f"Missing data index: {missing_file}")
        if self.rank == 0:
            logger.info(f"Dataset counter: {dataset_counter}")
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.text_lengths = text_lengths
        if self.rank == 0:
            logger.info(
                f"Avaliable data length: {len(self.audiopaths_sid_text)}/{ori_data_length} | {int(total_dur/60/60)} hours")
            
    def filter_mhubert_code(self, audiopath, spk, dur, ori_text, pho):
        return len(pho) / dur > 3 and len(pho) / dur < 25

    def filter_hubert_code(self, audiopath, spk, dur, ori_text, pho):
        return len(pho) / dur > 3 and len(pho) / dur < 25

    def filter_hubert(self, audiopath, spk, dur, ori_text, pho):
        return len(pho) / dur > 3 and len(pho) / dur < 25
            
    def pre_language_ref(self):
        if 'lang_ref_dict' in self.optional.keys():
            self.language_ref_dict = from_pickle(self.optional['lang_ref_dict'])
        else:
            self.language_ref_dict = from_pickle(f"/home/{self.username}/exp/s2/build_language/lang_ref")
            
    def _pre_nar_language_ref(self):
        self.pre_language_ref()
        
    def pre_trans_input_text_token(self):
        self.tokenizer["input_text"] = AutoTokenizer.from_pretrained(
                                        "google-t5/t5-small",
                                        trust_remote_code=True,
                                        use_fast=True,
                                        padding_side="left",
                                    )
        

    def torch_load_single(self, audiopath_sid_text, path_replaecments, return_key, post_process=[]):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        target_file = audiopath

        for k, v in path_replaecments.items():
            target_file = target_file.replace(k, v)

        assert os.path.exists(target_file), target_file
        data = torch.load(target_file, mmap = True, map_location=torch.device('cpu'), weights_only=False)

        for function in post_process:
            data = function(data)

        return {return_key: data}
    

    def __getitem__(self, index):
        if self.debug:
            print(f"GT path: {self.audiopaths_sid_text[index][0]}")
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return self.dataset_length


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, config=None, optional={}, return_ids=False, debug=False):

        self.return_ids = return_ids
        self.rank = config.dist.rank
        self.data_list = config.data.data_list
        self.optional = optional
        self.warn_msg = set()
        self.debug = debug
        

        self.tokenizer = {}
        self.tokenizer["input_text"] = AutoTokenizer.from_pretrained(
                                        "google-t5/t5-small",
                                        trust_remote_code=True,
                                        use_fast=True,
                                        padding_side="left",
                                    )
        
        func_counter = 0
        for func_name, func in inspect.getmembers(collect_funcs, inspect.isfunction):
            if func.__module__ == collect_funcs.__name__ and func_name.startswith("collect_"):
                setattr(self, func_name, func.__get__(self, self.__class__))
                func_counter += 1

        if self.rank == 0:
            logger.info(f"Import {func_counter} COLLECT functions from external.")

    def __call__(self, batch):
        sort_key = None
        res = {}

        for i in ['mel_spec', 'mel_spec_160', 'linear_spec', 'mms_feature_44b', 'mms_feature_44', 'mms_44_seg', 'xlsr2b_feature_48']:
            if i in batch[0].keys():
                sort_key = i
                _, ids_sorted_decreasing = torch.sort(
                    torch.LongTensor([x[sort_key].size(1) for x in batch]),
                    dim=0, descending=True)
                break

        # dim 0 排序
        if sort_key == None:
            for i in ['text', 'phoneme', 'wave_audio', 'seg_wave_audio', 'mms_code_seg', 'mms_code', 'length_dummy']:
                if i in batch[0].keys():
                    sort_key = i
                    _, ids_sorted_decreasing = torch.sort(
                        torch.LongTensor([x[sort_key].size(0) for x in batch]),
                        dim=0, descending=True)
                    break
        
        if sort_key == None:
            if "sort_key_not_found" not in self.warn_msg:
                print("sort_key_not_found in data collector!")
                self.warn_msg.add("sort_key_not_found")
            num_batch = len(batch)
            ids_sorted_decreasing = torch.arange(0, num_batch, dtype=torch.long)
            shuffle_dummy = torch.randperm(num_batch)
            ids_sorted_decreasing=ids_sorted_decreasing[shuffle_dummy]
            
        # print(type(ids_sorted_decreasing))
        # print(ids_sorted_decreasing)

        for key in self.data_list:
            try:
                func = getattr(self, f"collect_{key}")
                res.update(func(batch, ids_sorted_decreasing))
            except AttributeError:
                raise NotImplementedError(
                    "Class `{}` does not implement `{}`".format(self.__class__.__name__, key))

        return res


class VitsDistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
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
        self.boundaries = boundaries
  
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

        if len(boundaries) == 2:
            self.min_len, self.max_len = boundaries
            self.boundaries = list(range(self.min_len, self.max_len+1))
        else:
            self.boundaries = boundaries

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
            rem = (total_batch_size - (len_bucket %
                   total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)

        if self.rank == 0:
            logger.info("****************** Buckets *********************")
            for i in range(len(buckets)):
                logger.info(
                    f"[{self.boundaries[i], self.boundaries[i+1]}]:{len(buckets[i])}")
            logger.info("************************************************")

        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(
                    len(bucket), generator=g).tolist())
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
            ids_bucket = ids_bucket + ids_bucket * \
                (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j *
                                                           self.batch_size:(j+1)*self.batch_size]]
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


class OLDDistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
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
        self.boundaries = boundaries
  
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
                logger.info(
                    f"[{self.boundaries[i], self.boundaries[i+1]}]:{len(buckets[i])}")
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