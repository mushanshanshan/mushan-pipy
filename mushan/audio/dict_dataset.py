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
import lmdb
import itertools
from collections import defaultdict
from loguru import logger
from mushan.io import from_pickle, to_pickle
from mushan.text.eng.front_end import Frontend as ENFrontend
from mushan.text.chs.front_end import Frontend as CNFrontend
from mushan.models.bigv.utils import bigv_mel
from mushan.audio.hifi_mel import mel_spectrogram as hifi_mel_spectrogram
from mushan.audio.lang_info import *
from librosa.util import normalize
from einops import rearrange, repeat, reduce

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
        
        random.shuffle(self.audiopaths_sid_text)
        
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
            logger.info(f"Remove {remove_counter} from {len(self.audiopaths_sid_text)} files.")
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
                if self.debug:
                    raise Exception(f"Key error: {audiopath}")
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
            logger.info(f"Before furation balance resample: {len(self.audiopaths_sid_text)}")
            
        new_audiopaths_sid_text = []
        
        for i in self.audiopaths_sid_text:
            dur = i[2]
            sample_times = max(1, int(dur // self.optional['audio_dur_balance_step_size']))
            for _ in range(sample_times):
                new_audiopaths_sid_text.append(i)
        
        
        self.audiopaths_sid_text = new_audiopaths_sid_text
        logger.info(f"After furation balance resample: {len(self.audiopaths_sid_text)}")


    def _repeat(self):
        """
        repeat filelist
        """
        if "repeat" in self.optional.keys():
            ori_length = len(self.audiopaths_sid_text)
            self.audiopaths_sid_text = self.audiopaths_sid_text * \
                self.optional['repeat']
            self.audio_lengths = self.audio_lengths * self.optional['repeat']
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
        audio_lengths = []
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

                    audiopaths_sid_text_new.append(
                        [audiopath, spk, dur, ori_text, pho])
                    total_dur += dur
                    audio_lengths.append(dur)
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
        self.audio_lengths = audio_lengths
        self.text_lengths = text_lengths
        if self.rank == 0:
            logger.info(
                f"Avaliable data length: {len(self.audiopaths_sid_text)}/{ori_data_length} | {int(total_dur/60/60)} hours")
            
            
            
    def pre_language_ref(self):
        if 'lang_ref_dict' in self.optional.keys():
             self.language_ref_dict = from_pickle(self.optional['lang_ref_dict'])
        else:
            self.language_ref_dict = from_pickle(f"/home/{self.username}/exp/s2/build_language/lang_ref")

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
    
    def get_dummy(self, audiopath_sid_text):
        return {}
    
    def get_language_ref(self, audiopath_sid_text):
        language_idx = self.get_language_idx(audiopath_sid_text)['language_idx']
        spk = audiopath_sid_text[1]
        candidates = random.sample(self.language_ref_dict[language_idx], 10)
        target = candidates[-1]
        for c in candidates:
            if c[1] != spk:
                target = c
                break
        pt = target[0]
        if self.debug:
            print(f"Language Ref : {pt}")
        
        mms = pt.replace("/wave/", "/feature/mms/").replace(".flac", ".l.44.norm")
        mms = torch.load(mms, mmap = True, map_location=torch.device('cpu'), weights_only=False)
        
        mel = pt.replace("/wave/", "/feature/mel_spec/").replace(".flac", ".160")
        mel = torch.load(mel, mmap = True, map_location=torch.device('cpu'), weights_only=False)
        mel = mel[:20, :]
        
        if self.debug:
            ori_mms_length = mms.shape[-1]
            ori_mel_length = mel.shape[-1]
            
        
        if self.debug:
            print(mms.shape[-1], int(mel.shape[-1] / 2))
        
        seg_length = self.optional["language_ref_length"]
        max_len = min(mms.shape[-1], int(mel.shape[-1] / 2)) - seg_length
        rand_start = random.randint(0, max_len)
        
        
        mms = mms[:, rand_start: rand_start + seg_length].clone()
        mms = mms.repeat_interleave(2, dim=1)
        mel = mel[:, rand_start * 2 : (rand_start + seg_length) * 2].clone()
        
        if mel.shape[-1] > mms.shape[-1]:
            mel = mel[:, :mms.shape[-1]]
        
        ref = torch.cat([mms, mel], dim = 0)
        
        if self.debug:
            print(f"lang mms: {rand_start / ori_mms_length} -> {(rand_start + seg_length) / ori_mms_length}")
            print(f"lang mel: {rand_start * 2 / ori_mel_length} -> {(rand_start + seg_length) * 2 / ori_mel_length}")
            print(f"language_ref length: {seg_length}")
        
        
        return {"language_ref": ref}
        
        
    def get_language_idx(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        
        if '/libri_16/' in audiopath:
            language_name = 'English'
            lang_id =  self.language_code[language_name]
        elif '/ftspeech/' in audiopath:
            language_name = 'Danish'
            lang_id =  self.language_code[language_name]
        elif '/vp_ita' in audiopath:
            language_name = 'Italian'
            lang_id =  self.language_code[language_name]
        elif '/mls_16/' in audiopath:
            language_name = self.language_map[audiopath.split('/')[-5]]
            lang_id =  self.language_code[language_name]
        elif '/fleurs_16/' in audiopath:
            language_name = self.language_map[audiopath.split('/')[-3]]
            lang_id =  self.language_code[language_name]
        elif '/mmstts_16/' in audiopath:
            language_name = self.language_map[audiopath.split('/')[-2]]
            lang_id =  self.language_code[language_name]
        else:
            raise Exception(f"Unknow language dataset: {audiopath}")
        return {'language_idx': lang_id, 'language_name': language_name}

    def get_audiopath(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text

        return {"audio_path": audiopath.replace(f'/home/{self.username}/data/wave/', '')}

    def get_hubert(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        hu_filename = audiopath.replace(
            "/wave/", "/feature/hubert/").replace(".flac", ".code")
        assert os.path.exists(hu_filename), hu_filename

        hu = torch.load(hu_filename, map_location=torch.device('cpu'), weights_only=False)
        hu, dur = torch.unique_consecutive(hu, return_counts=True)

        return {"hubert_code": hu, "hubert_dur": dur}

    def get_xlsr2b_feature_48(self, audiopath_sid_text, post_fix=".48"):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        mms_file = audiopath.replace(
            "/wave/", "/feature/xlsr2b/").replace(".flac", post_fix)
        seg_length = self.optional['xlsr_seg_size']

        data = torch.load(mms_file, mmap=True, map_location=torch.device('cpu'), weights_only=False)
        # data = torch.load(mms_file, mmap=False)
        rand_idx = random.randint(0, data.shape[-1] - seg_length)
        data = data[:, rand_idx: rand_idx+seg_length]
        return {"xlsr2b_feature_48": data}

    def get_mms_feature_48(self, audiopath_sid_text, post_fix=".48"):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        mms_file = audiopath.replace(
            "/wave/", "/feature/mms/").replace(".flac", post_fix)
        seg_length = self.optional['mms_seg_size']
        
        data = torch.load(mms_file, mmap=True, map_location=torch.device('cpu'), weights_only=False)
        rand_idx = random.randint(0, data.shape[-1] - seg_length)
        data = data[:, rand_idx: rand_idx+seg_length]
        if 'mms_mean' in self.optional.keys():
            data = (data - self.optional['mms_mean']) / self.optional['mms_std']
            
        return {"mms_feature_48": data}

    def get_mms_feature_47(self, audiopath_sid_text):
        res = self.get_mms_feature_48(audiopath_sid_text, post_fix=".47")
        return {"mms_feature_47": res['mms_feature_48']}

    def get_mms_feature_45(self, audiopath_sid_text):
        res = self.get_mms_feature_48(audiopath_sid_text, post_fix=".45")
        return {"mms_feature_45": res['mms_feature_48']}

    def get_mms_feature_44(self, audiopath_sid_text):
        res = self.get_mms_feature_48(audiopath_sid_text, post_fix=".44")
        return {"mms_feature_44": res['mms_feature_48']}
    
    def get_mms_feature_44b(self, audiopath_sid_text):
        res = self.get_mms_feature_48(audiopath_sid_text, post_fix=".44b")
        return {"mms_feature_44b": res['mms_feature_48']}

    def get_mms_44_seg(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        mms_file = audiopath.replace(
            "/wave/", "/feature/mms/").replace(".flac", ".44")
        data = torch.load(mms_file, mmap=True, weights_only=False).repeat_interleave(2, dim=1)
        seg_length = self.optional['mms_seg_size'] * 2
        rand_idx = random.randint(0, data.shape[-1] - seg_length - 1)
        self.temp_arg = rand_idx
        # print(f"MEL_{audiopath_sid_text[0]}_{data.shape}_{rand_idx}_{rand_idx+seg_length}")
        data = data[:, rand_idx: rand_idx+seg_length]

        return {"mms_44_seg": data}
    
    def get_mms_rvq_code(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        mms_file = audiopath.replace(
            "/wave/", "/feature/mms/").replace(".flac", self.optional['mms_rvq_code_postfix'])
        data = torch.load(mms_file, mmap=True, map_location='cpu', weights_only=False)
        data = rearrange(data, 'g l q -> (q g) l')
        
        return {"mms_rvq_code": data}
    
    def get_mms_rvq_code_pad_seg(self, audiopath_sid_text):
        return self.get_mms_rvq_code(audiopath_sid_text)
    
    def get_double_mms_rvq_code(self, audiopath_sid_text):
        data = self.get_mms_rvq_code(audiopath_sid_text)['mms_rvq_code']
        seg_length = self.optional['mms_seg_size']
        
        if (data.shape[-1] > seg_length):
            rand_idx = random.randint(0, data.shape[-1] - seg_length - 1)
            self.temp_arg = rand_idx
            if self.debug:
                temp_l = data.shape[-1]
                print(f'mms code: {rand_idx / temp_l} -> {(rand_idx+seg_length) / temp_l} of {temp_l}')
            # print(f"MMS_{audiopath_sid_text[0]}_{data.shape}_{rand_idx}_{rand_idx+seg_length}")
            data = data[:, rand_idx: rand_idx+seg_length]
        else:
            self.temp_arg = 0
            if self.debug:
                temp_l = data.shape[-1]
                print(f'mms code: 0 -> 1.0 of {temp_l}')
        
        data = data.repeat_interleave(2, dim=-1)
        return {"mms_rvq_code": data} 
            
    
    def get_mms_rvq_code_seq(self, audiopath_sid_text):
        data = self.get_mms_rvq_code(audiopath_sid_text)['mms_rvq_code']
        # data = rearrange(data, 'q l ->l q')
        # print(data.shape)
        # print(self.optional)
        if 'double_mms_code' in self.optional.keys() and self.optional['double_mms_code']:
            data = data.repeat_interleave(2, dim=-1)
            # print(data.shape)
            seg_length = self.optional['mms_seg_size'] * 2
            rand_idx = random.randint(0, data.shape[-1] - seg_length - 1)
            self.temp_arg = rand_idx
            if self.debug:
                print(f'mms code: {rand_idx} -> {rand_idx+seg_length} of {data.shape[-1]}')
            # print(f"MMS_{audiopath_sid_text[0]}_{data.shape}_{rand_idx}_{rand_idx+seg_length}")

            data = data[:, rand_idx: rand_idx+seg_length]
            # print(data.shape)
        else:
            seg_length = self.optional['mms_seg_size']
            rand_idx = random.randint(0, data.shape[-1] - seg_length - 1)
            self.temp_arg = rand_idx
            temp_l = data.shape[-1]
            if self.debug:
                print(f'mms code: {rand_idx / temp_l} -> {(rand_idx+seg_length) / temp_l} of {temp_l}')
            # print(f"MMS_{audiopath_sid_text[0]}_{data.shape}_{rand_idx}_{rand_idx+seg_length}")
            data = data[:, rand_idx: rand_idx+seg_length]
            data = data.repeat_interleave(2, dim=-1)
            
        return {"mms_rvq_code": data}
        
    def get_mhubert_code(self, audiopath_sid_text):
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/mhubert/",
                ".flac": ".code"
                # ".flac": ".align.code"
            },
            return_key="mhubert_code",
        )
        

    def get_mms_code(self, audiopath_sid_text):
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/mms/",
                ".flac": self.optional['mms_code_postfix']
                # ".flac": ".align.code"
            },
            return_key="mms_code",
        )
        
    def get_mms_code_pad_seg(self, audiopath_sid_text):
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/mms/",
                ".flac": self.optional['mms_code_postfix']
                # ".flac": ".align.code"
            },
            return_key="mms_code",
        )

    def get_mms_code_seg(self, audiopath_sid_text):
        data = self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/mms/",
                ".flac": self.optional['mms_code_postfix']
                # ".flac": ".align.code"
            },
            return_key="mms_code",
        )['mms_code']
        if 'double_mms_code' in self.optional.keys() and self.optional['double_mms_code']:
            data = data.repeat_interleave(2, dim=0)
            seg_length = self.optional['mms_seg_size'] * 2
            rand_idx = random.randint(0, data.shape[-1] - seg_length - 1)
            self.temp_arg = rand_idx
            # print(f"MMS_{audiopath_sid_text[0]}_{data.shape}_{rand_idx}_{rand_idx+seg_length}")

            data = data[rand_idx: rand_idx+seg_length]
        else:
            seg_length = self.optional['mms_seg_size']
            rand_idx = random.randint(0, data.shape[-1] - seg_length - 1)
            self.temp_arg = rand_idx
            # print(f"MMS_{audiopath_sid_text[0]}_{data.shape}_{rand_idx}_{rand_idx+seg_length}")
            data = data[rand_idx: rand_idx+seg_length]

        return {"mms_code": data}

    def get_hubert_code(self, audiopath_sid_text):
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/hubert/",
                ".flac": ".code"
            },
            return_key="hubert_code",
        )

    def get_enhubert_code(self, audiopath_sid_text):
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/hubert/",
                ".flac": ".en.code"
            },
            return_key="hubert_code",
        )

    def filter_mhubert_code(self, audiopath, spk, dur, ori_text, pho):
        return len(pho) / dur > 3 and len(pho) / dur < 25

    def filter_hubert_code(self, audiopath, spk, dur, ori_text, pho):
        return len(pho) / dur > 3 and len(pho) / dur < 25

    def filter_hubert(self, audiopath, spk, dur, ori_text, pho):
        return len(pho) / dur > 3 and len(pho) / dur < 25

    def get_robert(self, audiopath_sid_text, layer=3):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        pho_filename = audiopath.replace(
            "/wave/", "/feature/robert/").replace(".flac", ".p2f")
        assert os.path.exists(pho_filename), pho_filename
        robert_filename = audiopath.replace(
            "/wave/", "/feature/robert/").replace(".flac", f".{layer}.pt")
        assert os.path.exists(pho_filename), pho_filename

        pho_data = from_pickle(pho_filename)
        ro_data = torch.load(robert_filename, map_location=torch.device('cpu'))
        pho = pho_data['pho']
        p2w = pho_data['p2w']

        robert_feature = torch.zeros(768, len(p2w))
        # 遍历 fea_idx 并填充长特征 tensor
        for idx, key in enumerate(p2w):
            if key == -1:
                continue  # 已经是0，无需操作
            else:
                indices = pho_data['w2f'][key]
                if len(indices) == 1:
                    robert_feature[:, idx] = ro_data[:, indices[0]]
                elif len(indices) > 1:
                    robert_feature[:, idx] = ro_data[:, indices].mean(dim=1)

        if torch.isnan(robert_feature).any():
            print("nan error:", audiopath)

        pho = self.frontend.pho_to_idx(pho_data['pho'], add_blank=False)

        return {"phoneme": pho, "robert_feature": robert_feature}

    def get_mel_spec(self, audiopath_sid_text):
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/mel_spec/",
                ".flac": ".mel"
            },
            return_key="mel_spec",
        )

    def get_mel_spec_160(self, audiopath_sid_text):
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/mel_spec/",
                ".flac": ".160"
            },
            return_key="mel_spec",
        )
        
    def get_mms_corr_mel_160(self, audiopath_sid_text):
        data = self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/mel_spec/",
                ".flac": ".160"
            },
            return_key="mel_spec",
        )['mel_spec']

        seg_length = self.optional['mms_seg_size'] * 2
        self.temp_arg *= 2
        assert self.temp_arg != None, "Put the mel_spec_seg at the last in the datalist"
        
        if self.debug:
            temp_l = data.shape[-1]
            print(f'mel spec: {self.temp_arg / temp_l} -> {min(1, (self.temp_arg + seg_length) / temp_l)} of {temp_l}')
        
        data = data[:, self.temp_arg: self.temp_arg + seg_length]

        if "mel_spec_160_seg_mean" in self.optional.keys():
            data = (data - self.optional["mel_spec_160_seg_mean"]
                    ) / self.optional["mel_spec_160_seg_std"]

        return {"mel_spec": data}

    def get_mel_spec_160_seg(self, audiopath_sid_text):
        data = self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/mel_spec/",
                ".flac": ".160"
            },
            return_key="mel_spec",
        )['mel_spec']

        seg_length = self.optional['mms_seg_size'] * 2
        self.temp_arg *= 2
        assert self.temp_arg != None, "Put the mel_spec_seg at the last in the datalist"
        # print(f"MEL_{audiopath_sid_text[0]}_{data.shape}_{self.temp_arg}_{self.temp_arg + seg_length}")
        if self.debug:
            temp_l = data.shape[-1]
            print(f'mel spec: {self.temp_arg / temp_l} -> {(self.temp_arg + seg_length) / temp_l} of {temp_l}')
        
        data = data[:, self.temp_arg: self.temp_arg + seg_length]

        if "mel_spec_160_seg_mean" in self.optional.keys():
            data = (data - self.optional["mel_spec_160_seg_mean"]
                    ) / self.optional["mel_spec_160_seg_std"]

        return {"mel_spec": data}
    
    
    def get_mamba_text_mms(self, audiopath_sid_text):
        data = {}
        data.update(self.get_mms_code(audiopath_sid_text))
        data.update(self.get_text(audiopath_sid_text))
        return data
    
    def get_mms_pretrain(self, audiopath_sid_text):
        data = {}
        data.update(self.get_mms_code(audiopath_sid_text))
        code = data['mms_code']
        original_tensor = [original_tensor[0].item()]

        # 遍历原始 tensor，从第二个元素开始
        for i in range(1, len(original_tensor)):
            if original_tensor[i] != original_tensor[i - 1]:
                result_list.append(original_tensor[i].item())

        # 转换结果列表为 tensor
        result_tensor = torch.tensor(result_list)
        
        data.update(self.get_text(audiopath_sid_text))
        return data
        

    def get_mel_ref(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        assert len(self.ref_dict[spk]) > 0, spk

        ref = random.choice(self.ref_dict[spk])
        if self.debug:
            print(f"Speaker Ref : {ref}")
        spec_filename = ref.replace(
            "/wave/", "/feature/mel_spec/").replace(".flac", f".mel")
        assert os.path.exists(spec_filename), spec_filename

        spec = torch.load(spec_filename, map_location='cpu', weights_only=False)
        return {"mel_ref": spec}
    
    def get_club(self, audiopath_sid_text, l = 480, pad_value = 0):
        if 'club_loop_num' in self.optional.keys():
            n = self.optional['club_loop_num']
        else:
            n = 4
        
        feature_padded = torch.FloatTensor(n, 80, l)
        feature_padded.fill_(pad_value)

        for i in range(n):
            pt = random.choice(self.audiopaths_sid_text)
            feature = torch.load(pt[0].replace("/wave/", "/feature/mel_spec/").replace(".flac", f".160"))
            if feature.shape[-1] > l:
                start_idx = random.randint(0, feature.shape[-1] - l - 1)
                feature = feature[:, start_idx: start_idx + l]
                feature_padded[i, :, :] = feature
            else:
                cur_l = feature.shape[-1]
                feature_padded[i, :, :cur_l] = feature
                
        return {"club": feature_padded}

    def get_mel_160_ref(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        assert len(self.ref_dict[spk]) > 0, spk

        ref = random.choice(self.ref_dict[spk])
        if self.debug:
            print(f"Spk Ref : {ref}")
        spec_filename = ref.replace(
            "/wave/", "/feature/mel_spec/").replace(".flac", f".160")
        assert os.path.exists(spec_filename), spec_filename

        spec = torch.load(spec_filename, map_location='cpu', mmap=True, weights_only=False)
        if "mel_spec_160_seg_len" in self.optional.keys() and spec.shape[-1] > self.optional["mel_spec_160_seg_len"]:
            ref_rand_start = random.randint(0, spec.shape[-1] - self.optional["mel_spec_160_seg_len"] - 1) 
            ref_end = ref_rand_start + self.optional["mel_spec_160_seg_len"]
            spec = spec[:, ref_rand_start:ref_end]
        if "mel_spec_160_seg_mean" in self.optional.keys():
            spec = (spec - self.optional["mel_spec_160_seg_mean"]
                    ) / self.optional["mel_spec_160_seg_std"]

        return {"mel_ref": spec}

    def get_mhubert_ref(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        assert len(self.ref_dict[spk]) > 0, spk

        ref = random.choice(self.ref_dict[spk])
        spec_filename = ref.replace(
            "/wave/", "/feature/mhubert/").replace(".flac", ".code")
        assert os.path.exists(spec_filename), spec_filename

        spec = torch.load(spec_filename, map_location=torch.device('cpu'))
        if spec.shape[-1] > 150:
            start = random.randint(0, spec.shape[-1] - 150)
            spec = spec[start: start+150]
        return {"mhubert_ref": spec}

    def get_text(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        text_norm = self.frontend.textonly_to_idx(ori_text)
        return {"text": text_norm}
    
    def get_custom_text(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        text_norm = self._text_to_idx(ori_text)
        return {"text": text_norm}

    def get_phoneme(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        if "phoneme_idx_intersperse" in self.optional.keys() and self.optional['phoneme_idx_intersperse'] == True:
            text_norm = self.frontend.pho_to_idx(text, add_blank=True)
        else:
            text_norm = self.frontend.pho_to_idx(text, add_blank=False)
        return {"phoneme": text_norm}

    def get_wave_audio(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        data, sr = torchaudio.load(audiopath)
        return {"wave_audio": data.squeeze(0)}

    def get_seg_wave_audio(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        data, sr = torchaudio.load(audiopath)
        rand_idx = random.randint(
            0, data.shape[-1] - self.optional['wave_seg_size'])
        data = data[:, rand_idx: rand_idx+self.optional['wave_seg_size']]
        return {"wave_audio": data.squeeze(0)}

    def get_voco_seg_wave_audio_with_mel(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        audio, sr = torchaudio.load(audiopath)
        audio = audio.squeeze(0).numpy()
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        rand_idx = random.randint(
            0, audio.shape[-1] - self.optional['wave_seg_size'])
        audio = audio[:, rand_idx: rand_idx + self.optional['wave_seg_size']]
        
        mel = hifi_mel_spectrogram(audio,
                                   self.optional['wave_seg_mel_n_fft'],
                                   self.optional['wave_seg_mel_n_mel'],
                                   self.optional['wave_seg_mel_sr'],
                                   self.optional['wave_seg_mel_hop_size'],
                                   self.optional['wave_seg_mel_win_size'],
                                   self.optional['wave_seg_mel_fmin'],
                                   self.optional['wave_seg_mel_fmax'],
                                   center=False)
        
        loss_mel = hifi_mel_spectrogram(audio,
                                        self.optional['wave_seg_mel_n_fft'],
                                        self.optional['wave_seg_mel_n_mel'],
                                        self.optional['wave_seg_mel_sr'],
                                        self.optional['wave_seg_mel_hop_size'],
                                        self.optional['wave_seg_mel_win_size'],
                                        self.optional['wave_seg_mel_fmin'],
                                        self.optional['wave_seg_mel_fmax_loss'],
                                        center=False)
        
        audio = audio.squeeze(0)
        mel = mel.squeeze(0)
        loss_mel = mel.squeeze(0)
        return {"wave_audio": audio, "mel_spec": mel, "loss_mel_spec": loss_mel}

    def get_pad_wave_audio(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        data, sr = torchaudio.load(audiopath)
        if data.shape[-1] % 240 == 239:
            data = torch.cat((data, torch.zeros(1, 1)), dim=-1)
        return {"wave_audio": data.squeeze(0)}

    def get_linear_spec(self, audiopath_sid_text):
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/linear_spec/",
                ".flac": ".linear"
            },
            return_key="linear_spec",
        )

    def get_240_mel_spec(self, audiopath_sid_text):
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/mel_spec/",
                ".flac": ".240.mel"
            },
            return_key="mel_spec",
        )

    def get_240_linear_spec(self, audiopath_sid_text):
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/linear_spec/",
                ".flac": ".240.pad.linear"
            },
            return_key="linear_spec",
        )

    def get_your_ref(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        assert len(self.ref_dict[spk]) > 0, spk

        ref = random.choice(self.ref_dict[spk])
        spec_filename = ref.replace(".flac", ".emb")
        assert os.path.exists(spec_filename), spec_filename

        spec = torch.load(spec_filename, map_location=torch.device('cpu'))
        return {"your_ref": spec}

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        res = {}
        self.temp_arg = None
        for key in self.data_list:
            try:
                func = getattr(self, f"get_{key}")
                res.update(func(audiopath_sid_text))
            except AttributeError:
                raise NotImplementedError("Class `{}` does not implement `get_{}`".format(self.__class__.__name__, key))
            # func = getattr(self, f"get_{key}")
            # res.update(func(audiopath_sid_text))

        return res

    def __getitem__(self, index):
        if self.debug:
            print(f"GT path: {self.audiopaths_sid_text[index][0]}")
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, config=None, optional={}, return_ids=False, tacotron=False):

        self.return_ids = return_ids
        self.data_list = config.data.data_list
        self.optional = optional

    def collect_language_idx(self, batch, ids_sorted_decreasing):
        lang_idx = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            lang_idx[i] = row['language_idx']

        return {"language_idx": lang_idx}

    def collect_2D_with_length(self, batch, ids_sorted_decreasing, feature_key, length_dim_idx=1, max_length = 0, pad_value=0, feature_dtype=torch.float):
        max_feature_len = max(
            [x[feature_key].size(length_dim_idx) for x in batch])
        max_feature_len = max(max_feature_len, max_length)
        feature_lengths = torch.LongTensor(len(batch))

        if feature_dtype == torch.float or feature_dtype == torch.float32:
            feature_padded = torch.FloatTensor(
                len(batch), batch[0][feature_key].size(0), max_feature_len)
        elif feature_dtype == torch.long:
            feature_padded = torch.LongTensor(
                len(batch), batch[0][feature_key].size(0), max_feature_len)
        else:
            raise NotImplementedError

        feature_padded.fill_(pad_value)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            feature = row[feature_key]
            feature_padded[i, :, :feature.size(1)] = feature
            feature_lengths[i] = feature.size(1)

        return {feature_key: feature_padded,
                f"{feature_key}_length": feature_lengths}

    def collect_1D_with_length(self, batch, ids_sorted_decreasing, feature_key, pad_value=0, feature_dtype=torch.float, one_more_pad = False):

        max_feature_len = max([len(x[feature_key]) for x in batch])
        
        if one_more_pad:
            max_feature_len += 1
            
        feature_lengths = torch.LongTensor(len(batch))

        if feature_dtype == torch.float or feature_dtype == torch.float32:
            feature_padded = torch.FloatTensor(len(batch), max_feature_len)
        elif feature_dtype == torch.long:
            feature_padded = torch.LongTensor(len(batch), max_feature_len)
        else:
            raise NotImplementedError

        feature_padded.fill_(pad_value)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            feature = row[feature_key]
            feature_padded[i, :feature.size(0)] = feature
            feature_lengths[i] = feature.size(0)

        return {feature_key: feature_padded,
                f"{feature_key}_length": feature_lengths}
        
    def collect_mms_rvq_code(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_rvq_code",
            feature_dtype=torch.long,
            pad_value = 1025
        )
        
    def collect_mms_rvq_code_seq(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_rvq_code",
            feature_dtype=torch.long,
            pad_value = 1025
        )
        
    def collect_double_mms_rvq_code(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_rvq_code",
            feature_dtype=torch.long,
            pad_value = 1025,
            max_length = self.optional['mms_seg_size'] * 2
        )

    def collect_linear_spec(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="linear_spec",
            feature_dtype=torch.float
        )

    def collect_240_mel_spec(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mel_spec",
            feature_dtype=torch.float
        )

    def collect_240_linear_spec(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="linear_spec",
            feature_dtype=torch.float
        )

    def collect_xlsr2b_feature_48(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="xlsr2b_feature_48",
            feature_dtype=torch.float
        )
        
    def collect_language_ref(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="language_ref",
            feature_dtype=torch.float
        )

    def collect_mms_feature_48(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_feature_48",
            feature_dtype=torch.float
        )

    def collect_mms_feature_44(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_feature_44",
            feature_dtype=torch.float
        )
        
    def collect_mms_feature_44b(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_feature_44b",
            feature_dtype=torch.float
        )

    def collect_mms_44_seg(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_44_seg",
            feature_dtype=torch.float
        )

    def collect_mms_feature_47(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_feature_47",
            feature_dtype=torch.float
        )

    def collect_mms_feature_45(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_feature_45",
            feature_dtype=torch.float
        )

    def collect_mel_spec(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mel_spec",
            feature_dtype=torch.float
        )

    def collect_mel_spec_160(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mel_spec",
            feature_dtype=torch.float
        )

    def collect_mel_spec_160_seg(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mel_spec",
            feature_dtype=torch.float
        )
        
    def collect_mms_corr_mel_160(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mel_spec",
            feature_dtype=torch.float,
            max_length=self.optional['mms_seg_size'] * 2
        )

    def collect_audiopath(self, batch, ids_sorted_decreasing):
        info = []

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            info.append(row["audio_path"])

        return {"audio_path": info}
    
    def collect_dummy(self, batch, ids_sorted_decreasing):
        return {}
    
    def collect_mel_ref(self, batch, ids_sorted_decreasing):
        max_ref_len = max([x["mel_ref"].size(1) for x in batch])
        ref_padded = torch.FloatTensor(
            len(batch), batch[0]['mel_ref'].size(0), max_ref_len)
        ref_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            ref = row["mel_ref"]
            ref_padded[i, :, :ref.size(1)] = ref

        return {"mel_ref": ref_padded}

    def collect_mel_160_ref(self, batch, ids_sorted_decreasing):
        return self.collect_mel_ref(batch, ids_sorted_decreasing)

    def collect_hubert(self, batch, ids_sorted_decreasing, pad_value=1025):
        max_hu_len = max([len(x['hubert_code']) for x in batch])

        hu_lengths = torch.LongTensor(len(batch))
        hu_padded = torch.LongTensor(len(batch), max_hu_len)
        hu_padded.fill_(pad_value)
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
        
    def collect_mamba_text_mms(self, batch, ids_sorted_decreasing, pad_value=2059, text_shift = 2060, split_idx = 2260):
        max_data_length = max([x['mms_code'].size(0) + x['text'].size(0) + 2 for x in batch])
        
        text_lengths = torch.LongTensor(len(batch))
        mms_code_length = torch.LongTensor(len(batch))
        end_idx = torch.LongTensor(len(batch))
        data_padded = torch.LongTensor(len(batch), max_data_length)
        data_padded.fill_(pad_value)
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            mms_code = row['mms_code']
            text = row['text'] + text_shift
            mms_code_length[i] = mms_code.size(0)
            text_lengths[i] = text.size(0) + 1
            end_idx[i] = mms_code.size(0) + text.size(0) + 1
            data_padded[i, :text.size(0)] = text
            data_padded[i, text.size(0)] = split_idx
            data_padded[i, text.size(0) + 1: text.size(0) + 1 + mms_code.size(0)] = mms_code
            

        return {"data": data_padded,
                "end_idx": end_idx,
                'num_last_tokens': max_data_length - min(text_lengths),
                "mms_code_length": mms_code_length,
                "text_lengths": text_lengths}
        
    def collect_mms_rvq_code_pad_seg(self, batch, ids_sorted_decreasing, pad_value=1025):
        data = self.collect_mms_rvq_code(batch, ids_sorted_decreasing)['mms_rvq_code']
        bs, q, l = data.shape
        
        if l <= self.optional['mms_seg_size']:
            pad_data = torch.LongTensor(bs, q, self.optional['mms_seg_size'])
            pad_data.fill_(pad_value)
            pad_data[:, :, :l] = data
        else:
            randstart = random.randint(0, l - self.optional['mms_seg_size'])
            pad_data = data[:, :, randstart:randstart+self.optional['mms_seg_size']].clone()
            
        return {'mms_rvq_code': pad_data}

    def collect_hubert_code(self, batch, ids_sorted_decreasing, pad_value=1025):
        return self.collect_1D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="hubert_code",
            pad_value=pad_value,
            feature_dtype=torch.long
        )

    def collect_mms_code(self, batch, ids_sorted_decreasing, pad_value=2057):
        return self.collect_1D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_code",
            pad_value=pad_value,
            feature_dtype=torch.long,
            one_more_pad = True
        )
        
    def collect_mms_code_pad_seg(self, batch, ids_sorted_decreasing, pad_value=2057):
        feature_key = "mms_code"
        
        max_feature_len = max([len(x[feature_key]) for x in batch])
        feature_lengths = torch.LongTensor(len(batch))
        feature_padded = torch.LongTensor(len(batch), max(max_feature_len, self.optional['mms_seg_size']))

        feature_padded.fill_(pad_value)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            feature = row[feature_key]
            feature_padded[i, :feature.size(0)] = feature
            feature_lengths[i] = feature.size(0)
            
        if max_feature_len <= self.optional['mms_seg_size']:
            return {feature_key: feature_padded,
                    f"{feature_key}_length": feature_lengths}
        else:
            randstart = random.randint(0, max_feature_len - self.optional['mms_seg_size'])
            feature_seg = feature_padded[:, randstart:randstart+self.optional['mms_seg_size']].clone()
            feature_lengths.fill_(self.optional['mms_seg_size'])
            return {feature_key: feature_seg,
                    f"{feature_key}_length": feature_lengths}

    def collect_mms_code_seg(self, batch, ids_sorted_decreasing, pad_value=4100):
        return self.collect_1D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_code",
            pad_value=pad_value,
            feature_dtype=torch.long
        )

    def collect_mhubert_code(self, batch, ids_sorted_decreasing, pad_value=1025):
        return self.collect_1D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mhubert_code",
            pad_value=pad_value,
            feature_dtype=torch.long
        )
        
    def collect_club(self, batch, ids_sorted_decreasing):
        features = torch.stack([batch[ids_sorted_decreasing[i]]['club'] for i in range(len(ids_sorted_decreasing))])
        features = features.reshape(features.shape[0] * features.shape[1], features.shape[2], features.shape[3])
        return {'club': features}

    def collect_mhubert_ref(self, batch, ids_sorted_decreasing, pad_value=1025):
        max_feature_len = max([len(x["mhubert_ref"]) for x in batch]) + 1
        feature_lengths = torch.LongTensor(len(batch))

        feature_padded = torch.LongTensor(len(batch), max_feature_len)

        feature_padded.fill_(pad_value)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            feature = row["mhubert_ref"]
            feature_padded[i, :feature.size(0)] = feature
            feature_lengths[i] = feature.size(0)

        return {"mhubert_ref": feature_padded}

    def collect_enhubert_code(self, batch, ids_sorted_decreasing, pad_value=511):
        return self.collect_1D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="hubert_code",
            pad_value=pad_value,
            feature_dtype=torch.long
        )

    def collect_robert(self, batch, ids_sorted_decreasing):
        max_pho_len = max([len(x['phoneme']) for x in batch])

        pho_lengths = torch.LongTensor(len(batch))
        pho_padded = torch.LongTensor(len(batch), max_pho_len)
        robert_padded = torch.FloatTensor(
            len(batch), batch[0]['robert_feature'].size(0), max_pho_len)
        pho_padded.zero_()
        robert_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            pho = row['phoneme']
            pho_padded[i, :pho.size(0)] = pho
            pho_lengths[i] = pho.size(0)

            robert = row['robert_feature']
            robert_padded[i, :, :robert.size(1)] = robert

        return {"phoneme": pho_padded,
                "phoneme_length": pho_lengths,
                "robert_feature": robert_padded}

    def collect_text(self, batch, ids_sorted_decreasing):
        max_pho_len = max([len(x['text']) for x in batch])
        pho_lengths = torch.LongTensor(len(batch))
        pho_padded = torch.LongTensor(len(batch), max_pho_len)
        pho_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            pho = row['text']
            pho_padded[i, :pho.size(0)] = pho
            pho_lengths[i] = pho.size(0)

        return {"text": pho_padded,
                "text_length": pho_lengths}
        
    def collect_custom_text(self, batch, ids_sorted_decreasing):
        return self.collect_text(batch, ids_sorted_decreasing)

    def collect_phoneme(self, batch, ids_sorted_decreasing):
        max_pho_len = max([len(x['phoneme']) for x in batch])
        pho_lengths = torch.LongTensor(len(batch))
        pho_padded = torch.LongTensor(len(batch), max_pho_len)
        if 'text_pad_token' in self.optional.keys():
            pho_padded.fill_(self.optional['text_pad_token'])
        else:
            pho_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            pho = row['phoneme']
            pho_padded[i, :pho.size(0)] = pho
            pho_lengths[i] = pho.size(0)

        return {"phoneme": pho_padded,
                "phoneme_length": pho_lengths}

    def collect_your_ref(self, batch, ids_sorted_decreasing):
        max_ref_len = max([len(x['your_ref']) for x in batch])
        ref_padded = torch.FloatTensor(len(batch), max_ref_len)
        ref_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            ref = row['your_ref']
            ref_padded[i, :ref.size(0)] = ref

        return {"your_ref": ref_padded}

    def collect_wave_audio(self, batch, ids_sorted_decreasing):
        max_wave_len = max([len(x['wave_audio']) for x in batch])
        wave_lengths = torch.LongTensor(len(batch))
        wave_padded = torch.FloatTensor(len(batch), max_wave_len)
        wave_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            wave = row['wave_audio']
            wave_padded[i, :wave.size(0)] = wave
            wave_lengths[i] = wave.size(0)

        return {"wave_audio": wave_padded,
                "wave_audio_length": wave_lengths}

    def collect_pad_wave_audio(self, batch, ids_sorted_decreasing):
        return self.collect_wave_audio(batch, ids_sorted_decreasing)

    def collect_seg_wave_audio(self, batch, ids_sorted_decreasing):
        return self.collect_wave_audio(batch, ids_sorted_decreasing)

    def collect_nor_seg_wave_audio(self, batch, ids_sorted_decreasing):
        return self.collect_wave_audio(batch, ids_sorted_decreasing)

    def collect_voco_seg_wave_audio_with_mel(self, batch, ids_sorted_decreasing):
        res = self.collect_wave_audio(batch, ids_sorted_decreasing)
        res.update(self.collect_mel_spec(batch, ids_sorted_decreasing))
        res.update(self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="loss_mel_spec",
            feature_dtype=torch.float
        ))
        return res

    def __call__(self, batch):
        sort_key = None
        res = {}

        for i in ['mel_spec', 'mel_spec_160', 'linear_spec', 'mms_feature_44b', 'mms_feature_44', 'xlsr2b_feature_48']:
            if i in batch[0].keys():
                sort_key = i
                _, ids_sorted_decreasing = torch.sort(
                    torch.LongTensor([x[sort_key].size(1) for x in batch]),
                    dim=0, descending=True)
                break

        # dim 0 排序
        if sort_key == None:
            for i in ['text', 'phoneme', 'wave_audio', 'seg_wave_audio', 'mms_code_seg', 'mms_code']:
                if i in batch[0].keys():
                    sort_key = i
                    _, ids_sorted_decreasing = torch.sort(
                        torch.LongTensor([x[sort_key].size(0) for x in batch]),
                        dim=0, descending=True)
                    break

        for key in self.data_list:
            try:
                func = getattr(self, f"collect_{key}")
                res.update(func(batch, ids_sorted_decreasing))
            except AttributeError:
                raise NotImplementedError(
                    "Class `{}` does not implement `{}`".format(self.__class__.__name__, key))

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
