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
from mushan.audio.codec_mel import mel_spectrogram as codec_mel_spectrogram
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
        for key in self.data_list:
            try:
                func = getattr(self, f"filter_{key}")
                func()
            except AttributeError:
                pass
        self._filter()
        
        
        self._duration_balance()
        self._language_balance()
        
        self._repeat()
        self.pre_nar_language_ref()
        
        random.shuffle(self.audiopaths_sid_text)
        self.audio_lengths = [float(i[2]) for i in self.audiopaths_sid_text]
        
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
            logger.info(f"Before duration balance resample: {len(self.audiopaths_sid_text)}")
            
        new_audiopaths_sid_text = []
        
        for i in self.audiopaths_sid_text:
            dur = i[2]
            sample_times = max(1, int(dur // self.optional['audio_dur_balance_step_size']))
                
            for _ in range(sample_times):
                new_audiopaths_sid_text.append(i)
        
        
        self.audiopaths_sid_text = new_audiopaths_sid_text
        if self.rank == 0:
            logger.info(f"After duration balance resample: {len(self.audiopaths_sid_text)}")


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
            
            
            
    def pre_language_ref(self):
        if 'lang_ref_dict' in self.optional.keys():
             self.language_ref_dict = from_pickle(self.optional['lang_ref_dict'])
        else:
            self.language_ref_dict = from_pickle(f"/home/{self.username}/exp/s2/build_language/lang_ref")
            
    def pre_nar_language_ref(self):
        self.pre_language_ref()

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
    
    def get_nar_language_ref(self, audiopath_sid_text):
        lang_ref_length = self.optional['language_ref_length']
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
        
        mms = pt.replace("/wave/", "/feature/mms/").replace(".flac", self.optional['mms_rvq_code_postfix'])
        mms = torch.load(mms, mmap = True, map_location=torch.device('cpu'), weights_only=False)
        mms = rearrange(mms, 'g l q -> (q g) l')
        
        
        melvq = pt.replace("/wave/", "/feature/melvq/").replace(".flac", self.optional['melvq_code_postfix'])
        melvq = torch.load(melvq, mmap = True, map_location=torch.device('cpu'), weights_only=False)
        
        
        
        
        if mms.shape[-1] > lang_ref_length and melvq.shape[-1] > lang_ref_length * 2:
            randidx = random.randint(0, mms.shape[-1] - lang_ref_length)
            if self.debug:
                print(f"lang mms: {randidx / mms.shape[-1]} -> {(randidx + lang_ref_length) / mms.shape[-1]}")
                print(f"lang mel: {randidx * 2 / melvq.shape[-1]} -> {(randidx + lang_ref_length) * 2 / melvq.shape[-1]}")
                print(f"language_ref length: {lang_ref_length}")
                
            mms = mms[:, randidx: randidx + lang_ref_length]
            melvq = melvq[:, :, randidx * 2: (randidx + lang_ref_length) * 2]

        else:
            mq, l = mms.shape
            g, q, _ = melvq.shape
            l = min(l, lang_ref_length)
            
            if self.debug:
                print(f"lang mms: 0 -> {l / mms.shape[-1]}")
                print(f"lang mel: 0 -> {l * 2 / melvq.shape[-1]}")
                print(f"language_ref length: {lang_ref_length}")
            
            mms_paded = torch.LongTensor(mq, lang_ref_length)
            mms_paded.fill_(self.optional['mms_pad_idx'])
            mms_paded[:, :l] = mms[:, :l]
            
            melvq_paded = torch.LongTensor(g, q, lang_ref_length * 2)
            melvq_paded.fill_(self.optional['mel_vq_pad_idx'])
            melvq_paded[:, :, :l * 2] = melvq[:, :, :l * 2]
            
            mms = mms_paded
            melvq = melvq_paded
            
            
            
        return {"mms_language_ref": mms,
                "melvq_language_ref": melvq}
            
    
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
        elif '/cv_16/ita' in audiopath:
            language_name = 'Italian'
            lang_id =  self.language_code[language_name]
        elif '/vp_ita' in audiopath:
            language_name = 'Italian'
            lang_id =  self.language_code[language_name]
        elif '/magic_cn' in audiopath:
            language_name = 'Mandarin Chinese'
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
    
    def get_codec_mel_spec(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        audio, sampling_rate = librosa.load(audiopath, sr=16000)
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        
        max_audio_start = audio.size(1) - 16000
        audio_start = random.randint(0, max_audio_start)
        audio = audio[:, audio_start:audio_start + 16000]
        
        mel = codec_mel_spectrogram(
            audio,
            n_fft = 1024,
            num_mels = 80,
            sampling_rate = 16000,
            hop_size = 160,
            win_size = 640,
            fmin = 0,
            fmax = 8000,
            center=False)


        mel_loss = codec_mel_spectrogram(
            audio,
            n_fft = 1024,
            num_mels = 80,
            sampling_rate = 16000,
            hop_size = 160,
            win_size = 640,
            fmin = 0,
            fmax = None,
            center=False)


        return {"wave_audio": audio, "mel_spec": mel, "mel_loss_spec": mel_loss}

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
        
        if 'mms_44_postfix' not in self.optional.keys():
            mms_44_postfix = ".44"
        else:
            mms_44_postfix = self.optional['mms_44_postfix']
        
        mms_file = audiopath.replace(
            "/wave/", "/feature/mms/").replace(".flac", mms_44_postfix)
        data = torch.load(mms_file, mmap=True, weights_only=False)
        
        if 'mms_repeate' in self.optional.keys():
            data = data.repeat_interleave(2, dim=1)
            seg_length = self.optional['mms_seg_size'] * 2
        else:
            seg_length = self.optional['mms_seg_size']
            
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
        
        if len(data.shape) == 3:
            data = rearrange(data, 'g l q -> (q g) l')
        
        return {"mms_rvq_code": data}
    
    def get_length_dummy(self, audiopath_sid_text):
        return {"get_length_dummy": "_"}
    
    def get_mms_rvq_code_pad_seg(self, audiopath_sid_text):
        return self.get_mms_rvq_code(audiopath_sid_text)
    
    def get_double_mms_rvq_code(self, audiopath_sid_text):
        
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        mms_file = audiopath.replace(
            "/wave/", "/feature/mms/").replace(".flac", self.optional['mms_rvq_code_postfix'])
        data = torch.load(mms_file, mmap=True, map_location='cpu', weights_only=False)
        
        if self.optional['mms_rvq_code_postfix'] not in ['.20c']:
            data = rearrange(data, 'g q l -> (q g) l')
        
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
    
    def get_double_mms_hid(self, audiopath_sid_text):
        
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        mms_file = audiopath.replace(
            "/wave/", "/feature/mms/").replace(".flac", '.h')
        data = torch.load(mms_file, mmap=True, map_location='cpu', weights_only=False)
        
        seg_length = self.optional['mms_seg_size']
        
        if (data.shape[0] > seg_length):
            rand_idx = random.randint(0, data.shape[0] - seg_length - 1)
            self.temp_arg = rand_idx
            if self.debug:
                temp_l = data.shape[0]
                print(f'mms code: {rand_idx / temp_l} -> {(rand_idx+seg_length) / temp_l} of {temp_l}')
            # print(f"MMS_{audiopath_sid_text[0]}_{data.shape}_{rand_idx}_{rand_idx+seg_length}")
            data = data[rand_idx: rand_idx+seg_length, :]
        else:
            self.temp_arg = 0
            if self.debug:
                temp_l = data.shape[0]
                print(f'mms code: 0 -> 1.0 of {temp_l}')
        
        data = rearrange(data, 'l h -> h l')
        data = data.repeat_interleave(2, dim=-1)
        return {"mms_hid": data} 
            
    
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

            data = data[:, rand_idx: rand_idx+seg_length]

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
        
    def get_des_code(self, audiopath_sid_text):
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/des_code/",
                ".flac": ".s"
                # ".flac": ".align.code"
            },
            return_key="des_code",
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
        if "mel_spec_suffix" in self.optional.keys():
            mel_spec_suffix = self.optional['mel_spec_suffix']
        else:
            mel_spec_suffix = ".mel"
        return self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/mel_spec/",
                ".flac": mel_spec_suffix
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
        
    def get_nar_random_stage(self, audiopath_sid_text):
        return {}
    
    def get_nar_feature(self, audiopath_sid_text):
        mms_file = audiopath_sid_text[0].replace(
            "/wave/", "/feature/mms/").replace(".flac", self.optional['mms_rvq_code_postfix'])
        mms_data = torch.load(mms_file, mmap=True, map_location='cpu', weights_only=False)
        
        mel_vq_file = audiopath_sid_text[0].replace(
            "/wave/", "/feature/melvq/").replace(".flac", self.optional['melvq_code_postfix'])
        mel_vq_data = torch.load(mel_vq_file, mmap=True, map_location='cpu', weights_only=False)
        
        mms_data = rearrange(mms_data, 'g q l -> (g q) l')
        
        return {"mms_rvq_code": mms_data,
                "mel_grvq_code": mel_vq_data,
                } 
        
    def get_mms_with_mel_vq(self, audiopath_sid_text):
        mms_data = self.get_mms_rvq_code(audiopath_sid_text)['mms_rvq_code']
        mms_data = mms_data.repeat_interleave(2, dim=-1)
        mel_vq_file = audiopath_sid_text[0].replace(
            "/wave/", "/feature/melvq/").replace(".flac", self.optional['melvq_code_postfix'])
        mel_vq_data = torch.load(mel_vq_file, mmap=True, map_location='cpu', weights_only=False)
        
        if self.debug:
            print(f"Shape of mms: {mms_data.shape}| Shape of melvq: {mel_vq_data.shape}")

        l = min(mel_vq_data.shape[-1], mms_data.shape[-1])
        mms_data = mms_data[:, :l]
        mel_vq_data = mel_vq_data[:, :, :l]
        
        return {"mms_rvq_code": mms_data,
                "mel_grvq_code": mel_vq_data,
                } 
        
    def get_mms_with_mel_vq_old(self, audiopath_sid_text):
        
        mms_seg_length = self.optional['mms_seg_size']
        mel_vq_seg_length = mms_seg_length * 2
        mms_pad_idx = self.optional['mms_pad_idx']
        mel_vq_pad_idx = self.optional['mel_vq_pad_idx']
        mms_length = 0
        mel_vq_length = 0
        
        
        mms_seg = torch.LongTensor(8, mms_seg_length)
        mms_seg.fill_(mms_pad_idx)
        
        mel_vq_seg = torch.LongTensor(4, 8, mel_vq_seg_length)
        mel_vq_seg.fill_(mel_vq_pad_idx)
        
        mms_data = self.get_mms_rvq_code(audiopath_sid_text)['mms_rvq_code']
    
        if (mms_data.shape[-1] > mms_seg_length + 5):
            rand_idx = random.randint(1, mms_data.shape[-1] - mms_seg_length - 1)
            if self.debug:
                print(f'mms code: {rand_idx / mms_data.shape[-1]} -> {(rand_idx+mms_seg_length) / mms_data.shape[-1]} of {mms_data.shape[-1]}')
            # print(f"MMS_{audiopath_sid_text[0]}_{data.shape}_{rand_idx}_{rand_idx+seg_length}")
            mms_seg = mms_data[:, rand_idx: rand_idx+mms_seg_length]
            mms_length = mms_seg_length
        else:
            rand_idx = 0
            mms_length = min(mms_data.shape[-1], mms_seg_length)
            target_mel_vq_length = mms_length * 2
            mms_seg[:, :mms_length] = mms_data[:, :mms_length]
            if self.debug:
                print(f'mms code: 0 -> 1.0 of {mms_data.shape[-1]}')
        
        mms_seg = mms_seg.repeat_interleave(2, dim=-1)
        mms_length *= 2
        target_mel_vq_length = mms_length
        rand_idx *= 2
        
        mel_vq_file = audiopath_sid_text[0].replace(
            "/wave/", "/feature/melvq/").replace(".flac", self.optional['melvq_code_postfix'])
        mel_vq_data = torch.load(mel_vq_file, mmap=True, map_location='cpu', weights_only=False)
        if rand_idx == 0:
            mel_vq_length = target_mel_vq_length
            mel_vq_seg[:, :, :target_mel_vq_length] = mel_vq_data[:, :, :target_mel_vq_length]
            if self.debug:
                print(f'melvq code: 0 -> 1.0 of {mel_vq_data.shape[-1]}')
        else:
            mel_vq_length = mel_vq_seg_length
            mel_vq_seg = mel_vq_data[:, :, rand_idx: rand_idx+mel_vq_seg_length]
            if self.debug:
                print(f'melvq code: {rand_idx / mel_vq_data.shape[-1]} -> {(rand_idx+mel_vq_seg_length) / mel_vq_data.shape[-1]} of {mel_vq_data.shape[-1]}')
            

        return {"mms_rvq_code": mms_seg,
                "mms_rvq_len": mms_length,
                "mel_grvq_code": mel_vq_seg,
                "mel_grvq_len": mel_vq_length,
                } 
        
        
    def get_mms_corr_mel_160(self, audiopath_sid_text):
        if "mel_spec_suffix" not in self.optional.keys():
            mel_spec_suffix = ".160"
        else:
            mel_spec_suffix = self.optional["mel_spec_suffix"]
            
        data = self.torch_load_single(
            audiopath_sid_text,
            path_replaecments={
                "/wave/": "/feature/mel_spec/",
                ".flac": mel_spec_suffix
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
    
    def get_melvq_ref(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        assert len(self.ref_dict[spk]) > 0, spk

        ref = random.choice(self.ref_dict[spk])
        if self.debug:
            print(f"Speaker Ref : {ref}")
        spec_filename = ref.replace(
            "/wave/", "/feature/melvq/").replace(".flac", self.optional['melvq_code_postfix'])
        assert os.path.exists(spec_filename), spec_filename

        spec = torch.load(spec_filename, map_location='cpu', weights_only=False).to(torch.long)
        
        current_length = spec.shape[-1]
        target_length = self.optional['mel_vq_ref_length']

        if current_length > target_length:
            # Randomly select a start point for cropping
            start_idx = random.randint(0, current_length - target_length)
            adjusted_spec = spec[:, :, start_idx:start_idx + target_length]
        
        elif current_length < target_length:
            # Repeat the tensor along the last dimension to match the target length
            repeat_factor = (target_length + current_length - 1) // current_length  # Calculate how many times to repeat
            repeated_spec = spec.repeat(1, 1, repeat_factor)  # Repeat along the last dimension
            adjusted_spec = repeated_spec[:, :, :target_length]  # Ensure it is exactly target_length
        
        else:
            # No adjustment needed if lengths are equal
            adjusted_spec = spec
        
        # Ensure final spec has the correct length
        assert adjusted_spec.shape[-1] == target_length, f"Expected {target_length}, but got {adjusted_spec.shape[-1]}"
        
        return {"melvq_ref": adjusted_spec}

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

        if "mel_spec_suffix" in self.optional.keys():
            mel_spec_suffix = self.optional['mel_spec_suffix']
        else:
            mel_spec_suffix = ".mel"
        ref = random.choice(self.ref_dict[spk])
        if self.debug:
            print(f"Spk Ref : {ref}")
        spec_filename = ref.replace(
            "/wave/", "/feature/mel_spec/").replace(".flac", mel_spec_suffix)
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
        ori_text = self.ori_text_dict[audiopath]
        text_norm = self.frontend.textonly_to_idx(ori_text)
        return {"text": text_norm}
    
    def get_custom_text(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        text_norm = self._text_to_idx(ori_text)
        return {"text": text_norm}

    def get_phoneme(self, audiopath_sid_text):
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        text = self.text_dict[audiopath]
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

    def __init__(self, config=None, optional={}, return_ids=False, debug=False):

        self.return_ids = return_ids
        self.data_list = config.data.data_list
        self.optional = optional
        self.warn_msg = set()
        self.debug = debug

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
        
    def collect_des_code(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="des_code",
            feature_dtype=torch.long,
            pad_value = 1024
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
        
    def collect_double_mms_hid(self, batch, ids_sorted_decreasing):
        return self.collect_2D_with_length(
            batch,
            ids_sorted_decreasing,
            feature_key="mms_hid",
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
        

    def collect_length_dummy(self, batch, ids_sorted_decreasing):
        return {}

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

    def collect_nar_random_stage(self, batch, ids_sorted_decreasing):
        try:
            stages = from_pickle(self.optional['nar_stage_list'])
            rand_stage = random.choice(stages)
        except:
            rand_stage = 0
            
        return {"nar_random_stage": rand_stage}
    
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
    
    def collect_nar_language_ref(self, batch, ids_sorted_decreasing):
        g, q, l = batch[0]["melvq_language_ref"].shape

        melvq_ref_padded = torch.LongTensor(
            len(batch), g, q, l)
        melvq_ref_padded.fill_(self.optional['mel_vq_pad_idx'])

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            ref = row["melvq_language_ref"]
            melvq_ref_padded[i, :, :] = ref
            
            
        q, l = batch[0]["mms_language_ref"].shape

        mms_ref_padded = torch.LongTensor(
            len(batch), q, l)
        mms_ref_padded.fill_(self.optional['mms_pad_idx'])

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            ref = row["mms_language_ref"]
            mms_ref_padded[i, :] = ref

        return {"melvq_language_ref": melvq_ref_padded,
                "mms_language_ref": mms_ref_padded,
                "mms_language_ref_len": mms_ref_padded.shape[-1]}
    
    
    def collect_melvq_ref(self, batch, ids_sorted_decreasing):
        max_ref_len = max([x["melvq_ref"].shape[-1] for x in batch])
        ref_padded = torch.LongTensor(
            len(batch), 4, self.optional['mel_vq_quan_size'], max_ref_len)
        ref_padded.fill_(self.optional['mel_vq_pad_idx'])

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            ref = row["melvq_ref"]
            ref_padded[i, :, :, :ref.shape[-1]] = ref

        return {"melvq_ref": ref_padded,
                "melvq_ref_length": max_ref_len}

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
        
    def collect_nar_feature(self, batch, ids_sorted_decreasing):
        seg_limit = self.optional['seg_limit']
        
        mms_min_len = min([x['mms_rvq_code'].shape[-1] for x in batch])
        mel_vq_min_len = min([x['mel_grvq_code'].shape[-1] for x in batch])
        
        mms_padded = torch.LongTensor(len(batch), 8, seg_limit)
        mms_padded.fill_(self.optional['mms_pad_idx'])
        mel_vq_padded = torch.LongTensor(len(batch), 4, self.optional['mel_vq_quan_size'], seg_limit * 2)
        mel_vq_padded.fill_(self.optional['mel_vq_pad_idx'])
        
        if mms_min_len < seg_limit:
            mms_start_index, mel_start_index = -1, -1
        else:
            mms_start_index = random.randint(0, mms_min_len - self.optional['seg_limit'])
            mms_end_index = mms_start_index + self.optional['seg_limit']
            
            mel_start_index = mms_start_index * 2
            mel_end_index = mel_start_index + self.optional['seg_limit'] * 2
        
        if self.debug:
            if mms_start_index == -1:
                print(f"MMS/MEL: 0 -> {seg_limit}")
            else:
                print(f"MMS: {mms_start_index / mms_min_len} -> {mms_end_index / mms_min_len}")
                print(f"MEL: {mel_start_index / mel_vq_min_len} -> {mel_end_index / mel_vq_min_len}")
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            mms = row['mms_rvq_code']
            melvq = row['mel_grvq_code']
            
            if mms_start_index == -1:
                temp_l = min(seg_limit, mms.shape[-1])
                mms_padded[i, :, :temp_l] = mms[:, :temp_l]
                mel_vq_padded[i, :, :, :temp_l * 2] = melvq[:, :, :temp_l * 2]
            else:
                mms_padded[i, :, :] = mms[:, mms_start_index: mms_end_index]
                mel_vq_padded[i, :, :, :] = melvq[:, :, mel_start_index: mel_end_index]
        
        mms_start_index = max(0, mms_start_index)
        mel_start_index = max(0, mel_start_index)
        
        return {"mms_rvq_code": mms_padded,
                "mel_grvq_code": mel_vq_padded,
                "mms_start_index": mms_start_index,
                "mel_start_index": mel_start_index}
        
        
        
    def collect_mms_with_mel_vq(self, batch, ids_sorted_decreasing):
        max_len = max([x['mms_rvq_code'].shape[-1] for x in batch])
        len_limit = self.optional['seg_limit']
        
        mms_lengths = torch.LongTensor(len(batch))
        mms_lengths.zero_()
        mms_padded = torch.LongTensor(len(batch), 8, max_len)
        mms_padded.fill_(self.optional['mms_pad_idx'])
        
        mel_vq_lengths = torch.LongTensor(len(batch))
        mel_vq_lengths.zero_()
        mel_vq_padded = torch.LongTensor(len(batch), 4, self.optional['mel_vq_quan_size'], max_len)
        mel_vq_padded.fill_(self.optional['mel_vq_pad_idx'])

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            mms = row['mms_rvq_code']
            melvq = row['mel_grvq_code']
            
            mms_padded[i, :, :mms.size(-1)] = mms
            mel_vq_padded[i, :, :, :mms.size(-1)] = melvq[:, :, :mms.size(-1)]
            
            mms_lengths[i] = min(mms.size(-1), len_limit)
            mel_vq_lengths[i] = mms_lengths[i]
        
        if max_len > len_limit:
            mms_padded = mms_padded[:, :, :len_limit].clone()
            mel_vq_padded = mel_vq_padded[:, :, :, :len_limit].clone()

        return {"mms_rvq_code": mms_padded,
                "mel_grvq_code": mel_vq_padded,
                "mms_rvq_length": mms_lengths,
                "mel_grvq_length": mel_vq_lengths}
        
    # def collect_mms_with_mel_vq_old(self, batch, ids_sorted_decreasing):
        
    #     mms_seg = torch.stack([i['mms_rvq_code'] for i in batch]).to(torch.long)
    #     mel_vq_seg = torch.stack([i['mel_grvq_code'] for i in batch]).to(torch.long)
    #     mms_length = torch.LongTensor([i['mms_rvq_len'] for i in batch])
    #     mel_vq_length = torch.LongTensor([i['mel_grvq_len'] for i in batch])
        
    #     max_len = mel_vq_length.max()
    #     mms_seg = mms_seg[:, :, :max_len]
    #     mel_vq_seg = mel_vq_seg[:, :, :, :max_len]
        
    #     return {"mms_rvq_code": mms_seg,
    #     "mms_rvq_len": mms_length,
    #     "mel_grvq_code": mel_vq_seg,
    #     "mel_grvq_len": mel_vq_length,
    #     } 
        
        
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
        
    # def collect_club(self, batch, ids_sorted_decreasing):
    #     features = torch.stack([batch[ids_sorted_decreasing[i]]['club'] for i in range(len(ids_sorted_decreasing))])
    #     features = features.reshape(features.shape[0] * features.shape[1], features.shape[2], features.shape[3])
    #     return {'club': features}

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
    
    def collect_codec_mel_spec(self, batch, ids_sorted_decreasing):
        wave_padded = torch.FloatTensor(len(batch), 16000)
        mel_padded = torch.FloatTensor(len(batch), 80, 100)
        mel_loss_padded = torch.FloatTensor(len(batch), 80, 100)

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            mel_padded[i, :, :] = row['mel_spec']
            mel_loss_padded[i, :, :] = row['mel_loss_spec']
            wave_padded[i, :] = row['wave_audio']

        return {"wave_audio": wave_padded,
                "mel_spec": mel_padded,
                "mel_loss_spec": mel_loss_padded}

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