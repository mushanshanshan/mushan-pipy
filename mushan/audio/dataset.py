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
        
        self.audio_type = ''
        for d in config.data.data_list:
            if 'audio' in d:
                if self.audio_type != "": raise TypeError
                self.audio_type = d
        
        self.spec_type = ''
        for d in config.data.data_list:
            if 'spec' in d:
                if self.spec_type != "": raise TypeError
                self.spec_type = d
        
        self.ref_type = '' 
        
        for d in config.data.data_list:
            if 'ref' in d:
                if self.ref_type != "": raise TypeError
                self.ref_type = d
            
        self.need_text = 'text' in config.data.data_list
        self.need_vc = 'vc' in config.data.data_list
        self.need_f0 = 'f0' in config.data.data_list
        
        self.aux_type = ''
        for d in config.data.data_list:
            if 'aux' in d:
                if self.aux_type != "": raise TypeError
                self.aux_type = d
                
        self.concat = (config.data.concat == True)
            
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


        if 'norm' in self.spec_type:
            if 'mel' in self.spec_type:
                self.spec_norm = from_pickle("/home/mushan/data/feature/mel_spec/norm.pk")
                if self.rank == 0:
                    logger.info(f"Using mel spec normalization.")
            elif 'codec' in self.spec_type:
                self.spec_norm = from_pickle("/home/mushan/data/feature/codec/norm.pk")
                if self.rank == 0:
                    logger.info(f"Using codec normalization.")
            elif 'c-codec' in self.spec_type:
                self.spec_norm = from_pickle("/home/mushan/data/feature/codec/c-norm.pk")
                if self.rank == 0:
                    logger.info(f"Using codec normalization.")
                
                
        if 'norm' in self.ref_type:
            if 'mel' in self.ref_type:
                self.ref_norm = from_pickle("/home/mushan/data/feature/mel_spec/norm.pk")
                if self.rank == 0:
                    logger.info(f"Using ref mel spec normalization.")
            elif 'codec' in self.ref_type:
                self.ref_norm = from_pickle("/home/mushan/data/feature/codec/norm.pk")
                if self.rank == 0:
                    logger.info(f"Using ref codec normalization.")
            elif 'c-codec' in self.spec_type:
                self.spec_norm = from_pickle("/home/mushan/data/feature/codec/c-norm.pk")
                if self.rank == 0:
                    logger.info(f"Using res codec normalization.")
        
        
        if self.concat:
            self._concat_filter()
        else:
            self._filter()
            
        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        
    def _concat_filter(self, min = 7, max = 10):
        
        def create_balanced_sublists(input_data, min = 7, max = 10):

            random.shuffle(input_data)
            temp_data = [[[],0] for i in range(len(input_data))]
            output_data = []

            for item in input_data:
                path, length = item

                for i in range(len(temp_data)):
                    if temp_data[i][1] + length < max:
                        temp_data[i][0].append(path)
                        temp_data[i][1] += length
                        break

            for t in temp_data:
                if t[1] > min:
                    output_data.append(t)
            return output_data
        
        concat_audiopaths_sid_text_new = []
        concat_audio_lengths = []
        concat_text_lengths = []
        missing_file = []
        audiopaths_sid_text_new = []
        speaker_dict = defaultdict(list)
        pho_dict = {}
        counr_after_concat = 0

        for i in tqdm(range(len(self.audiopaths_sid_text)), disable=self.rank != 0):
            audiopath, spk, dur, ori_text, pho = self.audiopaths_sid_text[i]
            dur = float(dur)
            if dur > self.min_audio_len:
                self.ref_dict[spk].append(audiopath)
            speaker_dict[spk].append([audiopath, dur])
            pho_dict[audiopath] = pho[:-1]
        
        if self.rank == 0 and len(missing_file) > 0:
            logger.error(f"Missing data index: {missing_file}")
            
        for spk, v in speaker_dict.items():
            _cat = create_balanced_sublists(v, min, max)
            for _c in _cat:
                concat_audiopaths_sid_text_new.append([_c[0], spk, _c[1], "", "$".join([pho_dict[i][:-1] if pho_dict[i][-1] == '.' else pho_dict[i] for i in _c[0]])])
                counr_after_concat += len(_c[0])
                
        if self.rank == 0:
            logger.info(f"Using speech concat:")
            logger.info(f"Total speech {len(self.audiopaths_sid_text)}, After coant: {counr_after_concat}")
            
        self.audiopaths_sid_text = concat_audiopaths_sid_text_new
        self.audio_lengths = [i[2] for i in concat_audiopaths_sid_text_new]
        self.text_lengths = [len(i[-1]) for i in concat_audiopaths_sid_text_new]
        
        if self.rank == 0:
            logger.info(f"Avaliable data length: {len(self.audiopaths_sid_text)}")
        
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
            
    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, spk, dur, ori_text, text = audiopath_sid_text
        #print(audiopath)
        
        if self.audio_type != '':
            wav = self.get_audio(audiopath)
        else:
            wav = torch.zeros(1,5)
            
        if self.audio_type == "norm_audio":
            print('norm')
            wav = self.wave_norm(wav)
        
        if "clip_wave" in self.optinal.keys():
            wav, _ = self.clip_spec(wav, self.optinal['clip_wave'])
            
        if 'mel_spec' in self.spec_type:
            if self.concat:
                spec = self.cat_data(audiopath, self.get_mel_spec)
            else:
                spec = self.get_mel_spec(audiopath)
        elif 'linear_spec' in self.spec_type:
            spec = self.get_linear_spec(audiopath)
        elif  'q_codec_spec' in self.spec_type:
            if self.concat:
                spec = self.cat_data(audiopath, self.get_q_codec)
            else:
                spec = self.get_q_codec(audiopath)
        elif 'c_codec_spec' in self.spec_type:
            if self.concat:
                spec = self.cat_data(audiopath, self.get_q_codec)
            else:
                spec = self.get_c_codec(audiopath)
        elif 'mel_bigv_online_spec' in self.spec_type:    
            spec = self.get_bigv_mel(wav)
        elif self.spec_type == '':
            spec = torch.zeros(1,1)
        else:
            raise NotImplemented
        
        if 'norm' in self.spec_type:
            spec = self.spec_norm.normalize(spec) 
            
        if "clip_spec" in self.optinal.keys():
            spec, tag = self.clip_spec(spec, self.optinal['clip_spec'])
            if not tag:
                logger.warning(f"Clip error: {audiopath}")

        
        if self.need_text:
            text = self.get_text(text)
        else:
            text = torch.zeros(1)
        
        if self.ref_type != "":
            ref_path = self.get_ref_path(spk)
            if ref_path == None: 
                ref_path = audiopath
            if 'mel_ref' in self.ref_type:
                ref = self.get_mel_spec(ref_path)
            elif 'q_codec_ref' in self.ref_type:
                ref = self.get_q_codec(ref_path)
            elif 'c_codec_ref' in self.ref_type:
                ref = self.get_c_codec(ref_path)
            elif 'copy_ref' in self.ref_type:
                ref = spec.clone()
            else:
                raise NotImplemented
        else:
            ref = torch.zeros((1,1))
        
        if 'copy_ref' not in self.ref_type:
            ref = self.clip_ref(ref)
            
            
        if 'norm' in self.ref_type:
            ref = self.ref_norm.normalize(ref) 
            
        if self.need_f0:
            f0 = self.get_f0(audiopath)
            assert spec.shape[-1] == f0.shape[-1], f"spec.shape={spec.shape}, f0.shape={f0.shape}"
        else:
            f0 = torch.zeros(1)
            
        if self.aux_type == 'hu_aux':
            aux = self.get_hu(audiopath)
            assert spec.shape[-1] == aux.shape[-1], f"spec.shape={spec.shape}, aux.shape={aux.shape}"
        elif self.aux_type == 'whisper_aux':
            if self.concat:
                aux = self.cat_data(audiopath, self.get_whisper)
            else:
                aux = self.get_whisper(audiopath)
            target_len = spec.shape[-1]
            
            # Align with spectrogram
            aux = torch.nn.functional.interpolate(aux.unsqueeze(0), 
                                                  size=target_len, 
                                                  mode='linear').squeeze(0)
            assert spec.shape[-1] == aux.shape[-1], f"spec.shape={spec.shape}, aux.shape={aux.shape}"
        else:
            aux = torch.zeros(1,1)
            
        if "random_clip_spec_aux" in self.optinal.keys():
            spec, aux = self.random_clip([spec, aux], self.optinal["random_clip_spec_aux"])
            
        if self.debug:
            print(f"{audiopath}|{spk}|{ref_path}|{ori_text}")
            
        return (text, spec, wav, ref, f0, aux)
    
    def wave_norm(self, wave):
        wave -= wave.min()
        wave /= wave.max()
        return (2 * wave - 1) * 0.95
    
    def random_clip(self, inputs, target_length):
        in_length = inputs[0].shape[-1]
        re = []
        for i in inputs:
            assert i.shape[-1] == in_length
            
        start_idx = random.randint(0, in_length - target_length - 1)
        for i in inputs:
            re.append(i[:, start_idx: start_idx + target_length].clone())
        
        return re
    
    
    def clip_ref(self, ref):
        if ref.shape[-1] > 500:
            start_idx = random.randint(0, ref.shape[1] - 500 - 1)
            ref = ref[:, start_idx: start_idx + 500].clone()
            return ref
        return ref
    
    def get_bigv_mel(self, wave):
        return bigv_mel(wave)
    
    def feature_normalization(self, feature, min_value, max_value):
        feature = torch.clamp(feature, 
                              min=min_value, 
                              max=max_value)
        feature = (feature - min_value) / (max_value - min_value) * 2 - 1
        return feature
        
    def cat_data(self, filenames, fn):
        feature = None
        for file in filenames:
            data = fn(file)
            
            if feature == None:
                feature = data
            else:
                feature = torch.cat((feature, data), dim = -1)
        return feature

    def get_ref_path(self, spk):
        try:
            ref = random.choice(self.ref_dict[spk])
        except Exception as e:
            logger.error(f"Ref Error: {spk}, {e}")
            print((f"Ref Error: {spk}, {e}"))
            return None
        return ref
    
    
    def clip_spec(self, spec, length):
        if spec.shape[1] < length:
            p = torch.zeros(spec.shape[0], length, dtype = spec.dtype, device = spec.device)
            p[:, :spec.shape[1]] = spec
            return p, False
        else:
            start_idx = random.randint(0, spec.shape[1] - length - 1)
            spec = spec[:, start_idx: start_idx + length].clone()
            return spec, True
    
    def min_clip_spec(self, spec):
        target_len = int(spec.shape[1] / 93.75) * 90
        start_idx = random.randint(0, spec.shape[1] - target_len - 1)
        spec = spec[:, start_idx: start_idx + target_len].clone()
        return spec
    
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
    
    def get_hu(self, filename):
        aux = torch.load(filename.replace("/wave/", "/feature/hubert/").replace(".flac", ".hubert"))
        return aux
    
    def get_whisper(self, filename):
        aux = torch.load(filename.replace("/wave/", "/feature/whisper/").replace(".flac", ".enc"))
        return aux
    
    def get_audio(self, filename):
        audio_norm, sampling_rate = torchaudio.load(filename)
        return audio_norm
    
    def get_q_codec(self, filename):
        spec_filename = filename.replace("/wave/", "/feature/codec/").replace(".flac", f".qemb")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename, map_location='cpu')
            assert spec.shape[0] == 128
        else:
            raise FileNotFoundError
        return spec
    
    def get_c_codec(self, filename):
        spec_filename = filename.replace("/wave/", "/feature/codec/").replace(".flac", f".cemb")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename, map_location='cpu')
            assert spec.shape[0] == 128
        else:
            raise FileNotFoundError
        return spec
    
    def get_linear_spec(self, filename):
        spec_filename = filename.replace("/wave/", "/feature/linear_spec/").replace(".flac", f".{self.spec_suffix}")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename, map_location='cpu')
            assert spec.shape[0] == 513
        else:
            raise FileNotFoundError
        return spec
    
    def get_mel_spec(self, filename):
        spec_filename = filename.replace("/wave/", "/feature/mel_spec/").replace(".flac", f".mel")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename, map_location='cpu')
            assert spec.shape[0] == 100
        else:
            raise FileNotFoundError
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

    def __init__(self, config=None, optional = {}, return_ids=False, tacotron=False):
        
        self.return_ids = return_ids
        self.optional = optional
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
        
        if "fix_len" in self.optional.keys():
            _len = self.optional["fix_len"]
            max_spec_len = _len * (max_spec_len // _len + 1)

        max_wav_len = max([x[2].size(1) for x in batch])
        max_ref_len = max([x[3].size(1) for x in batch])

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
            

        aux_padded = torch.FloatTensor(len(batch), batch[0][5].size(0), max_spec_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        ref_padded.zero_()
        f0_padded.zero_()
        aux_padded.zero_()
        
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
            
            aux = row[5]

            aux_padded[i, :, :aux.size(1)] = aux
            


        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ref_padded, ref_lengths, f0_padded, aux_padded


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
