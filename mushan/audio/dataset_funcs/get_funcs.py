from glob import glob
import librosa
from tqdm import tqdm
import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio
import math
import copy
import pickle
import lmdb
import itertools
from collections import defaultdict
from loguru import logger
from mushan.io import from_pickle, to_pickle, hashabledict, from_audio
from mushan.text.eng.front_end import Frontend as ENFrontend
from mushan.text.chs.front_end import Frontend as CNFrontend
from mushan.models.bigv.utils import bigv_mel
from mushan.audio.hifi_mel import mel_spectrogram as hifi_mel_spectrogram
from mushan.audio.lang_info import *
from mushan.audio.codec_mel import mel_spectrogram as codec_mel_spectrogram
from librosa.util import normalize
from einops import rearrange, repeat, reduce
from transformers import AutoTokenizer

def get_file_extension(file_path):
    _, ext = os.path.splitext(file_path)
    return ext

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
    
    mms = pt.replace("/wave/", "/feature/mms/").replace(get_file_extension(pt), self.optional['mms_rvq_code_postfix'])
    mms = torch.load(mms, mmap = True, map_location=torch.device('cpu'), weights_only=False)
    mms = rearrange(mms, 'g l q -> (q g) l')
    
    
    melvq = pt.replace("/wave/", "/feature/melvq/").replace(get_file_extension(pt), self.optional['melvq_code_postfix'])
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
    
    mms = pt.replace("/wave/", "/feature/mms/").replace(get_file_extension(pt), ".l.44.norm")
    mms = torch.load(mms, mmap = True, map_location=torch.device('cpu'), weights_only=False)
    
    mel = pt.replace("/wave/", "/feature/mel_spec/").replace(get_file_extension(pt), ".160")
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
    
    if '/libri' in audiopath or '/ljspeech' in audiopath:
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

def get_speaker_index(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    return {"speaker_index": self.spk_index[spk]}

def get_audio_path(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text

    return {"audio_path": audiopath.replace(f'/home/{self.username}/data/wave/', '')}

def get_hubert(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    hu_filename = audiopath.replace(
        "/wave/", "/feature/hubert/").replace(get_file_extension(pt), ".code")
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
        "/wave/", "/feature/xlsr2b/").replace(get_file_extension(pt), post_fix)
    seg_length = self.optional['xlsr_seg_size']

    data = torch.load(mms_file, mmap=True, map_location=torch.device('cpu'), weights_only=False)
    # data = torch.load(mms_file, mmap=False)
    rand_idx = random.randint(0, data.shape[-1] - seg_length)
    data = data[:, rand_idx: rand_idx+seg_length]
    return {"xlsr2b_feature_48": data}

def get_mms_feature_48(self, audiopath_sid_text, post_fix=".48"):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    mms_file = audiopath.replace(
        "/wave/", "/feature/mms/").replace(get_file_extension(pt), post_fix)
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
        "/wave/", "/feature/mms/").replace(get_file_extension(pt), mms_44_postfix)
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
        "/wave/", "/feature/mms/").replace(get_file_extension(pt), self.optional['mms_rvq_code_postfix'])
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
        "/wave/", "/feature/mms/").replace(get_file_extension(pt), self.optional['mms_rvq_code_postfix'])
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
        "/wave/", "/feature/mms/").replace(get_file_extension(pt), '.h')
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
    audiopath, _, _, _, _ = audiopath_sid_text
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/mhubert/",
            get_file_extension(audiopath): ".code"
        },
        return_key="mhubert_code",
    )
    

def get_mms_code(self, audiopath_sid_text):
    audiopath, _, _, _, _ = audiopath_sid_text
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/mms/",
            get_file_extension(audiopath): self.optional['mms_code_postfix']
        },
        return_key="mms_code",
    )
    
def get_des_code(self, audiopath_sid_text):
    audiopath, _, _, _, _ = audiopath_sid_text
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/des_code/",
            get_file_extension(audiopath): ".s"
        },
        return_key="des_code",
    )
    
def get_mms_code_pad_seg(self, audiopath_sid_text):
    audiopath, _, _, _, _ = audiopath_sid_text
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/mms/",
            get_file_extension(audiopath): self.optional['mms_code_postfix']
        },
        return_key="mms_code",
    )

def get_mms_code_seg(self, audiopath_sid_text):
    audiopath, _, _, _, _ = audiopath_sid_text
    data = self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/mms/",
            get_file_extension(audiopath): self.optional['mms_code_postfix']
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
    audiopath, _, _, _, _ = audiopath_sid_text
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/hubert/",
            get_file_extension(audiopath): ".code"
        },
        return_key="hubert_code",
    )

def get_enhubert_code(self, audiopath_sid_text):
    audiopath, _, _, _, _ = audiopath_sid_text
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/hubert/",
            get_file_extension(audiopath): ".en.code"
        },
        return_key="hubert_code",
    )

def get_text_token(self, audiopath_sid_text):
    assert "text_token_suffix" in self.optional.keys()
    audiopath, _, _, _, _ = audiopath_sid_text
    seq = self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/text_token/",
            get_file_extension(audiopath): self.optional["text_token_suffix"]
        },
        return_key="text_token",
    )["text_token"]
    
    if "text_token_intersperse" in self.optional.keys():
        inter_seq = torch.ones(seq.shape[-1] * 2 + 1, dtype=seq.dtype, device=seq.device) * self.optional["text_token_intersperse"]
        inter_seq[1::2] = seq
        seq = inter_seq
        
    return {"text_token": seq}

def get_robert(self, audiopath_sid_text, layer=3):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    pho_filename = audiopath.replace(
        "/wave/", "/feature/robert/").replace(get_file_extension(pt), ".p2f")
    assert os.path.exists(pho_filename), pho_filename
    robert_filename = audiopath.replace(
        "/wave/", "/feature/robert/").replace(get_file_extension(pt), f".{layer}.pt")
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
    audiopath, _, _, _, _ = audiopath_sid_text
    
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/mel_spec/",
            get_file_extension(audiopath): mel_spec_suffix
        },
        return_key="mel_spec",
    )
    
def get_mel_with_dac(self, audiopath_sid_text):
    audiopath, _, _, _, _ = audiopath_sid_text
    wave, sr = torchaudio.load(audiopath)
    
    length = wave.shape[-1]
    right_pad = math.ceil(length / 512) * 512 - length
    wave = torch.nn.functional.pad(wave, (0, right_pad))
    
    wave = torchaudio.functional.resample(wave, sr, 44100, lowpass_filter_width=6)
    mel = bigv_mel(wave, n_fft = 2048, win_size = 2048, sampling_rate = 44100, hop_size = 512, fmax=22050)
    
    dac = self.get_dac_audio_token(audiopath_sid_text)['dac_audio_token']

    if mel.shape[-1] < dac.shape[-1]:
        pad_size = dac.shape[-1] - mel.shape[-1]
        mel = torch.nn.functional.pad(mel, (0, pad_size))
    elif mel.shape[-1] > dac.shape[-1]:
        mel = mel[..., :dac.shape[-1]]
    
    if "mel_spec_for_dac_mean" in self.optional.keys():
        mel = (mel - self.optional["mel_spec_for_dac_mean"]
                ) / self.optional["mel_spec_for_dac_std"]
    
    if self.debug:
        print(f"MEL: {mel.shape[-1]}| DAC: {dac.shape[-1]}")
    
    return {"mel_spec": mel,
            "dac_audio_token": dac}
    
def get_dac_audio_token(self, audiopath_sid_text):
    if "dac_audio_token_suffix" in self.optional.keys():
        dac_audio_token_suffix = self.optional['dac_audio_token_suffix']
    else:
        dac_audio_token_suffix = ".c"
    audiopath, _, _, _, _ = audiopath_sid_text
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/dac/",
            get_file_extension(audiopath): dac_audio_token_suffix
        },
        return_key="dac_audio_token",
    )
    
def get_dac_audio_emb(self, audiopath_sid_text):
    dac_token = self.get_dac_audio_token(audiopath_sid_text)['dac_audio_token'].long().unsqueeze(0)
    with torch.no_grad():
        dac_emb, _, _ = self.tokenizer['dac_tokenizer'].quantizer.from_codes(dac_token)
    dac_emb = dac_emb.squeeze(0)
    return {
        "dac_audio_emb": dac_emb,
    }
    
def get_delay_dac_audio_token(self, audiopath_sid_text):
    return self.get_dac_audio_token(audiopath_sid_text)

def get_mel_spec_160(self, audiopath_sid_text):
    audiopath, _, _, _, _ = audiopath_sid_text
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/mel_spec/",
            get_file_extension(audiopath): ".160"
        },
        return_key="mel_spec",
    )
    
def get_nar_random_stage(self, audiopath_sid_text):
    return {}

def get_nar_feature(self, audiopath_sid_text):
    mms_file = audiopath_sid_text[0].replace(
        "/wave/", "/feature/mms/").replace(get_file_extension(pt), self.optional['mms_rvq_code_postfix'])
    mms_data = torch.load(mms_file, mmap=True, map_location='cpu', weights_only=False)
    
    mel_vq_file = audiopath_sid_text[0].replace(
        "/wave/", "/feature/melvq/").replace(get_file_extension(pt), self.optional['melvq_code_postfix'])
    mel_vq_data = torch.load(mel_vq_file, mmap=True, map_location='cpu', weights_only=False)
    
    mms_data = rearrange(mms_data, 'g q l -> (g q) l')
    
    return {"mms_rvq_code": mms_data,
            "mel_grvq_code": mel_vq_data,
            } 
    
def get_mms_with_mel_vq(self, audiopath_sid_text):
    mms_data = self.get_mms_rvq_code(audiopath_sid_text)['mms_rvq_code']
    mms_data = mms_data.repeat_interleave(2, dim=-1)
    mel_vq_file = audiopath_sid_text[0].replace(
        "/wave/", "/feature/melvq/").replace(get_file_extension(pt), self.optional['melvq_code_postfix'])
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
        "/wave/", "/feature/melvq/").replace(get_file_extension(pt), self.optional['melvq_code_postfix'])
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
    audiopath, _, _, _, _ = audiopath_sid_text
    data = self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/mel_spec/",
            get_file_extension(audiopath): mel_spec_suffix
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
    audiopath, _, _, _, _ = audiopath_sid_text
    data = self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/mel_spec/",
            get_file_extension(audiopath): ".160"
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
        "/wave/", "/feature/melvq/").replace(get_file_extension(pt), self.optional['melvq_code_postfix'])
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
        
    if "mel_spec_suffix" in self.optional.keys():
        mel_spec_suffix = self.optional['mel_spec_suffix']
    else:
        mel_spec_suffix = ".mel"
        
    spec_filename = ref.replace(
        "/wave/", "/feature/mel_spec/").replace(get_file_extension(audiopath), mel_spec_suffix)
    assert os.path.exists(spec_filename), spec_filename

    spec = torch.load(spec_filename, map_location='cpu', weights_only=False)
    return {"mel_ref": spec}


def get_ref_dac_token(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    assert len(self.ref_dict[spk]) > 0, spk

    ref = random.choice(self.ref_dict[spk])
    if self.debug:
        print(f"Speaker Ref : {ref}")
        
    if "dac_audio_token_suffix" in self.optional.keys():
        dac_audio_token_suffix = self.optional['dac_audio_token_suffix']
    else:
        dac_audio_token_suffix = ".c"
        
    dac_filename = ref.replace(
        "/wave/", "/feature/dac/").replace(get_file_extension(audiopath), dac_audio_token_suffix)
    assert os.path.exists(dac_filename), dac_filename

    ref_dac_token = torch.load(dac_filename, map_location='cpu', weights_only=False)
    return {"ref_dac_token": ref_dac_token}

def get_ref_dac_emb(self, audiopath_sid_text):
    dac_token = self.get_ref_dac_token(audiopath_sid_text)['ref_dac_token'].long().unsqueeze(0)
    with torch.no_grad():
        dac_emb, _, _ = self.tokenizer['dac_tokenizer'].quantizer.from_codes(dac_token)
    dac_emb = dac_emb.squeeze(0)
    return {
        "ref_dac_emb": dac_emb,
    }

def get_club(self, audiopath_sid_text, l = 480, pad_value = 0):
    if 'club_loop_num' in self.optional.keys():
        n = self.optional['club_loop_num']
    else:
        n = 4
    
    feature_padded = torch.FloatTensor(n, 80, l)
    feature_padded.fill_(pad_value)

    for i in range(n):
        pt = random.choice(self.audiopaths_sid_text)
        feature = torch.load(pt[0].replace("/wave/", "/feature/mel_spec/").replace(get_file_extension(pt), f".160"))
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
        "/wave/", "/feature/mel_spec/").replace(get_file_extension(pt), mel_spec_suffix)
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
        "/wave/", "/feature/mhubert/").replace(get_file_extension(pt), ".code")
    assert os.path.exists(spec_filename), spec_filename

    spec = torch.load(spec_filename, map_location=torch.device('cpu'))
    if spec.shape[-1] > 150:
        start = random.randint(0, spec.shape[-1] - 150)
        spec = spec[start: start+150]
    return {"mhubert_ref": spec}

def get_ori_text(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    return {"ori_text": self.ori_text_dict[audiopath]}

def get_text(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    ori_text = self.ori_text_dict[audiopath]
    text_norm = self.frontend.textonly_to_idx(ori_text)
    return {"text": text_norm}

def get_trans_input_text_token(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    ori_text = self.ori_text_dict[audiopath]
    ori_text = self.tokenizer["input_text"](ori_text)
    input_text_token = ori_text["input_ids"]

    return {"input_text_token": input_text_token}

def get_trans_spk_desc_token(self, audiopath_sid_text):
    spk_desc_token = self.tokenizer["input_text"]("Female")
    spk_desc_token = spk_desc_token["input_ids"]

    return {"spk_desc_token": spk_desc_token}

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

def get_resampled_wave_audio(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    data, sr = torchaudio.load(audiopath)
    data = torchaudio.functional.resample(data, self.optional['ori_sample_rate'], self.optional['tar_sample_rate'], lowpass_filter_width=6)
    return {"wave_audio": data.squeeze(0)}

def get_wave_audio(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    data, sr = torchaudio.load(audiopath)
    
    return {"wave_audio": data.squeeze(0)}

def get_seg_wave_audio(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    data, sr = torchaudio.load(audiopath)
    wave_seg_size = self.optional['wave_seg_size']

    # 如果音频长度小于 wave_seg_size，则在后面填充 0
    if data.shape[-1] < wave_seg_size:
        pad_size = wave_seg_size - data.shape[-1]
        data = torch.nn.functional.pad(data, (0, pad_size))  # 在最后补零
    else:
        # 随机截取一个片段
        rand_idx = random.randint(0, data.shape[-1] - wave_seg_size)
        data = data[:, rand_idx: rand_idx + wave_seg_size]
    
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
    audiopath, _, _, _, _ = audiopath_sid_text
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/linear_spec/",
            get_file_extension(audiopath): ".linear"
        },
        return_key="linear_spec",
    )

def get_240_mel_spec(self, audiopath_sid_text):
    audiopath, _, _, _, _ = audiopath_sid_text
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/mel_spec/",
            get_file_extension(audiopath): ".240.mel"
        },
        return_key="mel_spec",
    )

def get_240_linear_spec(self, audiopath_sid_text):
    audiopath, _, _, _, _ = audiopath_sid_text
    return self.torch_load_single(
        audiopath_sid_text,
        path_replaecments={
            "/wave/": "/feature/linear_spec/",
            get_file_extension(audiopath): ".240.pad.linear"
        },
        return_key="linear_spec",
    )
    
def get_spk_desc_with_neg_emb(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    spk_desc = from_pickle(audiopath.replace("/wave/", "/feature/spk_desc/").replace(get_file_extension(audiopath), ".pk"))
    spk_desc = random.choice(spk_desc)
    
    if len(spk_desc) > 720:
        spk_desc = random.choice(spk_desc)
        
    neg_emb = []
    if "sample_neg_speaker" in self.optional.keys():
        
        if "facodec_spk_postfix" in self.optional.keys():
            facodec_spk_postfix = self.optional["facodec_spk_postfix"]
        else:
            facodec_spk_postfix = ".s"
    
        pos_speaker_info = self.get_spk_info(audiopath_sid_text)['spk_info']
        neg_utt = self.get_random_neg_utt(pos_speaker_info)['neg_utt']
        for neg in neg_utt:
            spk_filename = neg.replace("/wave/", "/feature/fcodec/").replace(get_file_extension(neg), facodec_spk_postfix)
            assert os.path.exists(spk_filename), spk_filename
            spk = torch.load(spk_filename, map_location=torch.device('cpu'))
            neg_emb.append(spk)
    
    return {"spk_desc": spk_desc, "neg_emb": neg_emb}

def get_dual_neg_input(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    spk_descs = from_pickle(audiopath.replace("/wave/", "/feature/spk_desc/").replace(get_file_extension(audiopath), ".pk"))
    query_spk_desc = random.choice(spk_descs)
    if len(query_spk_desc) > 720:
        query_spk_desc = query_spk_desc[:720]
    
    query_wave, query_sr = from_audio(audiopath, target_sr=16000)
    query_wave = query_wave[:, :160000].squeeze()
    assert query_sr == 16000
    
    neg_spk_desc = []
    neg_wave = []
    
    if "sample_neg_speaker" in self.optional.keys():
        pos_speaker_info = self.get_spk_info(audiopath_sid_text)['spk_info']
        neg_utt = self.get_random_neg_utt(pos_speaker_info)['neg_utt']
        for neg in neg_utt:
            cur_neg_desc = from_pickle(neg.replace("/wave/", "/feature/spk_desc/").replace(get_file_extension(neg), ".pk"))
            cur_neg_desc = random.choice(cur_neg_desc)
            neg_spk_desc.append(cur_neg_desc[:720])
    
    if "sample_neg_speaker" in self.optional.keys():
        pos_speaker_info = self.get_spk_info(audiopath_sid_text)['spk_info']
        neg_utt = self.get_random_neg_utt(pos_speaker_info)['neg_utt']
        for neg in neg_utt:
            cur_neg_wave, cur_neg_sr = from_audio(neg, target_sr=16000)
            assert cur_neg_sr == 16000
            neg_wave.append(cur_neg_wave[:, :160000].squeeze())
    
    if self.debug:
        print("-" * 50)
        print("POS DES:", query_spk_desc[:20])
        print("NEG DES:")
        for i in neg_spk_desc:
            print(i[:20])
        print("POS WAVE:", query_wave[:8])
        print("NEG WAVE:")
        for i in neg_wave:
            print(i[:8])
            
    return {"query_spk_desc": query_spk_desc, "query_wave": query_wave, "neg_spk_desc": neg_spk_desc, "neg_wave": neg_wave,}

def get_spk_label(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    ori_spk_label = from_pickle(audiopath.replace("/wave/", "/feature/spk_desc/").replace(get_file_extension(audiopath), ".dic"))

    if "speaker_label_to_int_dict" not in self.optional.keys():
        self.optional['speaker_label_to_int_dict'] = from_pickle("/home/mushan/data/filelists/feature/spklabel2int.pk")
        
    if 'speaker_label_input' not in self.optional.keys():
        self.optional['speaker_label_input'] = ['speaking_rate', 'gender', 'pitch', 'speaker_accent', 'age']
    
    spk_label = []
    for key in self.optional['speaker_label_input']:
        spk_label.append(self.optional['speaker_label_to_int_dict'][key][ori_spk_label[key]])
        
    spk_label = torch.LongTensor(spk_label)

    return {"spk_label": spk_label}

def get_onehot_spk_label(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    ori_spk_label = from_pickle(audiopath.replace("/wave/", "/feature/spk_desc/").replace(get_file_extension(audiopath), ".dic"))

    if "speaker_label_to_int_dict" not in self.optional.keys():
        self.optional['speaker_label_to_int_dict'] = from_pickle("/home/mushan/data/filelists/feature/spklabel2int.pk")
        
    if 'speaker_label_input' not in self.optional.keys():
        self.optional['speaker_label_input'] = ['speaking_rate', 'gender', 'pitch', 'speaker_accent', 'age']
        
    spk_label = {}
    for key in self.optional['speaker_label_input']:
        val = torch.LongTensor([self.optional['speaker_label_to_int_dict'][key][ori_spk_label[key]]])
        num_cls = len(self.optional['speaker_label_to_int_dict'][key])
        spk_label[key] = torch.nn.functional.one_hot(val, num_cls).float().squeeze()

    return {"onehot_spk_label": spk_label}
    
    
def get_spk_desc(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    if "speaker_description_suffix" not in self.optional.keys():
        self.optional['speaker_description_suffix'] = ".pk"
    spk_desc = from_pickle(audiopath.replace("/wave/", "/feature/spk_desc/").replace(get_file_extension(audiopath), self.optional['speaker_description_suffix']))
    spk_desc = random.choice(spk_desc)
    

        
    neg_desc = []
    if "sample_neg_speaker" in self.optional.keys():
        pos_speaker_info = self.get_spk_info(audiopath_sid_text)['spk_info']
        neg_utt = self.get_random_neg_utt(pos_speaker_info)['neg_utt']
        for neg in neg_utt:
            cur_neg_desc = from_pickle(neg.replace("/wave/", "/feature/spk_desc/").replace(get_file_extension(neg), self.optional['speaker_description_suffix']))
            cur_neg_desc = random.choice(cur_neg_desc)
            neg_desc.append(cur_neg_desc)
    else:
        neg_desc.append(" ")
    
    return {"spk_desc": spk_desc, "neg_desc": neg_desc}

def get_spk_info(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    spk_info = from_pickle(audiopath.replace("/wave/", "/feature/spk_desc/").replace(get_file_extension(audiopath), ".dic"))
    return {"spk_info": spk_info}

def get_random_neg_utt(self, pos_speaker_info):
    assert self.optional['sample_neg_speaker'] > 0
    if 'neg_utt_dict' not in self.optional.keys():
        self.optional['neg_utt_dict'] = from_pickle(f"/home/mushan/data/filelists/feature/spk2pt_clean.pk")
        self.optional['neg_utt_dict_key'] = list(self.optional['neg_utt_dict'].keys())
        
    str_pos_info = hashabledict()
    for key, value in pos_speaker_info.items():
        if isinstance(value, str):
            str_pos_info[key] = value
            
    if self.debug:
        print("POS KEY:", str_pos_info)
    neg_utt = []
    while (len(neg_utt) < self.optional['sample_neg_speaker']):

        random_key = random.choice(self.optional['neg_utt_dict_key'])
        while (random_key == str_pos_info):
            random_key = random.choice(self.optional['neg_utt_dict_key'])
            
        if self.debug:
            print("NEG KEY:", random_key)
            
            
        random_utt = random.choice(self.optional['neg_utt_dict'][random_key])
        neg_utt.append(random_utt)
        
    return {"neg_utt": neg_utt}
    

def get_facodec_spk(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    
    if "facodec_spk_postfix" in self.optional.keys():
        facodec_spk_postfix = self.optional["facodec_spk_postfix"]
    else:
        facodec_spk_postfix = ".s"
        
    spk_filename = audiopath.replace("/wave/", "/feature/fcodec/").replace(get_file_extension(audiopath), facodec_spk_postfix)
    assert os.path.exists(spk_filename), spk_filename
    spk = torch.load(spk_filename, map_location=torch.device('cpu'))
    
    return {"facodec_speaker": spk}

    
def get_facodec(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    
    code_filename = audiopath.replace("/wave/", "/feature/fcodec/").replace(get_file_extension(audiopath), ".c")
    assert os.path.exists(code_filename), code_filename
    code = torch.load(code_filename, map_location=torch.device('cpu'))
    
    spk_filename = audiopath.replace("/wave/", "/feature/fcodec/").replace(get_file_extension(audiopath), ".s")
    assert os.path.exists(spk_filename), spk_filename
    spk = torch.load(spk_filename, map_location=torch.device('cpu'))
    
    return {"facodec_prosody": code[:1, :],
            "facodec_content": code[1:3, :],
            "facodec_details": code[3:, :],
            "facodec_speaker": spk}

def get_your_ref(self, audiopath_sid_text):
    audiopath, spk, dur, ori_text, text = audiopath_sid_text
    assert len(self.ref_dict[spk]) > 0, spk

    ref = random.choice(self.ref_dict[spk])
    spec_filename = ref.replace(get_file_extension(pt), ".emb")
    assert os.path.exists(spec_filename), spec_filename

    spec = torch.load(spec_filename, map_location=torch.device('cpu'))
    return {"your_ref": spec}

def get_audio_text_speaker_pair(self, audiopath_sid_text):
    res = {}
    self.temp_arg = None
    for key in self.data_list:
        # try:
        func = getattr(self, f"get_{key}")
        res.update(func(audiopath_sid_text))
        # except AttributeError:
        #     raise NotImplementedError("Class `{}` does not implement `get_{}`".format(self.__class__.__name__, key))
        # func = getattr(self, f"get_{key}")
        # res.update(func(audiopath_sid_text))

    return res