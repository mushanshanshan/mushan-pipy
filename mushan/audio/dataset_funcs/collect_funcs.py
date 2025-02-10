from glob import glob
import librosa
import math
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
import torch.nn.functional as F
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
    
    mel_spec = self.collect_2D_with_length(
        batch,
        ids_sorted_decreasing,
        feature_key="mel_spec",
        feature_dtype=torch.float
    )
    
    if "mel_spec_multiple_pad" in self.optional.keys():
        multiple = self.optional["mel_spec_multiple_pad"]

        seq_len = mel_spec["mel_spec"].shape[-1]
        target_len = math.ceil(seq_len / multiple) * multiple
        padding_value = 0
        if target_len > seq_len:
            pad_size = (0, target_len - seq_len)  # 在 seq_len 方向右侧 pad
            mel_spec["mel_spec"] = F.pad(mel_spec["mel_spec"], pad=pad_size, mode='constant', value=padding_value)

            
    return mel_spec
    

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
    
def collect_mel_spec_for_dac(self, batch, ids_sorted_decreasing):
    mel_spec =  self.collect_2D_with_length(
        batch,
        ids_sorted_decreasing,
        feature_key="mel_spec",
        feature_dtype=torch.float
    )["mel_spec"]
    
    if "dac_pad_multiple" in self.optional.keys():
        multiple = self.optional['dac_pad_multiple']
        seq_len = mel_spec.shape[-1]
        target_len = math.ceil(seq_len / multiple) * multiple
    
        if target_len > seq_len:
            pad_size = (0, target_len - seq_len)  # 在 seq_len 方向右侧 pad
            mel_spec = F.pad(mel_spec, pad=pad_size, mode='constant', value=0)

    return {"mel_spec": mel_spec}
    
def collect_dac_audio_emb(self, batch, ids_sorted_decreasing):
    dac_audio_emb = self.collect_2D_with_length(
        batch,
        ids_sorted_decreasing,
        feature_key="dac_audio_emb",
        feature_dtype=torch.float
    )['dac_audio_emb']
    
    if "dac_pad_multiple" in self.optional.keys():
        
        multiple = self.optional['dac_pad_multiple']
        seq_len = dac_audio_emb.shape[-1]
        target_len = math.ceil(seq_len / multiple) * multiple
    
        if target_len > seq_len:
            pad_size = (0, target_len - seq_len)  # 在 seq_len 方向右侧 pad
            dac_audio_emb = F.pad(dac_audio_emb, pad=pad_size, mode='constant', value=0)
            
    return {'dac_audio_emb': dac_audio_emb}
    
def collect_ref_dac_emb(self, batch, ids_sorted_decreasing):
    ref_dac_emb = self.collect_2D_with_length(
        batch,
        ids_sorted_decreasing,
        feature_key="ref_dac_emb",
        feature_dtype=torch.float
    )['ref_dac_emb']
            
    return {'ref_dac_emb': ref_dac_emb}
    

def collect_dac_audio_token(self, batch, ids_sorted_decreasing):
    
    def create_transformer_mask_for_codebooks(
        x: torch.Tensor, 
        pad_token_id: int = 0
    ):
        pad_mask = (x == pad_token_id).all(dim=-1)
        valid_mask = ~pad_mask
        valid_mask = valid_mask.unsqueeze(1)
        
        return valid_mask
    
    # dac_audio_token = [torch.tensor(i["dac_audio_token"]).transpose(0, 1) for i in batch]
    # dac_audio_token_length = torch.LongTensor([i["dac_audio_token"].shape[-1] for i in batch])
    dac_audio_token = []
    dac_audio_token_length = torch.LongTensor(len(batch))
    
    for i in range(len(ids_sorted_decreasing)):
        row = batch[ids_sorted_decreasing[i]]
        feature = row["dac_audio_token"]
        dac_audio_token_length[i] = feature.shape[-1]
        dac_audio_token.append(feature.transpose(0, 1))
    
    if "dac_pad_value" in self.optional.keys():
        padding_value = self.optional['dac_pad_value']
    else:
        padding_value = -100
        
    dac_audio_token = torch.nn.utils.rnn.pad_sequence(dac_audio_token, batch_first=True, padding_value=padding_value)
    
    if "dac_pad_multiple" in self.optional.keys():
        
        multiple = self.optional['dac_pad_multiple']
        seq_len = dac_audio_token.shape[1]
        target_len = math.ceil(seq_len / multiple) * multiple
    
        if target_len > seq_len:
            pad_size = (0, target_len - seq_len)  # 在 seq_len 方向右侧 pad
            dac_audio_token = dac_audio_token.transpose(-1, -2)
            dac_audio_token = F.pad(dac_audio_token, pad=pad_size, mode='constant', value=padding_value)
            dac_audio_token = dac_audio_token.transpose(-1, -2)

    dac_audio_token_mask = create_transformer_mask_for_codebooks(dac_audio_token, padding_value)
    
    return {"dac_audio_token": dac_audio_token.long(),
            "dac_audio_token_length": dac_audio_token_length.long(),
            "dac_audio_token_mask": dac_audio_token_mask.float()}
    
    
def collect_delay_dac_audio_token(self, batch, ids_sorted_decreasing):
    def build_delay_pattern_mask(
        input_ids: torch.LongTensor, bos_token_id: int, pad_token_id: int, max_length: int, num_codebooks: int
    ):
        """Build a delayed pattern mask to the input_ids. Each codebook is offset by the previous codebook by
        one, giving a delayed pattern mask at the start of sequence and end of sequence. Take the example where there
        are 4 codebooks and a max sequence length of 8, we have the delayed pattern mask of shape `(codebooks,
        seq_len)`:
        - [B, -1, -1, -1, -1, P, P, P]
        - [B, B, -1, -1, -1, -1, P, P]
        - [B, B, B, -1, -1, -1, -1, P]
        - [B, B, B, B, -1, -1, -1, -1]
        where P is the special padding token id and -1 indicates that the token is valid for prediction. If we include
        a prompt (decoder input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the
        mask is set to the value in the prompt:
        - [B, a, b, -1, -1, P, P, P]
        - [B, B, c, d, -1, -1, P, P]
        - [B, B, B, e, f, -1, -1, P]
        - [B, B, B, B, g, h, -1, -1]
        where a-h indicate the input prompt (decoder input ids) that are offset by 1. Now, we only override the -1
        tokens in our prediction.
        """
        # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
        input_ids = input_ids.reshape(-1, num_codebooks, input_ids.shape[-1])
        bsz, num_codebooks, seq_len = input_ids.shape

        input_ids_shifted = torch.ones((bsz, num_codebooks, max_length), dtype=torch.long, device=input_ids.device) * -1

        # we only apply the mask if we have a large enough seq len - otherwise we return as is
        if max_length < 2 * num_codebooks - 1:
            return input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1)

        # fill the shifted ids with the prompt entries, offset by the codebook idx
        for codebook in range(num_codebooks):
            # mono channel - loop over the codebooks one-by-one
            input_ids_shifted[:, codebook, codebook : seq_len + codebook] = input_ids[:, codebook]

        # construct a pattern mask that indicates the positions of padding tokens for each codebook
        # first fill the upper triangular part (the EOS padding)
        eos_delay_pattern = torch.triu(
            torch.ones((num_codebooks, max_length), dtype=torch.bool), diagonal=max_length - num_codebooks + 1
        )
        # then fill the lower triangular part (the BOS padding)
        bos_delay_pattern = torch.tril(torch.ones((num_codebooks, max_length), dtype=torch.bool))

        bos_mask = ~(bos_delay_pattern).to(input_ids.device)
        eos_mask = ~(eos_delay_pattern).to(input_ids.device)
        mask = ~(bos_delay_pattern + eos_delay_pattern).to(input_ids.device)
        input_ids = mask * input_ids_shifted + ~bos_mask * bos_token_id + ~eos_mask * pad_token_id

        # find the first position to start generating - this is the first place we have the -1 token
        # and will always be in the first codebook (since it has no codebook offset)
        first_codebook_ids = input_ids[:, 0, :]
        start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
        if len(start_ids) > 0:
            first_start_id = min(start_ids)
        else:
            # we have no tokens that need to be filled - return entire matrix of input ids
            first_start_id = seq_len

        # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
        pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
        input_ids = input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)
        return input_ids, pattern_mask


    def postprocess_dataset(labels):
        audio_encoder_bos_token_id = 1025
        audio_encoder_eos_token_id = 1024
        num_codebooks = 9
        # (1, codebooks, seq_len)
        labels = torch.tensor(labels).unsqueeze(0)
        bos_labels = torch.ones((1, num_codebooks, 1)) * audio_encoder_bos_token_id
        # add bos
        labels = torch.cat([bos_labels, labels], dim=-1)

        labels, delay_pattern_mask = build_delay_pattern_mask(
            labels,
            bos_token_id=audio_encoder_bos_token_id,
            pad_token_id=audio_encoder_eos_token_id,
            max_length=labels.shape[-1] + num_codebooks,
            num_codebooks=num_codebooks,
        )

        # the first ids of the delay pattern mask are precisely labels, we use the rest of the labels mask
        # to take care of EOS
        # we want labels to look like this:
        #  - [B, a, b, E, E, E, E]
        #  - [B, B, c, d, E, E, E]
        #  - [B, B, B, e, f, E, E]
        #  - [B, B, B, B, g, h, E]
        labels = torch.where(delay_pattern_mask == -1, audio_encoder_eos_token_id, delay_pattern_mask)

        # the first timestamp is associated to a row full of BOS, let's get rid of it
        # we also remove the last timestampts (full of PAD)
        output = labels[:, 1:]
        return output

    dac_audio_token = [postprocess_dataset(torch.tensor(i["dac_audio_token"])).transpose(0, 1) for i in batch]
    # print(dac_audio_token)
    # for i in dac_audio_token:
    #     print(i.shape)
    dac_audio_token = torch.nn.utils.rnn.pad_sequence(dac_audio_token, batch_first=True, padding_value=-100)
    return {"dac_audio_token": dac_audio_token}
    
def collect_mms_corr_mel_160(self, batch, ids_sorted_decreasing):
    return self.collect_2D_with_length(
        batch,
        ids_sorted_decreasing,
        feature_key="mel_spec",
        feature_dtype=torch.float,
        max_length=self.optional['mms_seg_size'] * 2
    )
    
    # def collect_2D_with_length(self, batch, ids_sorted_decreasing, feature_key, length_dim_idx=1, max_length = 0, pad_value=0, feature_dtype=torch.float):
    
def collect_fun_codec(self, batch, ids_sorted_decreasing):
    prosody = self.collect_2D_with_length(
        batch,
        ids_sorted_decreasing,
        feature_key="fun_codec_prosody",
        feature_dtype=torch.long,
        pad_value = 1024
    )
    
    content = self.collect_2D_with_length(
        batch,
        ids_sorted_decreasing,
        feature_key="fun_codec_content",
        feature_dtype=torch.long,
        pad_value = 1024
    )
    
    details = self.collect_2D_with_length(
        batch,
        ids_sorted_decreasing,
        feature_key="fun_codec_details",
        feature_dtype=torch.long,
        pad_value = 1024
    )
    
    spks = torch.stack([batch[i]["fun_codec_speaker"] for i in ids_sorted_decreasing])
    
    return {"fun_codec_prosody": prosody["fun_codec_prosody"],
            "fun_codec_content": content["fun_codec_content"],
            "fun_codec_details": details["fun_codec_details"],
            "fun_codec_speaker": spks}

def collect_audio_path(self, batch, ids_sorted_decreasing):
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
    
def collect_text_token(self, batch, ids_sorted_decreasing, pad_value=511):
    assert "text_token_pad" in self.optional.keys()
    return self.collect_1D_with_length(
        batch,
        ids_sorted_decreasing,
        feature_key="text_token",
        pad_value=self.optional["text_token_pad"],
        feature_dtype=torch.long
    )
    
def collect_ori_text(self, batch, ids_sorted_decreasing):
    all_text = []
    for i in range(len(ids_sorted_decreasing)):
        row = batch[ids_sorted_decreasing[i]]
        all_text.append(row['ori_text'])
    return {"ori_text": all_text}

        
        
        
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
    
def collect_trans_input_text_token(self, batch, ids_sorted_decreasing):
    input_text_token = [{"input_ids": feature["input_text_token"]} for feature in batch]
    input_text_token = self.tokenizer["input_text"].pad(
        input_text_token,
        return_tensors="pt",
        padding=True,
    )

    return {"input_text_token": input_text_token["input_ids"],
            "input_text_attention_mask": input_text_token["attention_mask"]}
    
def collect_trans_spk_desc_token(self, batch, ids_sorted_decreasing):
    spk_desc_token = [{"input_ids": feature["spk_desc_token"]} for feature in batch]
    spk_desc_token = self.tokenizer["input_text"].pad(
        spk_desc_token,
        return_tensors="pt",
        padding=True,
    )

    return {"spk_desc_token": spk_desc_token["input_ids"],
            "spk_desc_token_mask": spk_desc_token["attention_mask"]}

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