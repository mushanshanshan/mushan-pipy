import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
import unittest
from torch.utils.data import Dataset, DataLoader
import tqdm
import gdown
import fairseq
import hydra
from pathlib import Path
import os

def load_ssl_model(cp_path):
    ssl_model_type = "wav2vec_small.pt"
    wavlm =  "WavLM" in ssl_model_type
    if wavlm:
        checkpoint = torch.load(cp_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        ssl_model = WavLM(cfg)
        ssl_model.load_state_dict(checkpoint['model'])
        if 'Large' in ssl_model_type:
            SSL_OUT_DIM = 1024
        else:
            SSL_OUT_DIM = 768
    else:
        if ssl_model_type == "wav2vec_small.pt":
            SSL_OUT_DIM = 768
        elif ssl_model_type in ["w2v_large_lv_fsh_swbd_cv.pt", "xlsr_53_56k.pt"]:
            SSL_OUT_DIM = 1024
        else:
            print("*** ERROR *** SSL model type " + ssl_model_type + " not supported.")
            exit()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_path]
        )
        ssl_model = model[0]
        ssl_model.remove_pretraining_modules()
    return SSL_model(ssl_model, SSL_OUT_DIM, wavlm)

class SSL_model(nn.Module):
    def __init__(self,ssl_model,ssl_out_dim,wavlm) -> None:
        super(SSL_model,self).__init__()
        self.ssl_model, self.ssl_out_dim = ssl_model, ssl_out_dim
        self.WavLM = wavlm

    def forward(self,batch):
        wav = batch['wav'] 
        wav = wav.squeeze(1) # [batches, audio_len]
        if self.WavLM:
            x = self.ssl_model.extract_features(wav)[0]
        else:
            res = self.ssl_model(wav, mask=False, features_only=True)
            x = res["x"]
        return {"ssl-feature":x}
    def get_output_dim(self):
        return self.ssl_out_dim


class PhonemeEncoder(nn.Module):
    '''
    PhonemeEncoder consists of an embedding layer, an LSTM layer, and a linear layer.
    Args:
        vocab_size: the size of the vocabulary
        hidden_dim: the size of the hidden state of the LSTM
        emb_dim: the size of the embedding layer
        out_dim: the size of the output of the linear layer
        n_lstm_layers: the number of LSTM layers
    '''
    def __init__(self, vocab_size, hidden_dim, emb_dim, out_dim,n_lstm_layers,with_reference=True) -> None:
        super().__init__()
        self.with_reference = with_reference
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim,
                               num_layers=n_lstm_layers, dropout=0.1, bidirectional=True)
        self.linear = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim*self.with_reference, out_dim),
                nn.ReLU()
                )
        self.out_dim = out_dim

    def forward(self,batch):
        seq = batch['phonemes']
        lens = batch['phoneme_lens']
        reference_seq = batch['reference']
        reference_lens = batch['reference_lens']
        emb = self.embedding(seq)
        emb = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lens, batch_first=True, enforce_sorted=False)
        _, (ht, _) = self.encoder(emb)
        feature = ht[-1] + ht[0]
        if self.with_reference:
            if reference_seq==None or reference_lens ==None:
                raise ValueError("reference_batch and reference_lens should not be None when with_reference is True")
            reference_emb = self.embedding(reference_seq)
            reference_emb = torch.nn.utils.rnn.pack_padded_sequence(
                reference_emb, reference_lens, batch_first=True, enforce_sorted=False)
            _, (ht_ref, _) = self.encoder(emb)
            reference_feature = ht_ref[-1] + ht_ref[0]
            feature = self.linear(torch.cat([feature,reference_feature],1))
        else:
            feature = self.linear(feature)
        return {"phoneme-feature": feature}
    def get_output_dim(self):
        return self.out_dim

class DomainEmbedding(nn.Module):
    def __init__(self,n_domains,domain_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_domains,domain_dim)
        self.output_dim = domain_dim
    def forward(self, batch):
        return {"domain-feature": self.embedding(batch['domains'])}
    def get_output_dim(self):
        return self.output_dim


class LDConditioner(nn.Module):
    '''
    Conditions ssl output by listener embedding
    '''
    def __init__(self,input_dim, judge_dim, num_judges=None):
        super().__init__()
        self.input_dim = input_dim
        self.judge_dim = judge_dim
        self.num_judges = num_judges
        assert num_judges !=None
        self.judge_embedding = nn.Embedding(num_judges, self.judge_dim)
        # concat [self.output_layer, phoneme features]
        
        self.decoder_rnn = nn.LSTM(
            input_size = self.input_dim + self.judge_dim,
            hidden_size = 512,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        ) # linear?
        self.out_dim = self.decoder_rnn.hidden_size*2

    def get_output_dim(self):
        return self.out_dim


    def forward(self, x, batch):
        judge_ids = batch['judge_id']
        if 'phoneme-feature' in x.keys():
            concatenated_feature = torch.cat((x['ssl-feature'], x['phoneme-feature'].unsqueeze(1).expand(-1,x['ssl-feature'].size(1) ,-1)),dim=2)
        else:
            concatenated_feature = x['ssl-feature']
        if 'domain-feature' in x.keys():
            concatenated_feature = torch.cat(
                (
                    concatenated_feature,
                    x['domain-feature']
                    .unsqueeze(1)
                    .expand(-1, concatenated_feature.size(1), -1),
                ),
                dim=2,
            )
        if judge_ids != None:
            concatenated_feature = torch.cat(
                (
                    concatenated_feature,
                    self.judge_embedding(judge_ids)
                    .unsqueeze(1)
                    .expand(-1, concatenated_feature.size(1), -1),
                ),
                dim=2,
            )
            decoder_output, (h, c) = self.decoder_rnn(concatenated_feature)
        return decoder_output

class Projection(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, range_clipping=False):
        super(Projection, self).__init__()
        self.range_clipping = range_clipping
        output_dim = 1
        if range_clipping:
            self.proj = nn.Tanh()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim
    
    def forward(self, x, batch):
        output = self.net(x)

        # range clipping
        if self.range_clipping:
            return self.proj(output) * 2.0 + 3
        else:
            return output
    def get_output_dim(self):
        return self.output_dim


class BaselineLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.construct_model()
        self.save_hyperparameters()
    
    def construct_model(self):
        self.feature_extractors = nn.ModuleList([
            load_ssl_model(cp_path=str(Path(__file__).resolve().parent.joinpath("wav2vec"))),
            DomainEmbedding(3,128),
        ])
        output_dim = sum([ feature_extractor.get_output_dim() for feature_extractor in self.feature_extractors])
        output_layers = [
            LDConditioner(judge_dim=128,num_judges=3000,input_dim=output_dim)
        ]
        output_dim = output_layers[-1].get_output_dim()
        output_layers.append(
            Projection(hidden_dim=2048,activation=torch.nn.ReLU(),range_clipping=False,input_dim=output_dim)

        )

        self.output_layers = nn.ModuleList(output_layers)

    def forward(self, inputs):
        outputs = {}
        for feature_extractor in self.feature_extractors:
            outputs.update(feature_extractor(inputs))
        x = outputs
        for output_layer in self.output_layers:
            x = output_layer(x,inputs)
        return x


class Dataset(Dataset):
    def __init__(self, wavlist):
        self.wavlist = wavlist
        _, self.sr = torchaudio.load(self.wavlist[0])

    def __len__(self):
        return len(self.wavlist)

    def __getitem__(self, idx):
        fname = self.wavlist[idx]
        wav, _ = torchaudio.load(fname)
        sample = {
            "fname": fname,
            "wav": wav}
        return sample
    
    def collate_fn(self, batch):
        max_len = max([x["wav"].shape[1] for x in batch])
        names = []
        out = []
        # Performing repeat padding
        for t in batch:
            names.append(t["fname"])
            wav = t["wav"]
            amount_to_pad = max_len - wav.shape[1]
            padding_tensor = wav.repeat(1,1+amount_to_pad//wav.size(1))
            out.append(torch.cat((wav,padding_tensor[:,:amount_to_pad]),dim=1))
        return names, torch.stack(out, dim=0)

class Score:
    """Predicting score for each audio clip."""

    def __init__(
        self,
        input_sample_rate: int = 16000,
        device: str = "cpu"):
        """
        Args:
            ckpt_path: path to pretrained checkpoint of UTMOS strong learner.
            input_sample_rate: sampling rate of input audio tensor. The input audio tensor
                is automatically downsampled to 16kHz.
        """
        print(f"Using device: {device}")
        
        self.ckpt_path = str(Path(__file__).resolve().parent.joinpath("g_05000000"))
        if not os.path.exists(self.ckpt_path):
            gdown.download(url="https://drive.google.com/file/d/1XiLi8V4t40gde1oeL0uh7CTIiqupNJHm/view?usp=drive_link", output=self.ckpt_path, quiet=False, fuzzy=True)
            
        self.wav2vec_path = str(Path(__file__).resolve().parent.joinpath("wav2vec"))
        if not os.path.exists(self.wav2vec_path):
            gdown.download(url="https://drive.google.com/file/d/1Vi3FnK4kOijiFwy_SJXjarAXhDNwpIto/view?usp=share_link", output=self.wav2vec_path, quiet=False, fuzzy=True)

        
        self.device = device
        self.model = BaselineLightningModule.load_from_checkpoint(
            self.ckpt_path,  map_location=self.device).eval().to(self.device)
        self.in_sr = input_sample_rate
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=input_sample_rate,
            new_freq=16000,
            resampling_method="sinc_interpolation",
            lowpass_filter_width=6,
            dtype=torch.float32,
        ).to(device)
    
    def score(self, wavs: torch.tensor) -> torch.tensor:
        """
        Args:
            wavs: audio waveform to be evaluated. When len(wavs) == 1 or 2,
                the model processes the input as a single audio clip. The model
                performs batch processing when len(wavs) == 3. 
        """
        if len(wavs.shape) == 1:
            out_wavs = wavs.unsqueeze(0).unsqueeze(0)
        elif len(wavs.shape) == 2:
            out_wavs = wavs.unsqueeze(0)
        elif len(wavs.shape) == 3:
            out_wavs = wavs
        else:
            raise ValueError('Dimension of input tensor needs to be <= 3.')
        if self.in_sr != 16000:
            out_wavs = self.resampler(out_wavs)
        bs = out_wavs.shape[0]
        batch = {
            'wav': out_wavs,
            'domains': torch.zeros(bs, dtype=torch.int).to(self.device),
            'judge_id': torch.ones(bs, dtype=torch.int).to(self.device)*288
        }
        with torch.no_grad():
            output = self.model(batch)
        
        return output.mean(dim=1).squeeze(1).cpu().detach().numpy()*2 + 3
    
    def pred(self, wav_list, batch_size = 10):

        dataset = Dataset(wav_list)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            num_workers=4)
        
        set_res = {}
        for names, batch in tqdm.tqdm(loader):
            scores = self.score(batch.to(self.device))
            ds = {names[i] : scores[i] for i in range(len(names))}
            set_res.update(ds)
            
        return set_res