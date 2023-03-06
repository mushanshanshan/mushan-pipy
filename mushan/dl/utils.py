import os
import glob
import sys
import argparse
import logging
from dotmap import DotMap
from datetime import datetime
import tomli
import json
import subprocess
from loguru import logger
import numpy as np
from scipy.io.wavfile import read
import torch
import pprint
from collections import OrderedDict
import shutil
import soundfile
import random
import time



MATPLOTLIB_FLAG = False

def load_checkpoint(checkpoint_path, model, optimizer=None, slience=False, partially=False):
    global logger
    
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Backward compatibility
    if 'global_step' in checkpoint_dict.keys():
        global_step = checkpoint_dict['global_step']
    else:
        global_step = -1
        
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    suc = 0
    fail = 0
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            suc += 1
        except:
            logger.error("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
            fail += 1
            
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    if not slience:
        logger.info("Loaded checkpoint '{}' (iteration {}) : suc {}, fail {}".format(checkpoint_path, iteration, suc, fail))
    return model, optimizer, learning_rate, iteration, global_step

     
def load_for_tine_tune(uni_checkpoint_path, model):
    
    if hasattr(model, 'module'):
        ori_keys = set(model.module.state_dict().keys())
    else:
        ori_keys = set(model.state_dict().keys())
    
    uni_state_dict = torch.load(uni_checkpoint_path, map_location='cpu')['model']
    new_state_dict = {}
    
    for k, v in uni_state_dict.items():
        new_state_dict[k] = v
    
    new_keys = set(new_state_dict.keys())
        
    logger.info("Loaded checkpoint '{}' : suc {}".format(uni_checkpoint_path, len(list(new_state_dict.keys()))))
     
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)


def load_tine_tune(ft_checkpoint_path, uni_checkpoint_path, model):
    global logger
    
    if hasattr(model, 'module'):
        ori_state_dict = model.module.state_dict()
    else:
        ori_state_dict = model.state_dict()
        
    req_set = set(ori_state_dict.keys())
    
    uni_state_dict = torch.load(uni_checkpoint_path, map_location='cpu')['model']
    if ft_checkpoint_path != None:
        ft_state_dict = torch.load(ft_checkpoint_path, map_location='cpu')
    count = 0

    new_state_dict = {}
    new_dict = set()
    
    for k, v in uni_state_dict.items():
        new_state_dict[k] = uni_state_dict[k]
        new_dict.add(k)
        
    if ft_checkpoint_path != None:
        for k, v in ft_state_dict.items():
            count += 1

            new_state_dict[k] = ft_state_dict[k]
            new_dict.add(k)
    
    new_set = set(new_state_dict.keys())
    for r in req_set:
        assert r in new_set, f"Parameter not found: {r}"
            
    print(f"Load {count} fine-tuned parameters! All {len(req_set)} parameters check pass!")    
    
     
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)


def save_checkpoint(model, 
                    optimizer, 
                    learning_rate = 1e-4, 
                    iteration = 0, 
                    global_step = 0, 
                    checkpoint_path = "./logs/temp.pk"):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
        
    if optimizer != None:
        optsd = optimizer.state_dict()
    else:
        optsd = None
        
    torch.save({'model': state_dict,
                'iteration': iteration,
                'global_step': global_step,
                'optimizer': optsd,
                'learning_rate': learning_rate}, checkpoint_path)
    
    logger.info("Saving model and optimizer state at iteration {} to {}".format(
        iteration, checkpoint_path))
    
def save_fine_tune(model, checkpoint_path):
    fine_tune = OrderedDict()
    
    if hasattr(model, 'module'):
        model = model.module
    else:
        model = model

    for k,v in model.named_parameters():
        if v.requires_grad == True:
            fine_tune[k] = v.data
    
    assert len(list(fine_tune.keys())) > 0, "No parameters to save!"
    
    torch.save(fine_tune, checkpoint_path)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, image_dataformats='HWC',audios={}, audio_sampling_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats=image_dataformats)
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def vits_latest_checkpoint_path(checkpoint_path):
    G_list = glob.glob(checkpoint_path + "/G_*")
    D_list = glob.glob(checkpoint_path + "/D_*")
    
    G_list = [i.split("G_")[-1] for i in G_list]
    D_list = [i.split("D_")[-1] for i in D_list]
    
    avaliable_check_point = list(set(G_list).intersection(set(D_list)))
    
    assert len(avaliable_check_point) > 0, f"Can not fould avaliable checkpoints in {os.path.split(checkpoint_path)[0]}"
    
    avaliable_check_point.sort(key=lambda x:int(x.split(".")[0]))
    return avaliable_check_point[-1]

def tacotron_latest_checkpoint_path(checkpoint_path):
    G_list = glob.glob(checkpoint_path + "/G_*")
    G_list = [i.split("G_")[-1] for i in G_list]

    G_list.sort(key=lambda x:int(x.split(".")[0]))
    return G_list[-1]

def torch_latest_checkpoint_path(checkpoint_path):
    G_list = glob.glob(checkpoint_path + "/C_*")
    G_list = [i.split("C_")[-1] for i in G_list]

    G_list.sort(key=lambda x:int(x.split(".")[0]))
    return G_list[-1]

def uni_latest_checkpoint_path(checkpoint_path, prefix):
    G_list = glob.glob(checkpoint_path + f"/{prefix}_*")
    G_list = [i.split(f"{prefix}_")[-1] for i in G_list]

    G_list.sort(key=lambda x:int(x.split(".")[0]))
    return G_list[-1]


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    data, sampling_rate = soundfile.read(full_path)
    data = torch.FloatTensor(data.astype(np.float32)).unsqueeze(0)
    return data, sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True):
    global logger
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="default",
                        help='TOML file for configuration')
    parser.add_argument('-m', '--model', type=str, default="vits",
                        help='Model name')
    parser.add_argument('-t', '--tag', type=str, default="",
                        help='Training tags')
    parser.add_argument('-g', '--n_gpus', type=int, default=1,
                        help='Number of GPU')
    parser.add_argument('-port', type=int, default=-1,
                        help='Port for DDP')
    parser.add_argument('-cont', '--continue_train', action='store_true', default=False,
                        help='Continue training')
    parser.add_argument('-test', '--test_mode', action='store_true', default=False,
                        help='Test or not')
    args = parser.parse_args()

    config_file = args.config
    
    with open(config_file, "rb") as f:
        config = DotMap(tomli.load(f))

    config.train.tags += args.tag
    if "tags" in config.train.tune_dict.keys():
        config.train.tags += config.train.tune_dict.tags
    if args.port > 0:
        config.train.port = args.port
    else:
        config.train.port = random.randint(35777,36000)
    config.train.n_gpus = args.n_gpus
    config.train.model = args.model
    config.train.test = args.test_mode or args.tag == 'test'
    config.checkpoint.load_checkpoint = False
    
    if args.continue_train:
        checkpoint_path = os.path.dirname(config_file)
        if isinstance(config.train.ckp_prefix, str):
            newest_check_point = uni_latest_checkpoint_path(checkpoint_path, config.train.ckp_prefix)
            config.train.model_dir = f"{checkpoint_path}/"
            config.checkpoint.g = f"{checkpoint_path}/{config.train.ckp_prefix}_{newest_check_point}"
            config.checkpoint.load_checkpoint = True
            
            assert os.path.exists(config.checkpoint.g), f"Checkpoint {config.train.ckp_prefix} does not exist"
            
            logger.info(f"Using checkpoint:")
            logger.info(config.checkpoint.g)
        
        else:
            if args.model == 'vits':
                newest_check_point = vits_latest_checkpoint_path(checkpoint_path)
                config.train.model_dir = f"{checkpoint_path}/"
                config.checkpoint.g = f"{checkpoint_path}/G_{newest_check_point}"
                config.checkpoint.d = f"{checkpoint_path}/D_{newest_check_point}"
                config.checkpoint.load_checkpoint = True
                
                assert os.path.exists(config.checkpoint.g), "Checkpoint G does not exist"
                assert os.path.exists(config.checkpoint.d), "Checkpoint D does not exist"
                
                logger.info(f"Using checkpoint:")
                logger.info(config.checkpoint.g)
                logger.info(config.checkpoint.d)
                
            elif args.model == 'bigv':
                newest_check_point = torch_latest_checkpoint_path(checkpoint_path)
                config.train.model_dir = f"{checkpoint_path}/"
                config.checkpoint.g = f"{checkpoint_path}/C_{newest_check_point}"
                config.checkpoint.load_checkpoint = True
                
                assert os.path.exists(config.checkpoint.g), "Checkpoint C does not exist"
                
                logger.info(f"Using checkpoint:")
                logger.info(config.checkpoint.g)
            else:
                newest_check_point = tacotron_latest_checkpoint_path(checkpoint_path)
                config.train.model_dir = f"{checkpoint_path}/"
                config.checkpoint.g = f"{checkpoint_path}/G_{newest_check_point}"
                config.checkpoint.load_checkpoint = True
                
                assert os.path.exists(config.checkpoint.g), "Checkpoint G does not exist"
                
                logger.info(f"Using checkpoint:")
                logger.info(config.checkpoint.g)


            
    elif config.train.test:
        config.train.model_dir = f"./logs/test/"
    else:
        config.train.model_dir = f"./logs/{config.train.model}/{config.train.tags}/{datetime.now().strftime('%m_%d_%H_%M')}/"

    
    try:
        os.makedirs(config.train.model_dir)
    except:
        pass
        
    try:
        shutil.copyfile(config_file, f"{config.train.model_dir}config.toml")
    except shutil.SameFileError:
        pass
    except Exception as e:
        raise e
        
    # logger.info(pprint.pformat(config))
    return config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "rb") as f:
        config = DotMap(tomli.load(f))
    return config


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
            source_dir
        ))
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn("git hash values are different. {}(saved) != {}(current)".format(
                saved_hash[:8], cur_hash[:8]))
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def oc_mem(size = 24000, wait_time = 5, device = 0):
    logger.info(f"Occupy GPU mem with size = {size}, waiting for {wait_time} seconds.")
    x = torch.zeros(256,1024,int(size * 0.9),device=f'cuda:{device}')
    time.sleep(wait_time)
    del x
    time.sleep(wait_time)
    logger.info(f"Occupied GPU mem released.")
    
class TrainingState():
    def __init__(self):
        self.state = {}
        
    def set(self, key, value):
        self.state[key] = value
    
    def get(self, key):
        try:
            return self.state[key]
        except:
            return None
    
    def state_dict(self):
        return self.state
    
    def load_state_dict(self, state_dict):
        self.state = state_dict

class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
