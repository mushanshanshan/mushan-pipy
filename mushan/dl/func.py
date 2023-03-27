import torch
import os
from varname import argname


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def enable_debug_mode():
    os.environ['MUSHAN_DEBUG']= "1"
    print("Debug model: Enable")

def check_debug_mode():
    return os.getenv('path') == "1"

def disable_debug_mode():
    os.environ['MUSHAN_DEBUG']= "0"
    print("Debug model: Disable")
    
    
def debug_shape(*args):
    for i in range(len(args)):
        assert isinstance(args[i], torch.Tensor)
        print(f"{argname(f'args[{i}]')}.shape: {str(list(args[i].shape))}, {str(args[i].dtype)[6:]}")


def print_shape(*args):
    for i in range(len(args)):
        assert isinstance(args[i], torch.Tensor)
        print(f"{argname(f'args[{i}]')}.shape: {str(list(args[i].shape))}, {str(args[i].dtype)[6:]}")

def get_device():
    
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def disable_cuda(args=None):
    
    
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    if torch.cuda.is_available():
        print("Disable CUDA fail!")
    else:
        print("Disable CUDA success!")
        
        
def set_cuda(gpus=None):
    """_summary_

    Args:
        gpus (int, list): _description_
    """
    
    if gpus == None or gpus == -1:
        disable_cuda()
    else:
        _gpus = []
        if isinstance(gpus, list):
            for g in gpus:
                _gpus.append(str(g))
        elif isinstance(gpus, int):
            _gpus.append(str(gpus))
        else:
            print("Unknow input types!")
            return
            
        os.environ["CUDA_VISIBLE_DEVICES"]=",".join(_gpus)
        
        print("Current CUDA Devices: {}".format(torch.cuda.current_device()))
        print("Total Visible CUDA Device Count: {}".format(torch.cuda.device_count()))
    
    
