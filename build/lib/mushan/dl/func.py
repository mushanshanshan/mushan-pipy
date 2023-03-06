import torch
import os
from varname import argname
import torch.nn as nn
from inspect import getframeinfo, stack

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def enable_debug_mode():
    os.environ['MUSHAN_DEBUG'] = "1"
    print("Debug model: Enable")

def check_debug_mode():
    return os.getenv('MUSHAN_DEBUG', '0') == "1"

def disable_debug_mode():
    os.environ['MUSHAN_DEBUG'] = "0"
    print("Debug model: Disable")


def check_no_grad_parameters(model: nn.Module):
    """
    This function checks and prints the names of parameters in the model that do not have gradients (requires_grad=False).
    
    Args:
        model (nn.Module): The PyTorch model to check.
    
    Returns:
        List of parameter names that do not have gradients.
    """
    no_grad_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            no_grad_params.append(name)
    
    if no_grad_params:
        print("Parameters with no gradients (requires_grad=False):")
        for param_name in no_grad_params:
            print(param_name)
    else:
        print("All parameters have gradients (requires_grad=True).")
    
    return no_grad_params
    
def debug_shape(*args):
    if not check_debug_mode():
        return
    for i in range(len(args)):
        assert isinstance(args[i], torch.Tensor)
        print(f"{argname(f'args[{i}]')}.shape: {str(list(args[i].shape))}")
        
def debug_nan(*args):
    for i in range(len(args)):
        assert isinstance(args[i], torch.Tensor)
        if torch.isnan(args[i]).any():
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
    
    
def weight_init(m):
    import torch.nn.init as init
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    
