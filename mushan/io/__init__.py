from typing import List
import torch
import os
import pickle
import lilcom
import lzma
from dotmap import DotMap
import tomli
import hashlib

def make_parent_dir(f):
    os.makedirs(os.path.dirname(f), exist_ok = True)

def str_to_int(s):
    # 使用md5哈希函数获取字符串的哈希值
    hash_object = hashlib.md5(s.encode())
    # 取哈希值的十六进制表示的最后一个字符，然后对10取模
    digit = int(hash_object.hexdigest(), 16) % 10
    return digit

def from_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def from_toml(filename):
    with open(filename, "rb") as f:
        config = DotMap(tomli.load(f))
    return config

def from_text(filename):
    with open(filename, "r") as f:
        text = f.readlines()
    return text

def to_text(filename, data):
    assert isinstance(filename, str)
    with open(filename, "w", encoding='utf-8') as f:
        for d in data:
            if len(data) > 1 and d[-1] != '\n':
                d += '\n'
            f.write(d)

def to_pickle(filename, data):
    assert isinstance(filename, str)
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        
        
def to_lil(filename, data, tick_power=-5):
    assert isinstance(filename, str)
    with open(filename, "wb") as f:
        f.write(lilcom.compress(data, tick_power=tick_power))
        
        
def from_lil(filename):
    with open(filename, "rb") as f:
        data = lilcom.decompress(f.read())
    return data


def to_lzma(filename, data):
    assert isinstance(filename, str)
    with lzma.open(filename, "wb") as f:
        pickle.dump(data, f)
        
        
def from_lzma(filename):
    with lzma.open(filename, "rb") as f:
        data = pickle.load(f)
    return data