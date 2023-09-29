from typing import List
import torch
import os
import pickle
import lilcom
import lzma

def from_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def from_text(filename):
    with open(filename, "r") as f:
        text = f.readlines()
    return text

def to_text(filename, data):
    with open(filename, "w") as f:
        for d in data:
            f.write(d)

def to_pickle(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        
        
def to_lil(filename, data, tick_power=-5):
    with open(filename, "wb") as f:
        f.write(lilcom.compress(data, tick_power=tick_power))
        
        
def from_lil(filename):
    with open(filename, "rb") as f:
        data = lilcom.decompress(f.read())
    return data


def to_lzma(filename, data):
    with lzma.open(filename, "wb") as f:
        pickle.dump(data, f)
        
        
def from_lzma(filename):
    with lzma.open(filename, "rb") as f:
        data = pickle.load(f)
    return data