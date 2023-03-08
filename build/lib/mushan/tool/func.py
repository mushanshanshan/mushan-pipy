from typing import List
import torch
import os
import logging
import pickle
import random

def disable_logger(logger_name: List):
    for n in logger_name:
        logger = logging.getLogger(n)
        logger.propagate = False
        
        
def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def load_text(filename):
    with open(filename, "r") as f:
        text = f.readlines()
    return text


def dump_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        

def random_str(length):
    return os.urandom(length).hex()