"""
Utils functions
"""
import torch
import random
import numpy as np
from config import SEED


def set_seed():
    """ Set random seed to all """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """ Truncates a sequence pair in place to the maximum length. """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        tokens_b.pop()
