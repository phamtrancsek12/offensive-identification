"""
Utils functions
"""
import re
from microtext import microtext

def load_BERT_vocab(vocab_file):
    """
    Load BERT vocab to use when ranking OOV words
    """
    with open(vocab_file) as f:
        vocab = f.readlines()
        vocab = [x.strip() for x in vocab]
    return vocab

def convert_microtext(w):
    """
    Convert microtext using list of microtext from https://github.com/npuliyang/Microtext Normalization/
    """
    w = microtext[w][0]
    w = ' '.join(w.split("_"))
    return w

def count_duplicate(word, max_i):
    """
    Count duplicate letter in word
    """
    word += "  "
    new_word = ""
    for i in range(len(word) - max_i):
        if not (word[i] == word[i + 1] == word[i + max_i]):
            new_word += word[i]
    return new_word

def remove_nonlatin(text):
    text = re.sub('<[^>]*>', ' ', text)
    text = re.sub('[^A-Za-z0-9.,;()@\ ]+', '', text)
    return " ".join(text.split())