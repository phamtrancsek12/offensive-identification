"""
This file contains the script to preprocess Tweets data
"""
import re
import emoji
import wordsegment
import pandas as pd
from tqdm import tqdm
from itertools import groupby
from argparse import ArgumentParser
from utils import *


def normalize_duplicate_letter(word, vocab):
    """
    In tweets, user option uses duplicate letters to emphasize the word
    e.g: lovelyyyyyyy, cuteeeeee, goooood
    """
    normalize_word = word
    if word not in vocab:
        if word in microtext:
            normalize_word = convert_microtext(word)
        else:
            # English word can have 2 duplicate vowels
            remove_dup_2 = count_duplicate(word, 2)
            if remove_dup_2 not in vocab:
                remove_dup_1 = count_duplicate(remove_dup_2, 1)
                if remove_dup_1 not in vocab:
                    if remove_dup_1 in microtext:
                        normalize_word = convert_microtext(remove_dup_1)
                    else:
                        normalize_word = remove_dup_2
                        if remove_dup_2 in microtext:
                            normalize_word = convert_microtext(remove_dup_2)
                else:
                    normalize_word = remove_dup_1
            else:
                normalize_word = remove_dup_2

    return normalize_word


def preprocess(text, vocab):
    """
    Perform all steps of Tweets pre-processing: lower case, convert emoji,
                        convert microtext, normalize duplicate letter, etc.
    """
    text = text.lower()
    text = emoji.demojize(text)
    text = remove_nonlatin(text)
    # Remove duplicate words (such as "@user @user @user")
    text = " ".join([k for k, v in groupby(text.split())])
    # Remove symbols
    text = re.sub('(?<! )(?=[.,!?():])|(?<=[.,!?():])(?! )', r' ', text)

    normalized = []
    for phrase in text.split():
        # This's, I'm, etc.
        if "'" in phrase:
            normalized.append(phrase)
            continue
        # Segment hashtags
        if "#" in phrase:
            phrase = phrase.replace("#", "")
            words = wordsegment.segment(phrase)
            normalized.extend(words)
            continue
        normalized.append(phrase)

    normalized = [normalize_duplicate_letter(w, vocab) for w in normalized]
    return ' '.join(normalized)


def main():
    """ Main function """

    parser = ArgumentParser()
    parser.add_argument('--data_file', dest="data_file", required=True)
    parser.add_argument('--save_file', dest="save_file", required=True)
    parser.add_argument('--vocab_file', dest="vocab_file", required=True)
    args = vars(parser.parse_args())

    vocab = load_BERT_vocab(args["vocab_file"])
    wordsegment.load()
    tqdm.pandas()
    df = pd.read_csv(args["data_file"], sep="\t")
    df["text"] = df["text"].progress_apply(lambda x: preprocess(x, vocab))
    df.to_csv(args["save_file"], index=False)


if __name__ == '__main__':
    main()