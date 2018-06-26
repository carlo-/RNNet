#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 4
#
# Carlo Rapisarda (carlora@kth.se)
#

from pathlib import Path
import zipfile
import json
import re
import numpy as np

DATASET_FOLDER = Path('../dataset')


class Dataset:

    def __init__(self, text, separator=None):

        index_to_char = list(set(text))
        index_to_char.sort()

        K = len(index_to_char)
        encoded_chars = np.identity(K)

        char_to_index = {}
        char_to_one_hot = {}
        for i, c in enumerate(index_to_char):
            char_to_index[c] = i
            char_to_one_hot[c] = encoded_chars[i, :]

        self.text = text
        self.K = K
        self.char_to_one_hot = char_to_one_hot
        self.index_to_char = index_to_char
        self.separator = separator

    def seq_to_one_hot(self, seq):
        return np.array([self.char_to_one_hot[x] for x in seq])

    def get_labeled_data(self, offset, seq_len):
        X = self.text[offset : offset + seq_len]
        Y = self.text[offset + 1 : offset + seq_len + 1]
        sep_index = None
        if self.separator is not None and self.separator in X:
            sep_index = offset + X.index(self.separator)
        X = self.seq_to_one_hot(X)
        Y = self.seq_to_one_hot(Y)
        return X, Y, sep_index


def load_goblet_of_fire(limit_chars=None):

    file_path = DATASET_FOLDER / 'goblet_book.txt'

    if not file_path.exists():
        raise FileNotFoundError(f'Goblet of Fire not found at {file_path}')

    with open(file_path, 'r') as f:
        text = f.read()
        if limit_chars is not None:
            text = text[:limit_chars]
        return Dataset(text)
