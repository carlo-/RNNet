#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 4
#
# Carlo Rapisarda (carlora@kth.se)
#

from pathlib import Path
import numpy as np

DATASET_FOLDER = Path('../dataset')


class Dataset:

    def __init__(self, text):

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

    def seq_to_one_hot(self, seq):
        return np.array([self.char_to_one_hot[x] for x in seq])

    def get_labeled_data(self, offset, seq_len):
        X = self.seq_to_one_hot(self.text[offset : offset + seq_len])
        Y = self.seq_to_one_hot(self.text[offset + 1 : offset + seq_len + 1])
        return X, Y


def load_goblet_of_fire(limit_chars=None):
    file_path = DATASET_FOLDER / 'goblet_book.txt'
    with open(file_path, 'r') as f:
        text = f.read()
        if limit_chars is not None:
            text = text[:limit_chars]
        return Dataset(text)
