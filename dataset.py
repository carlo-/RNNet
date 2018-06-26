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


def load_trump_tweets(limit_chars=None):

    dir_path = DATASET_FOLDER / 'trump_tweet_data_archive-master'

    if not dir_path.exists():
        raise FileNotFoundError(f'trump_tweet_data_archive must be cloned into {dir_path}')

    def get_jsons():
        return [x for x in dir_path.glob('condensed*.json')]
    def get_zips():
        return [x for x in dir_path.glob('condensed*.zip')]
    def strip_urls(text):
        return re.sub(r"http\S+", "", text)

    if len(get_jsons()) == 0:
        zip_paths = get_zips()
        if len(zip_paths) == 0:
            raise FileNotFoundError()
        for zip_path in zip_paths:
            zip_ref = zipfile.ZipFile(zip_path, 'r')
            zip_ref.extractall(dir_path)
            zip_ref.close()

    tweets = []

    json_paths = get_jsons()
    for json_path in json_paths:
        with open(json_path, 'rb') as f:
            tweets += json.load(f)

    # filter out retweets
    tweets = [x for x in tweets if not x['is_retweet']]

    # filter out clipped tweets
    tweets = [x for x in tweets if '(cont)' not in x['text']]

    # text content only
    tweets_as_text = [x['text'] for x in tweets]

    # strip out urls
    tweets_as_text = [strip_urls(x) for x in tweets_as_text]

    # randomize order
    np.random.shuffle(tweets_as_text)

    special_char = "◊"
    assert len([x for x in tweets_as_text if special_char in x]) == 0, 'Special separator character found in tweets!'

    raw_text = special_char.join(tweets_as_text)

    if limit_chars is not None:
        raw_text = raw_text[:limit_chars]

    return Dataset(raw_text, separator=special_char)
