#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 4
#
# Carlo Rapisarda (carlora@kth.se)
#

import re
import numpy as np
import dataset as dt
from joblib import Parallel, delayed
from collections import Counter
from os.path import exists
from model import RNNet
from utilities import unpickle, pickle


def synthesize_trump(model_path):
    np.random.seed(42)
    tweets = dt.load_trump_tweets()
    net = RNNet.import_model(model_path)
    print(net.synthesize(140*100, tweets.char_to_one_hot, tweets.index_to_char, separator=tweets.separator, stop_at_separator=False))


def synthesize_special_seq(net: RNNet, dataset: dt.Dataset, x0, random_state=None, max_len: int = 40, min_len: int = 3) -> str:
    while True:
        synth = net.synthesize(max_len, dataset.char_to_one_hot, dataset.index_to_char, x0=x0, separator=dataset.separator, random_state=random_state)
        single_piece = synth.split(' ')[0]
        valid_words = re.findall(r'(\w+)', single_piece)
        if len(valid_words) > 0 and len(valid_words[0]) >= min_len:
            return x0 + valid_words[0]


def synthesize_special_seq_worker(n, net, dataset, x0, seed):
    random_state = np.random.RandomState(seed)
    return [synthesize_special_seq(net, dataset, x0, random_state) for _ in range(n)]


def synthesize_trump_special_seq(model_path, n=100, x0='#'):
    np.random.seed(42)
    tweets = dt.load_trump_tweets()
    net = RNNet.import_model(model_path)

    n_jobs = -1
    parallel = Parallel(n_jobs=n_jobs, verbose=5, backend='multiprocessing')
    res = parallel(delayed(synthesize_special_seq_worker)(n//n_jobs, net, tweets, x0, 42+i) for i in range(n_jobs))
    res = sum(res, [])

    res.sort()
    return res


def train_with_trump_tweets(output_dir, results_path=None):

    np.random.seed(42)

    tweets = dt.load_trump_tweets()
    net = RNNet(m=100, K=tweets.K)

    optimizer = RNNet.RMSProp(net, eta=0.0008, gamma=0.9)

    config = {
        'epochs': 5,
        'output_folder': output_dir,
        'optimizer': optimizer,
        'sequence_length': 70,
        'record_interval': 200,
        'test_length': 140
    }

    res = net.train(tweets, config)

    if results_path is not None:
        pickle(res, results_path)

    return res


def fine_tune_trump_tweets(output_dir, initial_model_path, results_path=None):

    np.random.seed(42)

    tweets = dt.load_trump_tweets()
    net = RNNet.import_model(initial_model_path)
    net.clear_grads()

    optimizer = RNNet.RMSProp(net, eta=0.0001, gamma=0.9)

    config = {
        'epochs': 5,
        'output_folder': output_dir,
        'optimizer': optimizer,
        'sequence_length': 70,
        'record_interval': 200,
        'test_length': 140
    }

    res = net.train(tweets, config)

    if results_path is not None:
        pickle(res, results_path)

    return res


def analyze_trump_seq(model_path, seq_type, working_dir="."):

    if seq_type == 'hashtags':
        x0 = '#'
    elif seq_type == 'mentions':
        x0 = '@'
    elif seq_type == 'words':
        x0 = ' '
    else:
        raise NotImplemented()

    sequences_fp = f'{working_dir}/{seq_type}.pkl'
    if not exists(sequences_fp):
        sequences = synthesize_trump_special_seq(model_path, 200_000, x0)
        pickle(sequences, sequences_fp)
    else:
        sequences = unpickle(sequences_fp)

    print(f'Found {len(sequences)} {seq_type}')

    with open(f'{working_dir}/{seq_type}.txt', 'w+') as f:
        for ht in sequences:
            print(ht, file=f)

    counter = Counter(sequences)
    most_common = counter.most_common(500)
    with open(f'{working_dir}/top500_{seq_type}.txt', 'w+') as f:
        for ht_c in most_common:
            print(f'{ht_c[0]} (x{ht_c[1]})', file=f)

    sequences_lc = [x.lower() for x in sequences]
    counter = Counter(sequences_lc)
    most_common = counter.most_common(500)
    with open(f'{working_dir}/top500_lc_{seq_type}.txt', 'w+') as f:
        for ht_c in most_common:
            print(f'{ht_c[0]} (x{ht_c[1]})', file=f)


def main():

    working_dir = '../out'
    best_model_path = '../trained_models/2018-06-16-1500-e5.pkl'

    # train_with_trump_tweets(output_dir)
    # fine_tune_trump_tweets(output_dir, '../trained_models/2018-06-16-1500-e1.pkl')
    synthesize_trump(best_model_path)

    analyze_trump_seq(best_model_path, 'hashtags', working_dir=working_dir)
    analyze_trump_seq(best_model_path, 'mentions', working_dir=working_dir)
    analyze_trump_seq(best_model_path, 'words', working_dir=working_dir)


if __name__ == '__main__':
    main()
