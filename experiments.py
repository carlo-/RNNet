#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 4
#
# Carlo Rapisarda (carlora@kth.se)
#

import numpy as np
import matplotlib.pyplot as plt
import dataset as dt
from os.path import exists
from model import RNNet
from utilities import compute_grads_numerical, compare_grads, unpickle, pickle

GOBLET_RESULTS_PATH = '../goblet_results.pkl'


def check_gradients():

    book = dt.load_goblet_of_fire()
    seq_len = 25
    m = 5

    X, Y = book.get_labeled_data(0, seq_len)
    h0 = np.zeros((m, 1))

    np.random.seed(42)
    net = RNNet(m=m, K=book.K)

    print('Computing numerical gradients...')
    num_grads = compute_grads_numerical(X, Y, h0, net)

    print('Computing analytical gradients...')
    grads = net._backward(X, Y, h0, *net._forward(X, h0))

    errors = compare_grads(num_grads, grads, m, book.K)
    errors_v = vars(errors)
    for k in errors_v:
        v = errors_v[k]
        print(f'MSEs for {k} -> max: {v.max()},\t avg: {v.mean()},\t std: {v.std()}')


def train_with_goblet_of_fire(results_path=None):

    book = dt.load_goblet_of_fire()

    np.random.seed(42)
    net = RNNet(m=100, K=book.K)

    # optimizer = RNNet.AdaGrad(net, eta=0.1)
    optimizer = RNNet.RMSProp(net, eta=0.001, gamma=0.9)

    config = {
        'epochs': 20,
        'output_folder': '../out',
        'optimizer': optimizer,
        'sequence_length': 25,
        'record_interval': 1_000,
        'test_length': 200
    }

    res = net.train(book, config)

    if results_path is not None:
        pickle(res, results_path)

    return res


def plot_results(res, fig_path=None):

    interval = res['interval']
    smooth_losses_by_interval = res['smooth_losses_by_interval']
    # smooth_losses_by_epoch = res['smooth_losses_by_epoch']

    plt.figure(figsize=(9, 4))

    plt.plot(np.arange(len(smooth_losses_by_interval)) * interval, smooth_losses_by_interval)
    plt.legend(['training loss'])
    plt.xlabel('step')
    plt.ylabel('loss')

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')

    plt.show()


def synthesize_with_best_model():
    book = dt.load_goblet_of_fire()
    net = RNNet.import_model('../trained_models/2018-06-12-1600-e50.pkl')
    print(net.synthesize(1000, book.char_to_one_hot, book.index_to_char))


if __name__ == '__main__':

    # check_gradients()

    if not exists(GOBLET_RESULTS_PATH):
        train_with_goblet_of_fire(GOBLET_RESULTS_PATH)

    results = unpickle(GOBLET_RESULTS_PATH)
    plot_results(results, '../Report/Figs/training_goblet.eps')
