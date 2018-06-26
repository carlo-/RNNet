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
from utilities import compute_grads_numerical, compare_grads, unpickle, pickle, eprint, simple_smooth_1d

GOBLET_RESULTS_PATH = '../goblet_results.pkl'


def check_gradients():

    book = dt.load_goblet_of_fire()
    seq_len = 25
    m = 5

    X, Y, _ = book.get_labeled_data(0, seq_len)
    h0 = np.zeros((m, 1))

    np.random.seed(42)
    net = RNNet(m=m, K=book.K)

    print('===> Computing numerical gradients...')
    num_grads = compute_grads_numerical(X, Y, h0, net)

    print('===> Computing analytical gradients...')
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
        'epochs': 10,
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
    smooth_losses_by_epoch = res['smooth_losses_by_epoch']

    epochs = len(smooth_losses_by_epoch)
    iters_per_epoch = 1.0 * len(smooth_losses_by_interval) * interval / epochs

    smoother = np.array(smooth_losses_by_interval)
    smoother = simple_smooth_1d(smoother, 0.95)

    fig = plt.figure(figsize=(9, 4))

    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(len(smooth_losses_by_interval)) * interval, smooth_losses_by_interval)
    ax1.plot(np.arange(smoother.size) * interval, smoother)
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss')

    ax2 = ax1.twiny()
    ax2.set_xlabel('epoch')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(np.arange(1,epochs+1) * iters_per_epoch)
    ax2.set_xticklabels(np.arange(1,epochs+1))

    ax2.grid()
    ax1.grid(axis='y')

    fig.tight_layout()
    fig.legend(['training loss', 'smoothed'], bbox_to_anchor=(0.98, 0.86), bbox_transform=fig.transFigure)

    if fig_path is not None:
        fig.savefig(fig_path, bbox_inches='tight')

    fig.show()


def print_evolution(res, interval, limit=None):
    smooth_losses = res['smooth_losses_by_interval']
    synth_samples = res['synthesized_text_by_interval']
    res_interval = res['interval']
    assert interval % res_interval == 0, 'Print interval must be a multiple of the recorded interval'
    selected_indexes = [x for x in range(0, len(synth_samples), interval // res_interval)]
    if limit is not None:
        selected_indexes = selected_indexes[:limit]
    # last_step = selected_indexes[-1] * res_interval
    # print(f'\nModel evolution from step 1 to {last_step}:\n')
    print('\n')
    for i in selected_indexes:
        step = max(i * res_interval, 1)
        text = synth_samples[i]
        smooth_loss = smooth_losses[i]
        print(f'===> Step: {step}, smooth_loss: {round(smooth_loss, 4)}, synthesized:\n{text}\n\n')


def synthesize_with_best_model():
    model_path = '../trained_models/2018-06-12-2205-e10.pkl'
    if exists(model_path):
        book = dt.load_goblet_of_fire()
        net = RNNet.import_model(model_path)
        np.random.seed(50)
        print(net.synthesize(1000, book.char_to_one_hot, book.index_to_char))
    else:
        eprint('Best trained model found!')


def main():

    check_gradients()

    if not exists(GOBLET_RESULTS_PATH):
        train_with_goblet_of_fire(GOBLET_RESULTS_PATH)

    results = unpickle(GOBLET_RESULTS_PATH)

    plot_results(results, '../Report/Figs/training_goblet.eps')

    print_evolution(results, 10_000, 11)

    print(f'===> Passage from the final model (smooth_loss: {results["smooth_losses_by_epoch"][-1]}):')
    synthesize_with_best_model()


if __name__ == '__main__':
    main()
