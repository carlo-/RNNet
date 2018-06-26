#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 4
#
# Carlo Rapisarda (carlora@kth.se)
#

import numpy as np
from sys import stderr
from model import RNNet
Theta = RNNet.Theta


def eprint(*args, **kwargs):
    print(*args, file=stderr, **kwargs)


def unpickle(filename):
    import pickle
    with open(filename, 'rb') as f:
        res = pickle.load(f, encoding='bytes')
    return res


def pickle(obj, filename):
    import pickle as pickle_
    with open(filename, 'wb') as f:
        pickle_.dump(obj, f)


def _compute_grads_numerical(X, Y, m, K, theta, loss_fn, h):

    grads = Theta.zeros(m, K)
    grads_v = vars(grads)
    theta = vars(theta)

    for k in theta:

        for i in range(theta[k].size):

            theta[k].itemset(i, theta[k].item(i) - h)
            l1 = loss_fn(X, Y)
            theta[k].itemset(i, theta[k].item(i) + h)

            theta[k].itemset(i, theta[k].item(i) + h)
            l2 = loss_fn(X, Y)
            theta[k].itemset(i, theta[k].item(i) - h)

            grads_v[k].itemset(i, (l2 - l1) / (2.0 * h))

    return grads


def compute_grads_numerical(X, Y, h0, net: RNNet, step_size=1e-5):

    old_theta = net.theta
    tmp_theta = old_theta.copy()
    m, K = net.m, net.K

    net.theta = tmp_theta

    def loss_fn(X_, Y_):
        return net.cross_entropy_loss(X_, Y_, h_prev=h0)

    grads = _compute_grads_numerical(X, Y, m, K, tmp_theta, loss_fn, step_size)

    net.theta = old_theta
    return grads


def relative_err(a,b,eps=1e-12):
    assert a.shape == b.shape
    return np.abs(a-b) / np.maximum(eps, np.abs(a)+np.abs(b))


def compare_grads(lhs: Theta, rhs: Theta, m, K):
    errors = Theta.zeros(m, K)
    errors_v = vars(errors)
    lhs = vars(lhs)
    rhs = vars(rhs)
    for k in lhs:
        errors_v[k] = relative_err(lhs[k], rhs[k])
    return errors


def simple_smooth_1d(x, alpha):
    assert len(x.shape) == 1, 'Function only works with 1D arrays'
    smooth_x = np.zeros(x.shape[0])
    smooth_x[0] = x[0]
    for i in range(1, smooth_x.size):
        smooth_x[i] = alpha * smooth_x[i-1] + (1.0 - alpha) * x[i]
    return smooth_x
