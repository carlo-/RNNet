#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 4
#
# Carlo Rapisarda (carlora@kth.se)
#

import numpy as np
import pickle
from timeit import default_timer as timer
from dataset import Dataset


class RNNet:


    # ==================== Optimizers ====================

    class AdaGrad:

        def __init__(self, net, eta, eps=1e-8):
            self.net = net
            self.eta = eta
            self.eps = eps

        def step(self):
            theta = vars(self.net.theta)
            grads = vars(self.net.grads)
            momentum = vars(self.net.momentum)
            for key in theta:
                momentum[key] += grads[key] ** 2
                theta[key] -= self.eta * grads[key] * ((momentum[key] + self.eps) ** (-0.5))
            return self.net.grads


    class RMSProp:

        def __init__(self, net, eta, gamma, eps=1e-8):
            self.net = net
            self.eta = eta
            self.gamma = gamma
            self.eps = eps

        def step(self):
            theta = vars(self.net.theta)
            grads = vars(self.net.grads)
            momentum = vars(self.net.momentum)
            for key in theta:
                momentum[key] = self.gamma * momentum[key] + (1 - self.gamma) * (grads[key] ** 2)
                theta[key] -= self.eta * grads[key] * ((momentum[key] + self.eps) ** (-0.5))
            return self.net.grads


    # ==================== Model parameters ====================

    class Theta:

        b = None
        c = None
        U = None
        V = None
        W = None

        @staticmethod
        def he(m, K, mu=0.0, sigma=0.01):
            theta = RNNet.Theta()
            theta.b = np.zeros((m, 1))
            theta.c = np.zeros((K, 1))
            theta.U = sigma * np.random.randn(m, K) + mu
            theta.V = sigma * np.random.randn(K, m) + mu
            theta.W = sigma * np.random.randn(m, m) + mu
            return theta

        @staticmethod
        def zeros(m, K):
            return RNNet.Theta.he(m, K, 0.0, 0.0)

        def keys(self):
            return vars(self).keys()

        def values(self):
            return vars(self).values()

        def copy(self):
            cp = RNNet.Theta()
            cp_v = vars(cp)
            self_v = vars(self)
            for k in self_v:
                cp_v[k] = self_v[k].copy()
            return cp


    # ==================== Initialization ====================

    def __init__(self, m, K, init_theta=True):

        self.m = m
        self.K = K

        if init_theta:
            self.theta = self.Theta.he(m, K)
            self.grads = self.Theta.zeros(m, K)
            self.momentum = self.Theta.zeros(m, K)
        else:
            self.theta = self.Theta()
            self.grads = self.Theta()
            self.momentum = self.Theta()


    # ==================== Import/Export ====================

    @classmethod
    def import_model(cls, filepath):
        with open(filepath, 'rb') as f:
            res = pickle.load(f, encoding='bytes')
        if not isinstance(res, RNNet):
            raise TypeError('File does not exist or is corrupted')
        return res

    def export_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)


    # ==================== FW/BW passes ====================

    def _forward(self, X, h0):

        K, m = self.K, self.m
        b, c, U, V, W = self.theta.values()

        tau = X.shape[0]
        h_prev = h0

        A = np.zeros((tau, m))
        H = np.zeros((tau, m))
        O = np.zeros((tau, K))
        P = np.zeros((tau, K))

        for t in range(tau):

            x_t = X[t].reshape(1,-1)
            h_prev = h_prev.reshape(-1, 1)

            A[t] = (W @ h_prev + U @ x_t.T + b).reshape(-1)
            H[t] = np.tanh(A[t]).reshape(-1)
            O[t] = (V @ H[t].reshape(-1,1) + c).reshape(-1)
            P[t] = self._softmax(O[t]).reshape(-1)

            h_prev = H[t]

        return A, H, O, P


    def _backward(self, X, Y, h0, A, H, O, P):

        tau = Y.shape[0]

        a_last = A[-1]

        d_O = (P - Y)
        d_h_last = d_O[-1] @ self.theta.V
        d_a_last = d_h_last @ np.diag(1.0 - np.tanh(a_last) ** 2.0)

        d_A = np.zeros(A.shape)
        d_H = np.zeros(H.shape)
        d_A[-1] = d_a_last
        d_H[-1] = d_h_last

        for t in range(tau-2, -1, -1):
            d_H[t] = d_O[t] @ self.theta.V + d_A[t+1] @ self.theta.W
            d_A[t] = d_H[t] @ np.diag(1.0 - np.tanh(A[t]) ** 2.0)

        self.grads.U = np.zeros(self.grads.U.shape)
        self.grads.V = np.zeros(self.grads.V.shape)
        self.grads.W = np.zeros(self.grads.W.shape)

        self.grads.W += d_A[0].reshape(-1,1) @ h0.reshape(1,-1)

        for t in range(tau):
            self.grads.U += d_A[t].reshape(-1,1) @ X[t].reshape(1,-1)
            self.grads.V += d_O[t].reshape(-1,1) @ H[t].reshape(1,-1)
            if t > 0:
                self.grads.W += d_A[t].reshape(-1,1) @ H[t-1].reshape(1,-1)

        self.grads.b = d_A.sum(axis=0).reshape(-1,1)
        self.grads.c = d_O.sum(axis=0).reshape(-1,1)

        return self.grads


    # ==================== Loss, Softmax, Utilities ====================

    def synthesize(self, n, char_to_one_hot, index_to_char, h0=None, x0="."):

        if h0 is None:
            h0 = np.zeros((self.m, 1))

        c_prev = x0
        h_prev = h0
        sequence = []

        for i in range(n):

            X = char_to_one_hot[c_prev].reshape(1,-1)
            _, H, _, P = self._forward(X, h_prev)

            cp = np.cumsum(P)
            r = np.random.rand()
            indexes = np.where(cp > r)
            index_next = indexes[0][0]

            c_next = index_to_char[index_next]
            sequence.append(c_next)

            c_prev = c_next
            h_prev = H[-1]

        return "".join(sequence)

    def cross_entropy_loss(self, X, Y, P=None, h_prev=None):
        if P is None:
            _, _, _, P = self._forward(X, h_prev)
        return np.sum(-Y * np.log(P))

    def _clip_gradients(self):
        grads = vars(self.grads)
        for k in grads:
            grads[k] = grads[k].clip(-5.0, 5.0)

    @staticmethod
    def _softmax(s, axis=0):
        exp_s = np.exp(s)
        exp_sum = np.sum(exp_s, axis=axis)
        return exp_s / exp_sum


    # ==================== Gradient descent ====================

    def train(self, dataset: Dataset, config: dict):

        tick_t = timer()

        epochs = config.get('epochs', 10)
        optimizer = config.get('optimizer', self.AdaGrad(self, 0.1))
        seq_len = config.get('sequence_length', 25)
        test_len = config.get('test_length', 100)
        output_folder = config.get('output_folder', None)
        record_interval = config.get('record_interval', 100)
        silent = config.get('silent', False)

        m = self.m
        char_to_one_hot = dataset.char_to_one_hot
        index_to_char = dataset.index_to_char
        text_len = len(dataset.text)

        max_text_index = text_len - seq_len
        total_steps = max_text_index // seq_len * epochs
        smooth_loss = None
        step_num = 0
        times = []
        smooth_losses_e = []
        smooth_losses_i = []
        synthesized_text_i = []

        # For each epoch
        for e in range(1, epochs + 1):

            tick_e = timer()
            h_prev = np.zeros((m, 1))

            # For each sequence
            for i in range(0, max_text_index, seq_len):

                step_num += 1

                X, Y = dataset.get_labeled_data(i, seq_len)
                x0 = dataset.text[i]

                A, H, O, P = self._forward(X, h_prev)
                self._backward(X, Y, h_prev, A, H, O, P)
                optimizer.step()
                self._clip_gradients()

                loss = self.cross_entropy_loss(X, Y, h_prev=h_prev)

                if step_num == 1:
                    smooth_loss = loss
                else:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss

                if step_num % record_interval == 0 or step_num == 1:
                    synt = self.synthesize(test_len, char_to_one_hot, index_to_char, h_prev, x0)
                    smooth_losses_i.append(smooth_loss)
                    synthesized_text_i.append(synt)
                    if not silent:
                        max_grad_mag = max([x.max() - x.min() for x in self.grads.values()])
                        print(f'===> Epoch {e}/{epochs}, step {step_num}/{total_steps}, smooth_loss: {smooth_loss}, max_grad_mag: {max_grad_mag}')
                        print(synt, '\n')

                h_prev = H[-1]

            smooth_losses_e.append(smooth_loss)

            if output_folder is not None:
                filepath = f'{output_folder}/model_epoch_{e}.pkl'
                self.export_model(filepath)

            if not silent:
                tock_e = timer()
                interval = tock_e - tick_e
                times.append(interval)
                rem = (epochs - e) * np.mean(times[-3:])
                print(f'===> Epoch {e} completed, {int(round(rem))}s remaining')

        if not silent:
            tock_t = timer()
            print("Done. Took ~{}s".format(round(tock_t - tick_t)))

        return {
            "interval": record_interval,
            "smooth_losses_by_epoch": smooth_losses_e,
            "smooth_losses_by_interval": smooth_losses_i,
            "synthesized_text_by_interval": synthesized_text_i
        }
