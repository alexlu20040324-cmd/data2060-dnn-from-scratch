"""Test-only mock model with the shared interface (see src/train.py).

One layer, no hidden units (softmax regression). Stands in for the real DNN
until forward/backward are merged.
"""

import numpy as np


class SoftmaxRegression:
    def __init__(self, n_in, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(0, 0.01, size=(n_in, n_out))]
        self.biases = [np.zeros(n_out)]

    def forward(self, X):
        Z = X @ self.weights[0] + self.biases[0]
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        self._X, self._P = X, E / E.sum(axis=1, keepdims=True)
        return self._P

    def loss(self, P, Y):
        eps = np.finfo(np.float64).eps  # same clipping as sklearn's log_loss
        return float(-np.mean(np.sum(Y * np.log(np.clip(P, eps, 1 - eps)), axis=1)))

    def backward(self, Y):
        G = (self._P - Y) / Y.shape[0]
        return [self._X.T @ G], [G.sum(axis=0)]

    def predict(self, X):
        return self.forward(X).argmax(axis=1)
