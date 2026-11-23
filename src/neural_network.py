"""Fully-connected NumPy neural network: Member A's initialization/inference.

Conventions shared with src/train.py:
  weights[l]: (n_in, n_out); biases[l]: (n_out,)
  X: (batch, n_features); forward(X): (batch, n_classes)
  activations: [X, A1, ..., P]; pre_activations: [Z1, ..., ZL]

Loss/backward are intentionally owned by Member B and are NOT implemented here.
"""
from __future__ import annotations

import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, random_state: int = 42):
        if not isinstance(layer_sizes, (list, tuple)) or len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output sizes")
        if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0
               for n in layer_sizes):
            raise ValueError("Every layer size must be a positive integer")
        self.layer_sizes = list(layer_sizes)
        rng = np.random.default_rng(random_state)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out, dtype=np.float64))
        self.activations = []
        self.pre_activations = []

    @staticmethod
    def relu(z):
        return np.maximum(0.0, np.asarray(z, dtype=np.float64))

    @staticmethod
    def sigmoid(z):
        """Stable sigmoid, including large positive/negative inputs."""
        z = np.asarray(z, dtype=np.float64)
        out = np.empty_like(z)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        e = np.exp(z[~positive])
        out[~positive] = e / (1.0 + e)
        return out

    @staticmethod
    def softmax(z):
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] == 0 or z.shape[0] == 0:
            raise ValueError("softmax expects a nonempty 2-D matrix")
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"X must have shape (batch, {self.layer_sizes[0]})")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        self.activations = [X]
        self.pre_activations = []
        A = X
        last = len(self.weights) - 1
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            self.pre_activations.append(Z)
            A = self.softmax(Z) if l == last else self.relu(Z)
            self.activations.append(A)
        return A

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def loss(self, P, Y):
        raise NotImplementedError("Member B will integrate mean cross-entropy loss")

    def backward(self, Y):
        raise NotImplementedError("Member B will integrate backpropagation")
