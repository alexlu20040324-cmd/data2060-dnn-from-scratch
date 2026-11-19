"""Vanilla mini-batch SGD trainer (Member 3).

No momentum, no L2, constant learning rate -- matches the baseline and
sklearn's MLPClassifier(solver="sgd", momentum=0, alpha=0,
learning_rate="constant").

Model interface this trainer expects (shared v1 interface):
    model.weights : list of arrays, weights[l] has shape (n_in, n_out)
    model.biases  : list of arrays, biases[l]  has shape (n_out,)
    model.forward(X)    -> P, softmax probabilities (n, K); caches what backward needs
    model.loss(P, Y)    -> float, mean categorical cross-entropy (Y one-hot)
    model.backward(Y)   -> (dW, db), lists with the same structure as weights/biases
    model.predict(X)    -> (n,) integer class labels
"""

from __future__ import annotations

import numpy as np


def iterate_minibatches(n_samples: int, batch_size: int, rng=None, shuffle: bool = True):
    """Yield index arrays that cover range(n_samples) exactly once.

    The last batch may be smaller than batch_size (same as sklearn).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    idx = np.arange(n_samples)
    if shuffle:
        if rng is None:
            raise ValueError("rng is required when shuffle=True")
        rng.shuffle(idx)
    for start in range(0, n_samples, batch_size):
        yield idx[start:start + batch_size]


def sgd_step(params, grads, lr: float) -> None:
    """In-place vanilla SGD update: p <- p - lr * g for each parameter."""
    if lr <= 0:
        raise ValueError("learning rate must be positive")
    if len(params) != len(grads):
        raise ValueError(f"got {len(params)} parameters but {len(grads)} gradients")
    for p, g in zip(params, grads):
        if p.shape != g.shape:
            raise ValueError(f"parameter shape {p.shape} != gradient shape {g.shape}")
        p -= lr * g


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred have different shapes")
    return float(np.mean(y_true == y_pred))


def train(model, X, Y, *, epochs: int = 100, batch_size: int = 32, lr: float = 0.01,
          seed: int = 2060, shuffle: bool = True, X_val=None, Y_val=None, verbose: bool = False):
    """Train `model` with mini-batch SGD.

    Parameters
    ----------
    X : (n, d) features; Y : (n, K) one-hot labels.
    epochs, batch_size, lr : SGD hyperparameters (batch_size >= n gives full-batch GD).
    seed : controls the per-epoch shuffling only (weight init is the model's job).
    X_val, Y_val : optional validation set, evaluated after every epoch.

    Returns
    -------
    history : dict of per-epoch lists
        "train_loss" -- sample-weighted mean of the mini-batch losses seen during
                        the epoch (same definition as sklearn's loss_curve_)
        "train_acc"  -- accuracy on the full training set after the epoch
        "val_loss", "val_acc" -- only if a validation set is given
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D (Y must be one-hot)")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y have different numbers of rows")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if (X_val is None) != (Y_val is None):
        raise ValueError("pass both X_val and Y_val, or neither")

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    y_int = Y.argmax(axis=1)
    history = {"train_loss": [], "train_acc": []}
    if X_val is not None:
        history["val_loss"], history["val_acc"] = [], []

    for epoch in range(epochs):
        total = 0.0
        for idx in iterate_minibatches(n, batch_size, rng, shuffle):
            Xb, Yb = X[idx], Y[idx]
            P = model.forward(Xb)
            batch_loss = model.loss(P, Yb)
            if not np.isfinite(batch_loss):
                raise FloatingPointError(f"loss became {batch_loss} at epoch {epoch + 1}; try a smaller lr")
            total += batch_loss * len(idx)
            dW, db = model.backward(Yb)
            sgd_step(model.weights, dW, lr)
            sgd_step(model.biases, db, lr)

        history["train_loss"].append(total / n)
        history["train_acc"].append(accuracy(y_int, model.predict(X)))
        if X_val is not None:
            history["val_loss"].append(float(model.loss(model.forward(X_val), Y_val)))
            history["val_acc"].append(accuracy(Y_val.argmax(axis=1), model.predict(X_val)))

        if verbose:
            msg = f"epoch {epoch + 1:4d}  loss {history['train_loss'][-1]:.4f}  acc {history['train_acc'][-1]:.4f}"
            if X_val is not None:
                msg += f"  val_loss {history['val_loss'][-1]:.4f}  val_acc {history['val_acc'][-1]:.4f}"
            print(msg)

    return history
