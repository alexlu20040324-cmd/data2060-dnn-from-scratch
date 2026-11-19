"""sklearn MLPClassifier reference harness (Member 3).

Builds an MLPClassifier configured to match our NumPy baseline, and compares
the two under controlled conditions.

Two comparison modes:
  * exact      (shuffle=False): both models start from the SAME initial
               parameters and see the SAME mini-batches in the same order,
               so parameters and loss curves should agree to ~1e-10.
  * statistical (shuffle=True): each library shuffles with its own RNG, so
               trajectories differ; compare final loss / accuracy instead.

sklearn is used only here (reference side), never inside our model.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from train import accuracy, train


def make_sklearn_mlp(hidden_layer_sizes=(64, 32), *, lr=0.01, batch_size=32, epochs=100,
                     seed=2060, shuffle=True) -> MLPClassifier:
    """MLPClassifier with the same settings as our baseline.

    ReLU hidden layers + softmax output + mean cross-entropy, plain SGD
    (no momentum, no Nesterov), no L2 (alpha=0), constant learning rate,
    and no early stopping, so it always runs exactly `epochs` epochs.
    """
    return MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation="relu",
        solver="sgd",
        alpha=0.0,
        batch_size=batch_size,
        learning_rate="constant",
        learning_rate_init=lr,
        momentum=0.0,
        nesterovs_momentum=False,
        max_iter=epochs,
        shuffle=shuffle,
        random_state=seed,
        early_stopping=False,
        tol=0.0,
        n_iter_no_change=epochs + 1,  # never stop early
    )


def glorot_uniform_init(layer_sizes, seed=2060):
    """Initial parameters drawn the same way sklearn does (Glorot uniform, ReLU).

    W[l], b[l] ~ U(-a, a), a = sqrt(6 / (fan_in + fan_out)).
    layer_sizes: e.g. [16, 64, 32, 7].  Returns (weights, biases) lists.
    Uses np.random.RandomState(seed) in sklearn's draw order, so the result
    equals MLPClassifier(random_state=seed)'s own initialization.
    """
    if len(layer_sizes) < 2:
        raise ValueError("need at least input and output sizes")
    rng = np.random.RandomState(seed)
    weights, biases = [], []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        weights.append(rng.uniform(-bound, bound, (fan_in, fan_out)))
        biases.append(rng.uniform(-bound, bound, fan_out))
    return weights, biases


def fit_sklearn_from_params(mlp: MLPClassifier, X, y, weights, biases) -> MLPClassifier:
    """Train `mlp` starting from the given initial parameters instead of its own.

    y: integer labels. Uses warm_start: a throw-away 1-epoch fit creates
    sklearn's internal state, then we overwrite the parameters and reset the
    training history before the real fit.
    """
    expected = [X.shape[1], *mlp.hidden_layer_sizes, len(np.unique(y))]
    shapes = [w.shape for w in weights]
    if shapes != list(zip(expected[:-1], expected[1:])):
        raise ValueError(f"weight shapes {shapes} do not match layer sizes {expected}")
    if [b.shape for b in biases] != [(n,) for n in expected[1:]]:
        raise ValueError("bias shapes do not match layer sizes")

    epochs = mlp.max_iter
    mlp.set_params(warm_start=True, max_iter=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit(X, y)  # set-up pass only; its result is discarded below

        mlp.coefs_ = [np.array(w, dtype=np.float64, copy=True) for w in weights]
        mlp.intercepts_ = [np.array(b, dtype=np.float64, copy=True) for b in biases]
        mlp.loss_curve_ = []
        mlp.best_loss_ = np.inf
        mlp._no_improvement_count = 0

        mlp.set_params(max_iter=epochs)
        mlp.fit(X, y)
    mlp.set_params(warm_start=False)
    return mlp


def compare_with_sklearn(model, X_train, Y_train, X_test, y_test, *, hidden_layer_sizes=(64, 32),
                         lr=0.01, batch_size=32, epochs=100, seed=2060, shuffle=False):
    """Train our model and a matched sklearn MLP from the same initial parameters.

    `model` must follow the interface in train.py; its current weights/biases
    are used as the shared starting point. Returns a report dict.
    """
    init_W = copy.deepcopy(model.weights)
    init_b = copy.deepcopy(model.biases)
    y_train = np.asarray(Y_train).argmax(axis=1)

    ours = train(model, X_train, Y_train, epochs=epochs, batch_size=batch_size, lr=lr,
                 seed=seed, shuffle=shuffle)

    mlp = make_sklearn_mlp(hidden_layer_sizes, lr=lr, batch_size=batch_size, epochs=epochs,
                           seed=seed, shuffle=shuffle)
    fit_sklearn_from_params(mlp, X_train, y_train, init_W, init_b)

    ours_pred = model.predict(X_test)
    sk_pred = mlp.predict(X_test)
    param_diff = max(
        float(np.max(np.abs(a - b)))
        for a, b in zip(model.weights + model.biases, mlp.coefs_ + mlp.intercepts_)
    )
    return {
        "mode": "statistical" if shuffle else "exact",
        "ours_loss_curve": ours["train_loss"],
        "sklearn_loss_curve": list(mlp.loss_curve_),
        "max_loss_curve_diff": float(np.max(np.abs(np.subtract(ours["train_loss"], mlp.loss_curve_)))),
        "max_param_diff": param_diff,
        "ours_test_acc": accuracy(y_test, ours_pred),
        "sklearn_test_acc": accuracy(y_test, sk_pred),
        "prediction_agreement": accuracy(ours_pred, sk_pred),
        "sklearn_model": mlp,
    }


def print_report(report) -> None:
    print(f"mode: {report['mode']}")
    print(f"final train loss  ours {report['ours_loss_curve'][-1]:.6f}  "
          f"sklearn {report['sklearn_loss_curve'][-1]:.6f}")
    print(f"max |loss diff| over epochs: {report['max_loss_curve_diff']:.2e}")
    print(f"max |param diff|:            {report['max_param_diff']:.2e}")
    print(f"test acc  ours {report['ours_test_acc']:.4f}  sklearn {report['sklearn_test_acc']:.4f}")
    print(f"prediction agreement: {report['prediction_agreement']:.4f}")
