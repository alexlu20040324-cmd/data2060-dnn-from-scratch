import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import one_hot, prepare_dry_bean  # noqa: E402
from train import accuracy, iterate_minibatches, sgd_step, train  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mock_models import SoftmaxRegression  # noqa: E402


def toy_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], n // 3)
    centers = np.array([[0, 3], [3, -3], [-3, -3]], dtype=float)
    X = centers[y] + rng.normal(size=(n, 2))
    return X, one_hot(y, 3), y


# ---------- iterate_minibatches ----------
def test_minibatches_cover_every_index_once():
    batches = list(iterate_minibatches(10, 3, np.random.default_rng(0)))
    assert [len(b) for b in batches] == [3, 3, 3, 1]  # last batch is partial
    assert np.array_equal(np.sort(np.concatenate(batches)), np.arange(10))


def test_minibatches_shuffle_and_reproducible():
    a = np.concatenate(list(iterate_minibatches(50, 7, np.random.default_rng(1))))
    b = np.concatenate(list(iterate_minibatches(50, 7, np.random.default_rng(1))))
    assert np.array_equal(a, b) and not np.array_equal(a, np.arange(50))
    no_shuffle = np.concatenate(list(iterate_minibatches(50, 7, shuffle=False)))
    assert np.array_equal(no_shuffle, np.arange(50))


def test_minibatches_edge_cases():
    batches = list(iterate_minibatches(5, 100, np.random.default_rng(0)))
    assert len(batches) == 1 and len(batches[0]) == 5  # batch_size > n: one full batch
    with pytest.raises(ValueError):
        list(iterate_minibatches(5, 0, np.random.default_rng(0)))
    with pytest.raises(ValueError):
        list(iterate_minibatches(5, 2, rng=None, shuffle=True))


# ---------- sgd_step ----------
def test_sgd_step_exact_update_in_place():
    W = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([0.5, -0.5])
    params = [W, b]
    sgd_step(params, [np.ones((2, 2)), np.array([1.0, -1.0])], lr=0.1)
    assert params[0] is W  # updated in place, same object
    assert np.allclose(W, [[0.9, 1.9], [2.9, 3.9]])
    assert np.allclose(b, [0.4, -0.4])


def test_sgd_step_multiple_layers_zero_grad():
    params = [np.ones((3, 4)), np.ones((4, 2))]
    sgd_step(params, [np.zeros((3, 4)), np.full((4, 2), 2.0)], lr=0.5)
    assert np.all(params[0] == 1.0) and np.all(params[1] == 0.0)


def test_sgd_step_rejects_bad_input():
    with pytest.raises(ValueError):
        sgd_step([np.ones((2, 2))], [np.ones((2, 3))], lr=0.1)  # shape mismatch
    with pytest.raises(ValueError):
        sgd_step([np.ones(2)], [np.ones(2), np.ones(2)], lr=0.1)  # count mismatch
    with pytest.raises(ValueError):
        sgd_step([np.ones(2)], [np.ones(2)], lr=0.0)


# ---------- accuracy ----------
def test_accuracy():
    assert accuracy([0, 1, 2, 2], [0, 1, 1, 2]) == 0.75
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])


# ---------- train ----------
def test_train_loss_decreases_and_fits_toy_data():
    X, Y, y = toy_data()
    model = SoftmaxRegression(2, 3)
    hist = train(model, X, Y, epochs=30, batch_size=16, lr=0.1, seed=0)
    assert len(hist["train_loss"]) == 30
    assert hist["train_loss"][-1] < 0.5 * hist["train_loss"][0]
    assert accuracy(y, model.predict(X)) > 0.95


def test_train_full_batch_equals_manual_gradient_descent():
    X, Y, _ = toy_data()
    m1, m2 = SoftmaxRegression(2, 3, seed=1), SoftmaxRegression(2, 3, seed=1)
    train(m1, X, Y, epochs=5, batch_size=len(X), lr=0.2, shuffle=False)
    for _ in range(5):
        m2.forward(X)
        dW, db = m2.backward(Y)
        m2.weights[0] -= 0.2 * dW[0]
        m2.biases[0] -= 0.2 * db[0]
    assert np.allclose(m1.weights[0], m2.weights[0]) and np.allclose(m1.biases[0], m2.biases[0])


def test_train_reproducible_with_same_seed():
    X, Y, _ = toy_data()
    h1 = train(SoftmaxRegression(2, 3), X, Y, epochs=5, batch_size=8, lr=0.1, seed=7)
    h2 = train(SoftmaxRegression(2, 3), X, Y, epochs=5, batch_size=8, lr=0.1, seed=7)
    h3 = train(SoftmaxRegression(2, 3), X, Y, epochs=5, batch_size=8, lr=0.1, seed=8)
    assert h1["train_loss"] == h2["train_loss"]
    assert h1["train_loss"] != h3["train_loss"]


def test_train_validation_history_and_bad_args():
    X, Y, _ = toy_data()
    hist = train(SoftmaxRegression(2, 3), X, Y, epochs=3, X_val=X[:30], Y_val=Y[:30])
    assert set(hist) == {"train_loss", "train_acc", "val_loss", "val_acc"}
    assert all(len(v) == 3 for v in hist.values())
    with pytest.raises(ValueError):
        train(SoftmaxRegression(2, 3), X, Y, X_val=X)  # Y_val missing
    with pytest.raises(ValueError):
        train(SoftmaxRegression(2, 3), X, Y.argmax(axis=1))  # integer labels, not one-hot

    class Diverged(SoftmaxRegression):
        def loss(self, P, Y):
            return float("nan")

    with pytest.raises(FloatingPointError):  # stop early instead of training on NaNs
        train(Diverged(2, 3), X, Y, epochs=5)


def test_train_on_dry_bean_smoke():
    d = prepare_dry_bean()
    model = SoftmaxRegression(16, 7)
    hist = train(model, d["X_train"], d["Y_train"], epochs=10, batch_size=64, lr=0.1,
                 X_val=d["X_val"], Y_val=d["Y_val"])
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    assert hist["val_acc"][-1] > 0.85  # linear model already separates most classes
