import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import prepare_dry_bean  # noqa: E402
from mock_models import SoftmaxRegression  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn_baseline import (  # noqa: E402
    compare_with_sklearn, fit_sklearn_from_params, glorot_uniform_init, make_sklearn_mlp,
)


@pytest.fixture(scope="module")
def bean():
    return prepare_dry_bean()


# ---------- make_sklearn_mlp ----------
def test_make_sklearn_mlp_matches_baseline_settings():
    p = make_sklearn_mlp((64, 32), lr=0.05, batch_size=16, epochs=7, seed=3).get_params()
    assert p["hidden_layer_sizes"] == (64, 32) and p["activation"] == "relu"
    assert p["solver"] == "sgd" and p["learning_rate"] == "constant"
    assert p["momentum"] == 0 and p["nesterovs_momentum"] is False and p["alpha"] == 0
    assert p["learning_rate_init"] == 0.05 and p["batch_size"] == 16
    assert p["max_iter"] == 7 and p["random_state"] == 3 and p["early_stopping"] is False


def test_make_sklearn_mlp_runs_all_epochs(bean):
    mlp = make_sklearn_mlp((8,), epochs=4, batch_size=256)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit(bean["X_train"], bean["y_train"])
    assert mlp.n_iter_ == 4 and len(mlp.loss_curve_) == 4  # no early stopping


# ---------- glorot_uniform_init ----------
def test_glorot_init_shapes_bounds_reproducible():
    W, b = glorot_uniform_init([16, 64, 32, 7], seed=1)
    assert [w.shape for w in W] == [(16, 64), (64, 32), (32, 7)]
    assert [x.shape for x in b] == [(64,), (32,), (7,)]
    assert np.all(np.abs(W[0]) <= np.sqrt(6 / 80)) and np.all(np.abs(b[2]) <= np.sqrt(6 / 39))
    W2, _ = glorot_uniform_init([16, 64, 32, 7], seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(W, W2))
    with pytest.raises(ValueError):
        glorot_uniform_init([16])


def test_glorot_init_equals_sklearns_own_init(bean):
    # A learning rate of 1e-300 leaves the parameters effectively unchanged,
    # so after one epoch sklearn's coefs_ are still its initial values.
    mlp = make_sklearn_mlp((64, 32), lr=1e-300, epochs=1, seed=42, shuffle=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit(bean["X_train"], bean["y_train"])
    W, b = glorot_uniform_init([16, 64, 32, 7], seed=42)
    assert all(np.allclose(x, y, rtol=0, atol=1e-12) for x, y in zip(W, mlp.coefs_))
    assert all(np.allclose(x, y, rtol=0, atol=1e-12) for x, y in zip(b, mlp.intercepts_))


# ---------- fit_sklearn_from_params ----------
def test_fit_from_params_starts_from_given_params(bean):
    W, b = glorot_uniform_init([16, 8, 7], seed=5)
    mlp = make_sklearn_mlp((8,), lr=1e-300, epochs=3, seed=0)
    fit_sklearn_from_params(mlp, bean["X_train"], bean["y_train"], W, b)
    assert np.allclose(mlp.coefs_[0], W[0], rtol=0, atol=1e-12)
    assert len(mlp.loss_curve_) == 3  # set-up pass is not counted
    assert W[0] is not mlp.coefs_[0]  # our arrays are copied, not shared


def test_fit_from_params_rejects_wrong_shapes(bean):
    W, b = glorot_uniform_init([16, 10, 7])
    with pytest.raises(ValueError):
        fit_sklearn_from_params(make_sklearn_mlp((8,)), bean["X_train"], bean["y_train"], W, b)


# ---------- compare_with_sklearn ----------
def test_exact_mode_matches_sklearn_to_machine_precision(bean):
    # Softmax regression == MLPClassifier with no hidden layers.
    model = SoftmaxRegression(16, 7, seed=3)
    rep = compare_with_sklearn(model, bean["X_train"], bean["Y_train"], bean["X_test"], bean["y_test"],
                               hidden_layer_sizes=(), lr=0.05, batch_size=64, epochs=5, shuffle=False)
    assert rep["mode"] == "exact"
    assert rep["max_param_diff"] < 1e-10
    assert rep["max_loss_curve_diff"] < 1e-10
    assert rep["prediction_agreement"] == 1.0


def test_statistical_mode_gives_similar_results(bean):
    model = SoftmaxRegression(16, 7, seed=3)
    rep = compare_with_sklearn(model, bean["X_train"], bean["Y_train"], bean["X_test"], bean["y_test"],
                               hidden_layer_sizes=(), lr=0.05, batch_size=64, epochs=10, shuffle=True)
    assert rep["mode"] == "statistical"
    assert abs(rep["ours_test_acc"] - rep["sklearn_test_acc"]) < 0.02
    assert abs(rep["ours_loss_curve"][-1] - rep["sklearn_loss_curve"][-1]) < 0.02
    assert len(rep["ours_loss_curve"]) == len(rep["sklearn_loss_curve"]) == 10
