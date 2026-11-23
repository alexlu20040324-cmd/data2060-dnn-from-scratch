"""Member A's tests: run with `python -m pytest tests/test_neural_network.py -q`."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neural_network import NeuralNetwork


# Initialization (3)
def test_parameter_shapes_and_count():
    m = NeuralNetwork([16, 64, 32, 7])
    assert [w.shape for w in m.weights] == [(16, 64), (64, 32), (32, 7)]
    assert [b.shape for b in m.biases] == [(64,), (32,), (7,)]
    assert sum(a.size for a in m.weights + m.biases) == 3399


def test_seed_reproducibility():
    a, b = NeuralNetwork([3, 5, 2], 7), NeuralNetwork([3, 5, 2], 7)
    assert all(np.array_equal(x, y) for x, y in zip(a.weights, b.weights))


def test_invalid_architecture():
    for layers in ([], [4], [4, 0, 2], [4, -1, 2], [4, 2.5, 2]):
        with pytest.raises(ValueError):
            NeuralNetwork(layers)


# ReLU (2)
def test_relu_values():
    assert np.array_equal(NeuralNetwork.relu([-2., 0., 3.]), [0., 0., 3.])


def test_relu_preserves_shape():
    assert NeuralNetwork.relu(np.ones((2, 3))).shape == (2, 3)


# Sigmoid (2)
def test_sigmoid_at_zero():
    assert NeuralNetwork.sigmoid(np.array([0.]))[0] == pytest.approx(0.5)


def test_sigmoid_large_inputs():
    p = NeuralNetwork.sigmoid(np.array([-1000., 1000.]))
    assert np.all(np.isfinite(p)) and p[0] == pytest.approx(0.) and p[1] == pytest.approx(1.)


# Softmax (3)
def test_softmax_row_sums():
    p = NeuralNetwork.softmax(np.array([[1., 2., 3.], [3., 1., -2.]]))
    assert p.shape == (2, 3) and np.allclose(p.sum(axis=1), 1.)


def test_softmax_large_logits():
    p = NeuralNetwork.softmax(np.array([[1000., 1001., 1002.], [-1000., -999., -998.]]))
    assert np.all(np.isfinite(p)) and np.allclose(p.sum(axis=1), 1.)


def test_softmax_invariant_to_row_shift():
    z = np.array([[1., 2., 3.]])
    assert np.allclose(NeuralNetwork.softmax(z), NeuralNetwork.softmax(z + 100000.))


# Forward (4)
def test_forward_shape_and_probability():
    p = NeuralNetwork([16, 64, 32, 7]).forward(np.ones((5, 16)))
    assert p.shape == (5, 7) and np.allclose(p.sum(axis=1), 1.)


def test_forward_caches():
    m = NeuralNetwork([4, 5, 3, 2]); m.forward(np.ones((7, 4)))
    assert [a.shape for a in m.activations] == [(7, 4), (7, 5), (7, 3), (7, 2)]
    assert [z.shape for z in m.pre_activations] == [(7, 5), (7, 3), (7, 2)]


def test_forward_known_example():
    m = NeuralNetwork([2, 2, 2])
    m.weights = [np.eye(2), np.eye(2)]
    m.biases = [np.array([0., 0.]), np.array([0., 0.])]
    p = m.forward([[1., -2.]])
    expected = np.exp([1., 0.]) / np.exp([1., 0.]).sum()
    assert np.allclose(p[0], expected)


def test_forward_multiple_depths_and_bad_shape():
    for layers in ([3, 2], [3, 4, 2], [3, 5, 4, 2]):
        assert NeuralNetwork(layers).forward(np.ones((6, 3))).shape == (6, 2)
    with pytest.raises(ValueError):
        NeuralNetwork([3, 2]).forward(np.ones((6, 4)))


# Predict probabilities (2)
def test_predict_proba_shape():
    assert NeuralNetwork([3, 2]).predict_proba(np.ones((5, 3))).shape == (5, 2)


def test_predict_proba_matches_forward():
    m = NeuralNetwork([3, 5, 2]); X = np.array([[1., 2., 3.]])
    assert np.allclose(m.predict_proba(X), m.forward(X))


# Predict (2)
def test_predict_class_indices():
    m = NeuralNetwork([3, 7]); classes = m.predict(np.ones((5, 3)))
    assert classes.shape == (5,) and np.issubdtype(classes.dtype, np.integer)
    assert np.all((0 <= classes) & (classes < 7))


def test_predict_matches_argmax():
    m = NeuralNetwork([2, 2]); X = np.array([[1., 2.], [4., -3.]])
    assert np.array_equal(m.predict(X), m.predict_proba(X).argmax(axis=1))
