import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import (  # noqa: E402
    StandardScaler, encode_labels, load_dry_bean, one_hot, prepare_dry_bean, stratified_split,
)


# ---------- load_dry_bean ----------
def test_load_shapes_and_classes():
    X, y, classes, feats = load_dry_bean()
    assert X.shape == (13611, 16) and y.shape == (13611,)
    assert classes == ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
    assert len(feats) == 16


def test_load_class_counts_and_no_nan():
    X, y, _, _ = load_dry_bean()
    assert np.bincount(y).tolist() == [1322, 522, 1630, 3546, 1928, 2027, 2636]
    assert np.isfinite(X).all()


def test_load_tiny_arff(tmp_path):
    p = tmp_path / "tiny.arff"
    p.write_text("@RELATION t\n@ATTRIBUTE a REAL\n@ATTRIBUTE b REAL\n"
                 "@ATTRIBUTE Class {B, A}\n@DATA\n1,2,B\n3,4,A\n")
    X, y, classes, feats = load_dry_bean(p)
    assert classes == ["A", "B"] and y.tolist() == [1, 0]
    assert np.array_equal(X, [[1, 2], [3, 4]]) and feats == ["a", "b"]


# ---------- encode_labels / one_hot ----------
def test_encode_labels_and_unknown():
    assert encode_labels(["b", "a", "b"], ["a", "b"]).tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        encode_labels(["z"], ["a", "b"])


def test_one_hot_basic():
    Y = one_hot(np.array([0, 2, 1]), 3)
    assert np.array_equal(Y, np.eye(3)[[0, 2, 1]])
    assert np.allclose(Y.sum(axis=1), 1)


def test_one_hot_edge_cases():
    assert one_hot(np.array([0]), 7).shape == (1, 7)  # missing classes still get columns
    with pytest.raises(ValueError):
        one_hot(np.array([3]), 3)
    with pytest.raises(ValueError):
        one_hot(np.array([-1]), 3)


# ---------- stratified_split ----------
def test_split_partitions_all_rows():
    X = np.arange(200, dtype=float).reshape(100, 2)
    y = np.repeat(np.arange(4), 25)
    Xtr, Xva, Xte, *_ = stratified_split(X, y, 0.2, 0.2, seed=0)
    allrows = np.concatenate([Xtr[:, 0], Xva[:, 0], Xte[:, 0]])
    assert np.array_equal(np.sort(allrows), X[:, 0])  # no overlap, nothing lost


def test_split_is_stratified():
    _, y, _, _ = load_dry_bean()
    X = np.zeros((y.size, 1))
    *_, ytr, yva, yte = stratified_split(X, y, seed=1)
    p = np.bincount(y) / y.size
    for part in (ytr, yva, yte):
        assert np.allclose(np.bincount(part, minlength=7) / part.size, p, atol=0.01)


def test_split_reproducible_and_bad_args():
    X = np.random.default_rng(0).normal(size=(50, 3))
    y = np.repeat([0, 1], 25)
    a = stratified_split(X, y, seed=5)
    b = stratified_split(X, y, seed=5)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    with pytest.raises(ValueError):
        stratified_split(X, y, 0.6, 0.5)


# ---------- StandardScaler ----------
def test_scaler_matches_sklearn():
    from sklearn.preprocessing import StandardScaler as SK
    X = np.random.default_rng(0).normal(3, 5, size=(40, 4))
    assert np.allclose(StandardScaler().fit_transform(X), SK().fit_transform(X))


def test_scaler_constant_feature_and_unfitted():
    X = np.array([[1.0, 2.0], [1.0, 4.0]])
    Z = StandardScaler().fit_transform(X)
    assert np.all(Z[:, 0] == 0) and np.isfinite(Z).all()
    with pytest.raises(RuntimeError):
        StandardScaler().transform(X)


# ---------- prepare_dry_bean ----------
def test_prepare_uses_train_stats_only():
    d = prepare_dry_bean(seed=2060)
    assert np.allclose(d["X_train"].mean(axis=0), 0, atol=1e-10)
    assert np.allclose(d["X_train"].std(axis=0), 1)
    assert not np.allclose(d["X_test"].mean(axis=0), 0, atol=1e-10)  # test not refit
    assert np.array_equal(d["Y_test"].argmax(axis=1), d["y_test"])
