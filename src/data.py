"""Dry Bean dataset pipeline (Member 3).

Loads the UCI Dry Bean dataset, encodes labels, makes a stratified
train/val/test split, and standardizes features using training statistics only.
Pure NumPy so it can sit next to the from-scratch model.

Conventions (shared interface):
    X : float64 array, shape (n_samples, n_features)      -- rows are samples
    y : int64 array,   shape (n_samples,), values 0..K-1  -- integer labels
    Y : float64 array, shape (n_samples, K)               -- one-hot labels
    Class index order = alphabetical class names, which matches
    sklearn's MLPClassifier.classes_ ordering for string labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_ARFF = Path(__file__).resolve().parent.parent / "data" / "DryBeanDataset" / "Dry_Bean_Dataset.arff"


def load_dry_bean(path: str | Path = DEFAULT_ARFF):
    """Read the ARFF file.

    Returns
    -------
    X : (n, 16) float64
    y : (n,) int64, integer labels in alphabetical class order
    class_names : list[str], class_names[k] is the name of label k
    feature_names : list[str]
    """
    feature_names: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            upper = line.upper()
            if not in_data:
                if upper.startswith("@ATTRIBUTE"):
                    name = line.split()[1]
                    if name.lower() != "class":
                        feature_names.append(name)
                elif upper.startswith("@DATA"):
                    in_data = True
                continue
            rows.append(line.split(","))

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    X = np.array([r[:-1] for r in rows], dtype=np.float64)
    labels = np.array([r[-1].strip() for r in rows])
    class_names = sorted(set(labels.tolist()))
    y = encode_labels(labels, class_names)

    if X.shape[1] != len(feature_names):
        raise ValueError("Feature count does not match ARFF header")
    return X, y, class_names, feature_names


def encode_labels(labels, class_names) -> np.ndarray:
    """Map string labels to integer indices given an ordered class list."""
    lookup = {name: i for i, name in enumerate(class_names)}
    try:
        return np.array([lookup[l] for l in labels], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"Unknown label {e.args[0]!r}") from None


def one_hot(y, num_classes: int | None = None) -> np.ndarray:
    """Integer labels (n,) -> one-hot matrix (n, K)."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be 1-D")
    if y.size and (y.min() < 0 or not np.issubdtype(y.dtype, np.integer)):
        raise ValueError("y must contain non-negative integers")
    if num_classes is None:
        num_classes = int(y.max()) + 1 if y.size else 0
    if y.size and y.max() >= num_classes:
        raise ValueError("label out of range for num_classes")
    Y = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def stratified_split(X, y, val_size=0.15, test_size=0.15, seed=2060):
    """Stratified train/val/test split; class proportions preserved in each part.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    if not (0 <= val_size < 1 and 0 <= test_size < 1 and val_size + test_size < 1):
        raise ValueError("val_size and test_size must be in [0,1) and sum to < 1")
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different numbers of rows")

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_test = int(round(test_size * idx.size))
        n_val = int(round(val_size * idx.size))
        test_idx.append(idx[:n_test])
        val_idx.append(idx[n_test:n_test + n_val])
        train_idx.append(idx[n_test + n_val:])

    # Shuffle within each split so classes are interleaved.
    out = []
    for parts in (train_idx, val_idx, test_idx):
        ids = np.concatenate(parts)
        rng.shuffle(ids)
        out.append(ids)
    tr, va, te = out
    return X[tr], X[va], X[te], y[tr], y[va], y[te]


class StandardScaler:
    """z = (x - mean) / std, statistics fitted on training data only."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0  # constant feature: avoid divide-by-zero
        self.scale_ = std
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def prepare_dry_bean(path=DEFAULT_ARFF, val_size=0.15, test_size=0.15, seed=2060):
    """Full pipeline: load -> stratified split -> standardize (train stats).

    Returns a dict with X_train/X_val/X_test, y_* (int), Y_* (one-hot),
    class_names, feature_names, scaler.
    """
    X, y, class_names, feature_names = load_dry_bean(path)
    X_tr, X_va, X_te, y_tr, y_va, y_te = stratified_split(X, y, val_size, test_size, seed)
    scaler = StandardScaler().fit(X_tr)
    K = len(class_names)
    return {
        "X_train": scaler.transform(X_tr), "y_train": y_tr, "Y_train": one_hot(y_tr, K),
        "X_val": scaler.transform(X_va), "y_val": y_va, "Y_val": one_hot(y_va, K),
        "X_test": scaler.transform(X_te), "y_test": y_te, "Y_test": one_hot(y_te, K),
        "class_names": class_names,
        "feature_names": feature_names,
        "scaler": scaler,
    }


if __name__ == "__main__":
    d = prepare_dry_bean()
    print("classes:", d["class_names"])
    for s in ("train", "val", "test"):
        X, y = d[f"X_{s}"], d[f"y_{s}"]
        print(f"{s:5s} X{X.shape} class counts {np.bincount(y, minlength=7).tolist()}")
