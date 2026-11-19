# DATA 2060 — Deep Neural Network from Scratch

A fully connected deep neural network (MLP) implemented in Python + NumPy,
supporting an arbitrary number of hidden layers, validated against
scikit-learn's `MLPClassifier` on the UCI Dry Bean dataset.

## Layout

```
data/    UCI Dry Bean dataset
src/     NumPy implementation
tests/   unit tests (pytest)
```

## Dataset

[UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
(Koklu & Ozkan, 2020) — 13,611 samples, 16 numerical features, 7 classes.

## Environment

With conda:

```bash
conda env create -f environment.yml
conda activate data2060-dnn
```

Or with pip (Python 3.14):

```bash
pip install -r requirements.txt
```

`numpy` and `scikit-learn` are pinned to exact versions so the sklearn
comparison is reproducible.

## Running tests

```bash
python -m pytest tests -q
```

## Workflow

`main` should always run and pass tests. Work on your own branch and merge
into `main` through a pull request reviewed by a teammate.
