"""Run after copying src/neural_network.py into the team repository root."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from neural_network import NeuralNetwork
from data import prepare_dry_bean

if __name__ == '__main__':
    data = prepare_dry_bean()
    model = NeuralNetwork([16, 64, 32, 7], random_state=42)
    X = data['X_val'][:5]
    P = model.predict_proba(X)
    print('Input shape:', X.shape)
    print('Output shape:', P.shape)
    print('Parameter count:', sum(p.size for p in model.weights + model.biases))
    print('Row sums:', P.sum(axis=1))
    print('Predicted class indices:', P.argmax(axis=1))
    assert P.shape == (5, 7)
    assert np.all(np.isfinite(P))
    assert np.allclose(P.sum(axis=1), 1)
    print('Forward smoke test PASSED (untrained model; no accuracy claim)')
