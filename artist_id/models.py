"""Feature loading, preprocessing pipeline and the two classical classifiers.

Pipeline (fit on the training split only, so PCA never sees validation/test data):

    concatenated features -> MinMaxScaler [0, 1] -> PCA(100) -> classifier

Classifiers
* Logistic Regression - multinomial softmax baseline.
* RBF SVM - 15 one-vs-rest binary SVMs (``OneVsRestClassifier(SVC)``), exactly the
  formulation in Section 4.1: min ||w||^2 + C * sum xi  s.t.  y_i (w^T phi(x_i) + b) >= 1 - xi.
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from . import config as C
from .data import load_splits

_cache: dict[str, np.ndarray] = {}


def load_feature(name: str) -> np.ndarray:
    if name not in _cache:
        _cache[name] = np.load(C.FEATURE_DIR / f"{name}.npy")
    return _cache[name]


def load_combo(combo) -> np.ndarray:
    """Concatenate feature matrices horizontally, e.g. ('gist', 'hu')."""
    return np.hstack([load_feature(n) for n in combo])


def get_splits(combo):
    """X/y for train, val and test for a feature combination."""
    df = load_splits()
    X = load_combo(combo)
    y = df.label.to_numpy()
    out = {}
    for s in ("train", "val", "test"):
        m = (df.split == s).to_numpy()
        out[s] = (X[m], y[m])
    return out


def all_combinations(features=C.FEATURES):
    return [c for r in range(1, len(features) + 1) for c in combinations(features, r)]


def combo_name(combo) -> str:
    return " + ".join(C.FEATURE_LABELS[n] for n in combo)


def _pca(n_features: int) -> PCA:
    return PCA(n_components=min(C.PCA_COMPONENTS, n_features), random_state=C.SEED)


def make_svm(n_features: int, C_: float = 1.0, gamma: float | str = "scale", multiclass: str = "ovr") -> Pipeline:
    """RBF SVM pipeline.

    multiclass="ovr" - 15 one-vs-rest binary SVMs (the paper's formulation; used for
                       the full grid search and all final models).
    multiclass="ovo" - libsvm's native one-vs-one scheme; same kernel and C/gamma
                       meaning but several times faster, so it is used only to
                       *screen* the 63 feature combinations.
    """
    svm = SVC(kernel="rbf", C=C_, gamma=gamma, cache_size=1000)
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("pca", _pca(n_features)),
        ("svm", OneVsRestClassifier(svm) if multiclass == "ovr" else svm),
    ])


def make_logreg(n_features: int, C_: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scale", MinMaxScaler()),
        ("pca", _pca(n_features)),
        ("lr", LogisticRegression(C=C_, max_iter=5000)),
    ])


SVM_PARAM = {"C": "svm__estimator__C", "gamma": "svm__estimator__gamma"}     # one-vs-rest pipeline
SVM_OVO_PARAM = {"C": "svm__C", "gamma": "svm__gamma"}                         # one-vs-one pipeline
LR_PARAM = {"C": "lr__C"}


def timed_fit(model, X, y):
    t0 = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - t0


def timed_predict(model, X):
    """Predictions and model-only inference time in ms per image."""
    t0 = time.perf_counter()
    pred = model.predict(X)
    return pred, (time.perf_counter() - t0) / len(X) * 1000


def feature_ms(combo) -> float:
    """Mean feature-extraction time (ms/image) for a combination, from step 2's log."""
    import pandas as pd

    t = pd.read_csv(C.RESULTS_DIR / "feature_extraction_times.csv").set_index("feature")["ms_per_image"]
    return float(sum(t[n] for n in combo))
