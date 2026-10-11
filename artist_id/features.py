"""Global and local image descriptors (OpenCV + Mahotas + scikit-image).

Every ``*_features`` function takes a 256x256 BGR uint8 image and returns a 1-D
float32 vector. ``extract_all`` runs every descriptor on one image and also
records how long each one took (used for the inference-time comparison).

Descriptor         | Library      | Dim  | What it captures
-------------------|--------------|------|------------------------------------------
GIST               | NumPy (ours) | 512  | spatial envelope: scale/orientation energy per region
Hu moments         | OpenCV       | 7    | shape, invariant to translation/scale/rotation
Color histogram    | OpenCV       | 512  | joint HSV colour distribution (8 bins/channel)
SIFT bag-of-words  | OpenCV       | 100  | histogram of local keypoint "visual words"
HOG                | scikit-image | 1764 | local gradient orientation / brush-stroke direction
Haralick           | Mahotas      | 13   | GLCM texture statistics (contrast, entropy, ...)
LBP (extra)        | scikit-image | 26   | micro-texture patterns
"""
from __future__ import annotations

import time

import cv2
import mahotas
import numpy as np
from skimage.feature import hog, local_binary_pattern

from . import config as C
from .gist import gist

HOG_SIZE = 128
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")
LBP_P, LBP_R = 24, 3

_sift = None


def _get_sift():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=C.SIFT_MAX_KEYPOINTS)
    return _sift


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------- global descriptors
def gist_features(img: np.ndarray) -> np.ndarray:
    return gist(to_gray(img))


def hu_features(img: np.ndarray) -> np.ndarray:
    """7 Hu invariants, log-scaled because raw values span ~20 orders of magnitude."""
    hu = cv2.HuMoments(cv2.moments(to_gray(img))).ravel()
    return (-np.sign(hu) * np.log10(np.abs(hu) + 1e-30)).astype(np.float32)


def color_hist_features(img: np.ndarray) -> np.ndarray:
    """Joint HSV histogram, 8 bins per channel -> 512 bins, L1-normalised."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, list(C.COLOR_BINS), [0, 180, 0, 256, 0, 256]).ravel()
    return (h / max(h.sum(), 1)).astype(np.float32)


def haralick_features(img: np.ndarray) -> np.ndarray:
    """13 Haralick GLCM statistics averaged over the 4 directions."""
    return mahotas.features.haralick(to_gray(img), return_mean=True).astype(np.float32)


def hog_features(img: np.ndarray) -> np.ndarray:
    gray = cv2.resize(to_gray(img), (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    return hog(gray, **HOG_PARAMS, feature_vector=True).astype(np.float32)


def lbp_features(img: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(to_gray(img), LBP_P, LBP_R, method="uniform")
    h, _ = np.histogram(lbp, bins=np.arange(LBP_P + 3), density=True)
    return h.astype(np.float32)


# --------------------------------------------------------------------------- SIFT bag of visual words
def sift_descriptors(img: np.ndarray) -> np.ndarray:
    _, desc = _get_sift().detectAndCompute(to_gray(img), None)
    return np.zeros((0, 128), np.float32) if desc is None else desc


def sift_bovw_features(img: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """Normalised histogram of nearest visual words (vocab: k x 128 k-means centres)."""
    desc = sift_descriptors(img)
    k = len(vocab)
    if len(desc) == 0:
        return np.zeros(k, np.float32)
    d2 = (desc ** 2).sum(1)[:, None] - 2 * desc @ vocab.T + (vocab ** 2).sum(1)[None]
    words = d2.argmin(1)
    h = np.bincount(words, minlength=k).astype(np.float32)
    return h / h.sum()


def build_sift_vocab(images, k: int = C.SIFT_VOCAB_SIZE, per_image: int = 60, seed: int = C.SEED) -> np.ndarray:
    """k-means codebook from a random sample of SIFT descriptors of training images."""
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(seed)
    sample = []
    for img in images:
        d = sift_descriptors(img)
        if len(d) > per_image:
            d = d[rng.choice(len(d), per_image, replace=False)]
        sample.append(d)
    sample = np.vstack(sample)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3).fit(sample)
    return km.cluster_centers_.astype(np.float32)


# --------------------------------------------------------------------------- everything at once
EXTRACTORS = {
    "gist": gist_features,
    "hu": hu_features,
    "color_hist": color_hist_features,
    "hog": hog_features,
    "haralick": haralick_features,
    "lbp": lbp_features,
}


def extract_all(img: np.ndarray, vocab: np.ndarray) -> tuple[dict, dict]:
    """Return ({feature: vector}, {feature: seconds}) for one image."""
    feats, times = {}, {}
    for name, fn in EXTRACTORS.items():
        t0 = time.perf_counter()
        feats[name] = fn(img)
        times[name] = time.perf_counter() - t0
    t0 = time.perf_counter()
    feats["sift"] = sift_bovw_features(img, vocab)
    times["sift"] = time.perf_counter() - t0
    return feats, times
