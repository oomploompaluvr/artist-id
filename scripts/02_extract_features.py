"""Step 2 - extract every descriptor for every painting and cache it to data/features/.

* builds the SIFT visual-word vocabulary from TRAINING images only (no leakage),
* runs all extractors in parallel (joblib),
* saves one float32 matrix per descriptor, rows aligned with data/splits.csv,
* logs the mean extraction time per image for each descriptor
  (results/feature_extraction_times.csv) - needed for the inference-time comparison.
"""
import _path  # noqa: F401

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from artist_id import config as C
from artist_id import features as F
from artist_id.data import load_image, load_splits


def _sample_sift(rel_path, per_image, seed):
    rng = np.random.default_rng(seed)
    d = F.sift_descriptors(load_image(rel_path))
    if len(d) > per_image:
        d = d[rng.choice(len(d), per_image, replace=False)]
    return d


def _extract(rel_path, vocab):
    return F.extract_all(load_image(rel_path), vocab)


def build_vocab(train_paths, per_image=60):
    from sklearn.cluster import MiniBatchKMeans

    descs = Parallel(n_jobs=C.N_JOBS, batch_size=32)(
        delayed(_sample_sift)(p, per_image, C.SEED + i) for i, p in enumerate(tqdm(train_paths, desc="SIFT sample"))
    )
    sample = np.vstack(descs)
    print(f"k-means on {len(sample):,} SIFT descriptors -> {C.SIFT_VOCAB_SIZE} visual words")
    km = MiniBatchKMeans(n_clusters=C.SIFT_VOCAB_SIZE, random_state=C.SEED, batch_size=4096, n_init=3).fit(sample)
    return km.cluster_centers_.astype(np.float32)


if __name__ == "__main__":
    df = load_splits()
    t_start = time.time()

    vocab_path = C.FEATURE_DIR / "sift_vocab.npy"
    if vocab_path.exists():
        vocab = np.load(vocab_path)
    else:
        vocab = build_vocab(df.loc[df.split == "train", "filename"].tolist())
        np.save(vocab_path, vocab)

    out = Parallel(n_jobs=C.N_JOBS, batch_size=16)(
        delayed(_extract)(p, vocab) for p in tqdm(df.filename, desc="extracting")
    )

    names = C.FEATURES + C.EXTRA_FEATURES
    for name in names:
        X = np.vstack([feats[name] for feats, _ in out]).astype(np.float32)
        np.save(C.FEATURE_DIR / f"{name}.npy", X)
        print(f"{name:<11} {X.shape}")
    np.save(C.FEATURE_DIR / "labels.npy", df.label.to_numpy())

    # Timing under 12 parallel workers is inflated by contention, so re-time each
    # descriptor serially (one core, no other load) on 100 test images.
    test_paths = df.loc[df.split == "test", "filename"].tolist()[::7][:100]
    times = pd.DataFrame([F.extract_all(load_image(p), vocab)[1] for p in tqdm(test_paths, desc="timing")])
    summary = pd.DataFrame({
        "feature": names,
        "label": [C.FEATURE_LABELS[n] for n in names],
        "dim": [np.load(C.FEATURE_DIR / f"{n}.npy").shape[1] for n in names],
        "ms_per_image": [times[n].mean() * 1000 for n in names],
    })
    summary.to_csv(C.RESULTS_DIR / "feature_extraction_times.csv", index=False)
    print(summary.to_string(index=False))
    print(f"total wall time: {(time.time() - t_start) / 60:.1f} min")
