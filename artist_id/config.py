"""Central configuration for the classical-ML half of the artist-identification project.

Everything that the scripts share (paths, the list of artists, image size, the
random seed, PCA size, hyper-parameter grids) lives here so the experiments are
reproducible and easy to tweak from one place.
"""
import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "images"            # data/images/<Artist Name>/<file>.jpg, resized to 256x256
SPLITS_CSV = DATA_DIR / "splits.csv"     # filename, artist, label, split (built from the team CSVs)
FEATURE_DIR = DATA_DIR / "features"      # cached feature matrices (.npy)
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
MODEL_DIR = RESULTS_DIR / "models"

for _d in (DATA_DIR, RAW_DIR, FEATURE_DIR, RESULTS_DIR, FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- dataset
# Shared team dataset (github.com/oomploompaluvr/artist-id, data/): 15 Painters-by-Numbers
# artists with a fixed train/val/test split and label map, so the SVM and the CNN are
# trained and tested on exactly the same paintings with the same label ids.
TEAM_SPLIT_CSVS = {s: DATA_DIR / f"{s}.csv" for s in ("train", "val", "test")}
LABEL_MAP_JSON = DATA_DIR / "label_map.json"
# Google-Drive zips with the original images (selected_images/<Artist Name>/<file>.jpg).
# Put them in data/zips/ (or point ARTIST_ID_ZIP_DIR at the folder that holds them).
ZIP_DIR = Path(os.environ.get("ARTIST_ID_ZIP_DIR", DATA_DIR / "zips"))
IMAGE_ZIPS = sorted(ZIP_DIR.glob("selected_images-*.zip"))


def _load_artists():
    import json

    if LABEL_MAP_JSON.exists():
        m = json.loads(LABEL_MAP_JSON.read_text(encoding="utf-8"))
        return [a for a, _ in sorted(m.items(), key=lambda kv: kv[1])]
    return []


ARTISTS = _load_artists()                 # index == label id from label_map.json
ARTIST_TO_LABEL = {a: i for i, a in enumerate(ARTISTS)}
SHORT_NAMES = [a.split()[-1] for a in ARTISTS]  # "Durer", "Pissarro", ...

IMG_SIZE = 256  # every painting is resized to 256 x 256, as in the paper
SEED = 42

# --------------------------------------------------------------------------- features
# The six descriptors from the paper. All 63 non-empty combinations are searched.
FEATURES = ["gist", "hu", "color_hist", "sift", "hog", "haralick"]
# Extra descriptor shown on the poster (extracted + visualised, not in the combination search).
EXTRA_FEATURES = ["lbp"]

FEATURE_LABELS = {
    "gist": "GIST",
    "hu": "Hu Moments",
    "color_hist": "Color Hist",
    "sift": "SIFT (BoVW)",
    "hog": "HOG",
    "haralick": "Haralick",
    "lbp": "LBP",
}

SIFT_VOCAB_SIZE = 100        # k for the SIFT bag-of-visual-words codebook
SIFT_MAX_KEYPOINTS = 500     # keypoints kept per image
COLOR_BINS = (8, 8, 8)       # HSV bins per channel (8 per channel as in the paper)

# --------------------------------------------------------------------------- model
PCA_COMPONENTS = 100
CV_FOLDS = 3

# Full grid from the paper: gamma, C in (1e-5, 1e5). Used on the best feature sets.
FULL_C_GRID = np.logspace(-5, 5, 11)
FULL_GAMMA_GRID = np.logspace(-5, 5, 11)

# Coarser grid used while screening all 63 feature combinations (keeps run-time sane).
COARSE_C_GRID = np.logspace(0, 2, 3)        # 1, 10, 100
COARSE_GAMMA_GRID = np.logspace(-2, 1, 4)    # 0.01, 0.1, 1, 10

# How many of the best combinations (by validation accuracy) get the full 11x11 grid.
TOP_K_COMBOS_FOR_FULL_GRID = 3

N_JOBS = -1
