"""Team dataset preparation (shared train/val/test split) and image loading."""
from __future__ import annotations

import zipfile

import cv2
import numpy as np
import pandas as pd

from . import config as C


def load_team_split() -> pd.DataFrame:
    """Concatenate the team's train/val/test CSVs into one table: filename, artist, label, split."""
    parts = []
    for split, path in C.TEAM_SPLIT_CSVS.items():
        d = pd.read_csv(path)
        parts.append(pd.DataFrame({"filename": d.image_path, "artist": d.artist, "label": d.label, "split": split}))
    df = pd.concat(parts, ignore_index=True)
    # The label ids must agree with label_map.json (the CNN uses the same map).
    assert (df.artist.map(C.ARTIST_TO_LABEL) == df.label).all(), "labels disagree with label_map.json"
    assert df.filename.is_unique
    return df.sort_values(["label", "filename"]).reset_index(drop=True)


def _decode(buf: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:  # unusual formats (CMYK, GIF, ...) -> fall back to Pillow
        import io

        from PIL import Image

        try:
            img = cv2.cvtColor(np.asarray(Image.open(io.BytesIO(buf)).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


def _resize_members(zip_path, members, wanted):
    """Worker: decode the given zip members, resize to 256x256 and write them to data/images/."""
    done, failed = [], []
    with zipfile.ZipFile(zip_path) as z:
        for m in members:
            rel = m.split("/", 1)[1]  # strip leading "selected_images/"
            if rel not in wanted:
                continue
            out = C.RAW_DIR / rel
            if not out.exists():
                img = _decode(z.read(m))
                if img is None:
                    failed.append(rel)
                    continue
                img = cv2.resize(img, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_AREA)
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            done.append(rel)
    return done, failed


def prepare_images(df: pd.DataFrame) -> tuple[set, list]:
    """Extract every image listed in the team split from the Drive zips, resized to 256 x 256."""
    from joblib import Parallel, delayed

    if not C.IMAGE_ZIPS:
        raise FileNotFoundError(f"selected_images-*.zip not found in {C.ZIP_DIR} (set ARTIST_ID_ZIP_DIR)")
    wanted = set(df.filename)
    jobs = []
    for zp in C.IMAGE_ZIPS:
        with zipfile.ZipFile(zp) as z:
            members = [m for m in z.namelist() if not m.endswith("/")]
        for i in range(0, len(members), 200):
            jobs.append((zp, members[i:i + 200]))
    out = Parallel(n_jobs=C.N_JOBS)(delayed(_resize_members)(zp, ms, wanted) for zp, ms in jobs)
    done = set().union(*(o[0] for o in out))
    failed = [f for o in out for f in o[1]]
    return done, failed


def load_splits() -> pd.DataFrame:
    if not C.SPLITS_CSV.exists():
        raise FileNotFoundError(f"{C.SPLITS_CSV} not found - run scripts/01_prepare_data.py first")
    return pd.read_csv(C.SPLITS_CSV)


def load_image(rel_path: str) -> np.ndarray:
    """Return a 256x256 BGR uint8 image (OpenCV convention)."""
    img = cv2.imread(str(C.RAW_DIR / rel_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read {rel_path}")
    if img.shape[:2] != (C.IMG_SIZE, C.IMG_SIZE):
        img = cv2.resize(img, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img
