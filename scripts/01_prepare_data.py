"""Step 1 - build the dataset from the team's shared split.

Inputs (from github.com/oomploompaluvr/artist-id and the team Google Drive):
    data/train.csv, data/val.csv, data/test.csv, data/label_map.json
    D:/Downloads/selected_images-*.zip   (original-resolution paintings)

Every painting listed in the CSVs is extracted, resized to 256 x 256 (as in the paper)
and written to data/images/<Artist Name>/<file>.jpg; data/splits.csv records
filename, artist, label and split for the rest of the pipeline.
"""
import _path  # noqa: F401

from artist_id import config as C
from artist_id.data import load_team_split, prepare_images

if __name__ == "__main__":
    df = load_team_split()
    done, failed = prepare_images(df)
    missing = sorted(set(df.filename) - done)
    if failed or missing:
        print(f"WARNING: {len(failed)} unreadable and {len(missing)} missing images are dropped:", (failed + missing)[:20])
        df = df[df.filename.isin(done)]
    df.to_csv(C.SPLITS_CSV, index=False)

    print(f"\nSaved {len(df)} paintings from {df.artist.nunique()} artists -> {C.SPLITS_CSV}")
    print(df.split.value_counts().to_string())
    print(df.groupby(["label", "artist"]).size().to_string())
