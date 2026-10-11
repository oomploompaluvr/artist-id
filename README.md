# artist-id

Images: download the `selected_images-*.zip` files from the team Drive
(https://drive.google.com/drive/folders/14f4SQF5noo2BsbjKi2T7D3VcDROGYDaH, also in `data/dataset actual img.txt`)
and put them in `data/zips/`. The train/val/test split and label map are in `data/`.

---

## Classical ML & Feature Engineering

Reproduction of the **feature-extraction + SVM** half of *"Comparison of Machine Learning
Techniques for Artist Identification"* (Chen & Deng, Stanford CS229).

| Task (from the work split) | Where |
|---|---|
| GIST, Hu moments, color histograms (+ SIFT, HOG, Haralick, LBP) with OpenCV / Mahotas | `artist_id/features.py`, `artist_id/gist.py` |
| Min-Max scaling + PCA to 100 components | `artist_id/models.py` (pipeline) |
| Logistic Regression baseline | `artist_id/models.py`, `scripts/04_…`, `scripts/05_…` |
| Multi-class (one-vs-rest) RBF SVM, scikit-learn | `artist_id/models.py` |
| 3-fold CV grid search over C and γ | `scripts/04_feature_combination_search.py`, `scripts/05_train_final_models.py` |
| Write-up Section 3 + Section 4.1 | `scripts/07_build_writeup.py` (generates Section 3 + 4.1 as .docx/.md) |
| Feature plots + classical-model evaluation figures | `figures/` |

### Results

| Model | Train acc | Val acc | Test acc | Train time (s) | Inference (ms/img) | Features | C | γ |
|---|---|---|---|---|---|---|---|---|
| LR (baseline) | 72.3% | 62.2% | **61.1%** | 0.54 | 108.9 | GIST + Hu Moments + Color Hist + SIFT (BoVW) + Haralick | 1 |  |
| SVM | 91.4% | 65.7% | **65.4%** | 3.94 | 110.0 | GIST + Color Hist + SIFT (BoVW) + Haralick | 100 | 0.01 |
| SVM | 92.3% | 65.0% | **64.7%** | 3.95 | 110.1 | GIST + Hu Moments + Color Hist + SIFT (BoVW) + Haralick | 100 | 0.01 |
| SVM | 90.5% | 62.6% | **65.1%** | 4.0 | 97.3 | GIST + Color Hist + SIFT (BoVW) | 100 | 0.01 |

The final SVM (GIST + Color Hist + SIFT (BoVW) + Haralick, C = 100, γ = 0.01) gets **65.4%**
test accuracy on the team split, against 61.1% for the LR baseline (paper, different artists: 68.1% / 61.2%).
The SVM overfits, with 91.4% training accuracy. Average effect of adding each descriptor:
GIST +8.6 pp, Color Hist +5.7 pp, Haralick +3.6 pp, SIFT (BoVW) +3.6 pp, Hu Moments +0.5 pp, HOG -5.6 pp. The most confused pairs are Saryan/Picasso and
Chagall/Picasso. Inference is ≈110 ms/image on one core, almost all of it feature extraction.

### Dataset

This uses the team's shared dataset from [oomploompaluvr/artist-id](https://github.com/oomploompaluvr/artist-id).
It holds 5,779 *Painters by Numbers* paintings by 15 artists (367–412 each) with a fixed split of
4,615 train / 580 val / 584 test. The CNN uses the same split, so the two models are tested on the same paintings.

* `data/train.csv`, `val.csv`, `test.csv`, `label_map.json` are copied unchanged from the repo. The label ids
  (Durer = 0 … Beksinski = 14) are the team's.
* The original-resolution images come from the team Drive (`selected_images-*.zip`, ~3 GB). Put the zips in
  `data/zips/`, or set `ARTIST_ID_ZIP_DIR` to the folder that holds them. Step 1 resizes every painting to 256 × 256,
  as in the paper, and writes it to `data/images/<Artist>/<file>.jpg` (205 MB, not committed).

### Pipeline

```
painting (256x256) ─► 6 descriptors ─► concatenate (63 combos) ─► MinMaxScaler ─► PCA(100) ─► LR  /  RBF-SVM (one-vs-rest)
```

1. **`01_prepare_data.py`** reads the team CSVs, extracts and resizes the listed images from the Drive zips,
   and writes `data/splits.csv`.
2. **`02_extract_features.py`** builds the SIFT vocabulary from training images only and extracts all
   descriptors in parallel to `data/features/*.npy`. It times each descriptor on one core
   (`results/feature_extraction_times.csv`).
3. **`03_visualize_features.py`** writes the feature figures to `figures/features/`.
4. **`04_feature_combination_search.py`** screens all 63 combinations with a 3-fold CV grid
   (SVM: C ∈ {1, 10, 100}, γ ∈ {0.01, 0.1, 1, 10}; LR: C ∈ {0.01 … 1000}) and writes `results/combination_search.csv`.
   Screening uses libsvm's one-vs-one multiclass because it is about 7× faster. Final models are one-vs-rest.
5. **`05_train_final_models.py`** runs the paper's full grid, C, γ ∈ {10⁻⁵ … 10⁵} (11 × 11, 3-fold CV),
   on the top-3 combinations with one-vs-rest SVMs. It picks the final model by validation accuracy and evaluates it
   once on the test split (`results/final_results.csv`, `final_models.json`, confusion matrices,
   classification reports, `results/models/*.joblib`).
6. **`06_plot_results.py`** writes the evaluation figures to `figures/models/`.
7. **`07_build_writeup.py`** writes Section 3 + 4.1 with every number taken from `results/`.

### How to run

```bash
pip install -r requirements.txt
python run_all.py          # or: python run_all.py 4 5 6 7
```

Each step caches its output. Run-time on an 8-core laptop CPU (i5-12450H): data prep ≈ 2 min,
features ≈ 7 min, screening ≈ 25 min, full grid search ≈ 50 min (3 × ~17 min).

### Implementation notes

* **GIST** is our own NumPy port of Torralba's `LMgist.m` (pre-filtering, 4 scales × 8 orientations of
  Gabor filters, 4 × 4 grid, 512-d), because `lear-gist` needs FFTW and does not build on Windows.
* **Hu moments** are log-scaled, `-sign(h)·log10|h|`, because raw values span about 20 orders of magnitude.
* **Color histogram**: HSV with 8 bins per channel gives a 512-bin joint histogram.
* **SIFT** is encoded as a bag of visual words (100-word k-means vocabulary).
  **HOG**: 128 × 128, 9 orientations, 16 px cells, 2 × 2 blocks (1,764-d). **Haralick**: 13 GLCM statistics (Mahotas).
* The scaler and PCA live inside the scikit-learn `Pipeline`, so they are re-fit in every CV fold and
  never see validation or test data.
