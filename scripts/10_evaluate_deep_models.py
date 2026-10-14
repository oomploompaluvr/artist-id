"""
Test-set evaluation of the deep-learning models + comparison with the SVM.

Loads the checkpoints written by 08_train_baseline_cnn.py and 09_train_mobilenetv2.py,
evaluates them once on the held-out test split (584 paintings) and writes:

    results/cnn_classification_report.csv          per-artist precision / recall / F1
    results/mobilenet_classification_report.csv
    results/cnn_confusion_matrix.png (+ .npy)      rows = true artist, columns = predicted
    results/mobilenet_confusion_matrix.png (+ .npy)
    results/mobilenet_metrics.json
    results/cnn_model_comparison.csv               baseline CNN vs MobileNetV2
    results/final_model_comparison.csv             + SVM (from results/final_models.json)
    results/latency_comparison.csv                 inference time per painting, CNN / MobileNetV2 / SVM

Images are preprocessed exactly as in training: read the original painting, resize to
128 x 128 and scale to [0, 1] (MobileNetV2 rescales to [-1, 1] inside the model).

Run in Colab after training (same PAINTER_PROJECT_DIR as the training scripts):
    python scripts/10_evaluate_deep_models.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(os.environ.get("PAINTER_PROJECT_DIR", "/content/drive/MyDrive/PainterProject"))
IMAGE_DIR = PROJECT_DIR / "selected_images"
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
LATENCY_SAMPLES = 50

MODELS = {
    "cnn": {"name": "Baseline CNN", "file": "best_cnn.keras"},
    "mobilenet": {"name": "MobileNetV2", "file": "best_mobilenet.keras"},
}

try:  # same image pipeline as training when TensorFlow is available (Colab)
    import tensorflow as tf

    def load_image(path):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        return (tf.cast(image, tf.float32) / 255.0).numpy()
except ImportError:  # fallback (e.g. Keras on the PyTorch backend): bilinear resize with OpenCV
    import cv2

    def load_image(path):
        image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        return image.astype(np.float32) / 255.0

import keras  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402


def predict(model, paths):
    """Class probabilities for every painting, in batches."""
    out = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = np.stack([load_image(p) for p in paths[i:i + BATCH_SIZE]])
        out.append(model.predict(batch, verbose=0))
    return np.concatenate(out)


def latency_ms(model, paths):
    """Mean time to classify ONE painting (read + resize + predict), batch size 1."""
    model.predict(np.stack([load_image(paths[0])]), verbose=0)  # warm-up
    t0 = time.perf_counter()
    for p in paths[:LATENCY_SAMPLES]:
        model.predict(np.stack([load_image(p)]), verbose=0)
    return (time.perf_counter() - t0) / LATENCY_SAMPLES * 1000


def plot_confusion(cm, artists, title, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(artists)), artists, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(artists)), artists, fontsize=8)
    for i in range(len(artists)):
        for j in range(len(artists)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted artist")
    ax.set_ylabel("Actual artist")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    with open(DATA_DIR / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    artists = [a for a, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    paths = [str(IMAGE_DIR / p) for p in test_df["image_path"]]
    y_true = test_df["label"].to_numpy()
    print("Test paintings:", len(paths))

    rows, latency = [], []
    for key, info in MODELS.items():
        model_path = MODEL_DIR / info["file"]
        if not model_path.exists():
            print(f"Skipping {info['name']}: {model_path} not found")
            continue
        model = keras.models.load_model(model_path, compile=False)  # inference only; avoids re-compiling the training setup
        y_pred = predict(model, paths).argmax(axis=1)

        acc = float((y_pred == y_true).mean())
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        print(f"{info['name']}: test accuracy {acc:.4f}, macro F1 {macro_f1:.4f}")

        report = classification_report(y_true, y_pred, labels=range(len(artists)), target_names=artists,
                                       output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(RESULTS_DIR / f"{key}_classification_report.csv")
        cm = confusion_matrix(y_true, y_pred, labels=range(len(artists)))
        np.save(RESULTS_DIR / f"{key}_confusion_matrix.npy", cm)
        plot_confusion(cm, artists, f"{info['name']} Confusion Matrix (test accuracy {acc:.1%})",
                       RESULTS_DIR / f"{key}_confusion_matrix.png")
        if key == "mobilenet":
            with open(RESULTS_DIR / "mobilenet_metrics.json", "w") as f:
                json.dump({"model": "MobileNetV2 Transfer Learning", "test_accuracy": acc, "test_macro_f1": macro_f1,
                           "test_samples": len(paths), "number_of_classes": len(artists)}, f, indent=4)

        rows.append({"Model": info["name"], "Test Accuracy (%)": round(100 * acc, 2), "Test Macro F1": round(macro_f1, 4)})
        latency.append({"Model": info["name"], "Inference (ms / painting)": round(latency_ms(model, paths), 1),
                        "Parameters": model.count_params()})

    if not rows:
        raise SystemExit("No trained models found in " + str(MODEL_DIR))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "cnn_model_comparison.csv", index=False)

    # Add the SVM (classical part) from its own results file, if present.
    svm_json = RESULTS_DIR / "final_models.json"
    if svm_json.exists():
        svm = json.load(open(svm_json))["svm"]
        rows.insert(1, {"Model": "SVM (Person 1)", "Test Accuracy (%)": round(100 * svm["test_acc"], 2),
                        "Test Macro F1": round(svm["macro_f1"], 4)})
        latency.append({"Model": "SVM (Person 1)", "Inference (ms / painting)": round(svm["total_inference_ms"], 1),
                        "Parameters": None})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "final_model_comparison.csv", index=False)
    pd.DataFrame(latency).to_csv(RESULTS_DIR / "latency_comparison.csv", index=False)

    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(latency).to_string(index=False))


if __name__ == "__main__":
    main()
