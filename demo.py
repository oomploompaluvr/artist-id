"""
Live demo: identify the artist of a painting with every trained model side by side.

    python demo.py                      # opens the web app (Gradio) in the browser
    python demo.py --image path.jpg     # command line, no browser

* RBF SVM (classical): extracts GIST / colour histogram / SIFT / Haralick features and predicts
  with results/models/svm.joblib. The "What the SVM sees" tab shows the extracted features.
* Baseline CNN and MobileNetV2 (deep learning): loaded from DEEP_MODEL_DIR
  (default results/models/: best_cnn.keras, best_mobilenet.keras) if the files are present.

Each prediction shows the top-3 artists and the time it took. CNN / MobileNetV2 show softmax
probabilities; the SVM shows its decision scores (an SVM does not output probabilities).
"""
import argparse
import importlib.util
import os
import time
from pathlib import Path

if "KERAS_BACKEND" not in os.environ and importlib.util.find_spec("tensorflow") is None:
    os.environ["KERAS_BACKEND"] = "torch"   # Keras 3 can also run the .keras models on PyTorch

import cv2  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from artist_id import config as C  # noqa: E402
from artist_id import features as F  # noqa: E402

DEEP_MODEL_DIR = Path(os.environ.get("DEEP_MODEL_DIR", C.MODEL_DIR))
DEEP_MODELS = {"Baseline CNN": "best_cnn.keras", "MobileNetV2": "best_mobilenet.keras"}
DEEP_SIZE = (128, 128)


# ------------------------------------------------------------------ models
class SVMModel:
    """The tuned RBF SVM pipeline (MinMax -> PCA(100) -> 15 one-vs-rest SVMs)."""

    name = "RBF SVM"
    probabilistic = False  # SVMs give decision scores, not probabilities

    def __init__(self):
        import json

        self.pipe = joblib.load(C.MODEL_DIR / "svm.joblib")
        self.vocab = np.load(C.FEATURE_DIR / "sift_vocab.npy")
        self.combo = json.load(open(C.RESULTS_DIR / "final_models.json"))["svm"]["combo"].split("+")

    def predict(self, rgb):
        bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_AREA)
        feats = []
        for name in self.combo:
            feats.append(F.sift_bovw_features(bgr, self.vocab) if name == "sift" else F.EXTRACTORS[name](bgr))
        return self.pipe.decision_function(np.concatenate(feats)[None])[0]  # one score per artist


class KerasModel:
    probabilistic = True  # softmax output

    def __init__(self, name, path):
        import keras

        self.name = name
        self.model = keras.models.load_model(path, compile=False)  # inference only
        self.model.predict(np.zeros((1, *DEEP_SIZE, 3), np.float32), verbose=0)  # warm-up: first call is slow

    def predict(self, rgb):
        x = cv2.resize(rgb, DEEP_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        return np.asarray(self.model.predict(x[None], verbose=0))[0]


def load_models():
    models = [SVMModel()]
    for name, file in DEEP_MODELS.items():
        path = DEEP_MODEL_DIR / file
        if path.exists():
            models.append(KerasModel(name, path))
        else:
            print(f"[demo] {name} not loaded: {path} not found")
    return models


def run_all(models, rgb):
    """[(model name, [(artist, value) x3], ms, is_probability)] for every model."""
    out = []
    for m in models:
        t0 = time.perf_counter()
        p = m.predict(rgb)
        ms = (time.perf_counter() - t0) * 1000
        top = np.argsort(p)[::-1][:3]
        out.append((m.name, [(C.ARTISTS[i], float(p[i])) for i in top], ms, m.probabilistic))
    return out


def fmt(value, is_prob):
    """Probability as a percentage; SVM decision score as a plain number."""
    return f"{value:.0%}" if is_prob else f"score {value:+.2f}"


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ feature views (Person 1 part)
def feature_views(rgb):
    from skimage.feature import hog

    bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_AREA)
    kp, _ = F._get_sift().detectAndCompute(F.to_gray(bgr), None)
    sift = cv2.cvtColor(cv2.drawKeypoints(bgr, kp, None, color=(214, 120, 42),
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), cv2.COLOR_BGR2RGB)
    gray = cv2.resize(F.to_gray(bgr), (F.HOG_SIZE, F.HOG_SIZE), interpolation=cv2.INTER_AREA)
    _, h = hog(gray, **F.HOG_PARAMS, visualize=True)
    h = (np.clip(h / (np.percentile(h, 99.5) + 1e-9), 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = pd.DataFrame({ch: cv2.calcHist([hsv], [i], None, [8], [0, r]).ravel() / (C.IMG_SIZE ** 2)
                         for i, (ch, r) in enumerate([("Hue", 180), ("Saturation", 256), ("Value", 256)])})
    hist.insert(0, "bin", range(1, 9))
    return sift, f"{len(kp)} SIFT keypoints", h, hist


# ------------------------------------------------------------------ test-set helpers
def test_set():
    if not C.SPLITS_CSV.exists():
        return None
    df = pd.read_csv(C.SPLITS_CSV)
    df = df[df.split == "test"]
    return df[[(C.RAW_DIR / f).exists() for f in df.filename]]


# ------------------------------------------------------------------ web app
def launch(models, open_browser=True):
    import gradio as gr

    tests = test_set()
    rng = np.random.default_rng()

    def predict(image, true_artist=""):
        if image is None:
            return "Upload a painting or pick a random test painting.", None
        rows, text = [], []
        for name, top, ms, prob in run_all(models, image):
            rows.append([name, top[0][0], fmt(top[0][1], prob), ", ".join(f"{a} ({fmt(v, prob)})" for a, v in top[1:]),
                         f"{ms:.0f} ms"])
            mark = "" if not true_artist else (" ✓" if top[0][0] == true_artist else " ✗")
            text.append(f"**{name}**: {top[0][0]}{mark}")
        header = f"True artist: **{true_artist}**\n\n" if true_artist else ""
        return header + "  \n".join(text), pd.DataFrame(rows, columns=["Model", "Prediction", "Confidence", "Next guesses", "Time"])

    def random_test():
        r = tests.iloc[rng.integers(len(tests))]
        img = read_rgb(C.RAW_DIR / r.filename)
        return (img, r.artist, *predict(img, r.artist))

    def benchmark(n):
        sample = tests.sample(int(n), random_state=int(rng.integers(1 << 30)))
        stats = {m.name: [0, 0.0] for m in models}
        for r in sample.itertuples():
            for name, top, ms, _ in run_all(models, read_rgb(C.RAW_DIR / r.filename)):
                stats[name][0] += top[0][0] == r.artist
                stats[name][1] += ms
        return pd.DataFrame([[k, f"{c}/{len(sample)}", f"{c / len(sample):.0%}", f"{t / len(sample):.0f} ms"]
                             for k, (c, t) in stats.items()], columns=["Model", "Correct", "Accuracy", "Avg. time"])

    with gr.Blocks(title="Artist Identification Demo") as app:
        gr.Markdown("# Artist Identification\nWhich of the 15 artists painted this? "
                    f"Models loaded: {', '.join(m.name for m in models)}")
        true_artist = gr.State("")
        with gr.Tab("Predict"):
            with gr.Row():
                with gr.Column():
                    img = gr.Image(type="numpy", label="Painting")
                    with gr.Row():
                        go = gr.Button("Identify artist", variant="primary")
                        rnd = gr.Button("Random test painting", interactive=tests is not None and len(tests) > 0)
                with gr.Column():
                    summary = gr.Markdown()
                    table = gr.Dataframe(label="Predictions")
            go.click(lambda im: predict(im), img, [summary, table])
            img.upload(lambda: "", None, true_artist)
            rnd.click(random_test, None, [img, true_artist, summary, table])
        with gr.Tab("What the SVM sees"):
            gr.Markdown("Hand-crafted features extracted from the painting in the *Predict* tab. "
                        "(The SVM ranks artists by decision score; higher = more likely.)")
            show = gr.Button("Extract features")
            with gr.Row():
                sift_img = gr.Image(label="SIFT keypoints")
                hog_img = gr.Image(label="HOG gradients")
            n_kp = gr.Markdown()
            hist = gr.BarPlot(x="bin", y="Value", label="Colour histogram (HSV value channel)")
            hist_tbl = gr.Dataframe(label="HSV histograms (fraction of pixels per bin)")

            def views(im):
                if im is None:
                    return None, None, "Load a painting first.", None, None
                s, k, h, hdf = feature_views(im)
                return s, h, k, hdf, hdf
            show.click(views, img, [sift_img, hog_img, n_kp, hist, hist_tbl])
        with gr.Tab("Live benchmark"):
            gr.Markdown("Runs every model on random test paintings and reports accuracy and time per painting.")
            n = gr.Slider(10, 100, value=30, step=10, label="Number of test paintings")
            bench = gr.Button("Run benchmark", interactive=tests is not None and len(tests) > 0)
            out = gr.Dataframe()
            bench.click(benchmark, n, out)
    app.launch(inbrowser=open_browser)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="classify one painting on the command line instead of opening the web app")
    ap.add_argument("--no-browser", action="store_true", help="start the web app without opening a browser tab")
    args = ap.parse_args()
    models = load_models()
    if args.image:
        for name, top, ms, prob in run_all(models, read_rgb(args.image)):
            print(f"{name:<14} {top[0][0]:<28} {fmt(top[0][1], prob):>12}   ({ms:.0f} ms)   next: "
                  + ", ".join(f"{a} {fmt(v, prob)}" for a, v in top[1:]))
    else:
        launch(models, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
