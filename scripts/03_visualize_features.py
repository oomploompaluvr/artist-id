"""Step 3 - feature-extraction figures for the write-up and slides (figures/features/)."""
import _path  # noqa: F401

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern

from artist_id import config as C
from artist_id import features as F
from artist_id import plotstyle as S
from artist_id.data import load_image, load_splits
from artist_id.gist import gabor_bank, gabor_responses

plt = S.plt
OUT = C.FIG_DIR / "features"
OUT.mkdir(parents=True, exist_ok=True)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def example(df, artist, k=0):
    rows = df[df.artist == artist].filename.tolist()
    return load_image(rows[k])


def sift_overlay(img):
    kp, _ = F._get_sift().detectAndCompute(F.to_gray(img), None)
    return rgb(cv2.drawKeypoints(img, kp, None, color=(214, 120, 42),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)), len(kp)


def hog_image(img):
    gray = cv2.resize(F.to_gray(img), (F.HOG_SIZE, F.HOG_SIZE), interpolation=cv2.INTER_AREA)
    _, vis = hog(gray, **F.HOG_PARAMS, visualize=True)
    return np.clip(vis / (np.percentile(vis, 99.5) + 1e-9), 0, 1)  # brighten for display


def plot_hsv_hist(ax, img, title="Color histogram (HSV, 8 bins/channel)"):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    names, ranges = ["Hue", "Saturation", "Value"], [180, 256, 256]
    colors = [S.SERIES[0], S.SERIES[1], S.SERIES[2]]
    w = 0.27
    for c, (n, r, col) in enumerate(zip(names, ranges, colors)):
        h = cv2.calcHist([hsv], [c], None, [8], [0, r]).ravel()
        ax.bar(np.arange(8) + (c - 1) * w, h / h.sum(), width=w - 0.03, color=col, label=n)
    ax.set_xticks(range(8), [f"{i + 1}" for i in range(8)])
    ax.set_xlabel("bin"), ax.set_ylabel("fraction of pixels")
    ax.set_title(title), ax.legend(loc="upper right")


# --------------------------------------------------------------------------- figures
def fig_dataset_grid(df):
    fig, axes = plt.subplots(3, 5, figsize=(13, 8.4))
    for ax, artist in zip(axes.ravel(), C.ARTISTS):
        ax.imshow(rgb(example(df, artist, 3)))
        S.no_axes(ax)
        ax.set_title(f"{C.ARTIST_TO_LABEL[artist]}: {artist}", fontsize=9.5)
    fig.suptitle("One painting from each of the 15 artists (256 x 256)", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "dataset_samples.png"), plt.close(fig)


def fig_class_distribution(df):
    counts = df.groupby(["artist", "split"]).size().unstack()[["train", "val", "test"]].loc[C.ARTISTS[::-1]]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    left = np.zeros(len(counts))
    for col, color in zip(["train", "val", "test"], [S.SERIES[0], S.SERIES[1], S.SERIES[2]]):
        ax.barh(counts.index, counts[col], left=left, color=color, label=f"{col} ({counts[col].sum()})",
                edgecolor=S.SURFACE, linewidth=1.5, height=0.7)
        left += counts[col].to_numpy()
    for y, tot in enumerate(left):
        ax.text(tot + 5, y, f"{int(tot)}", va="center", fontsize=8.5, color=S.TEXT_2)
    ax.set_xlabel("number of paintings"), ax.grid(axis="y", visible=False)
    ax.set_title(f"Dataset: {len(df)} paintings, 15 artists, shared team train / val / test split")
    ax.legend(loc="lower right", ncol=3, bbox_to_anchor=(1, -0.16))
    fig.savefig(OUT / "class_distribution.png"), plt.close(fig)


def fig_feature_overview(df):
    """Paper Figure 2 / poster panel: original, colour histogram, LBP, SIFT keypoints, HOG."""
    img = example(df, "Pierre-Auguste Renoir", 7)
    sift_img, n_kp = sift_overlay(img)
    lbp = local_binary_pattern(F.to_gray(img), F.LBP_P, F.LBP_R, method="uniform")
    fig = plt.figure(figsize=(15, 3.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1.45, 1, 1, 1])
    ax = fig.add_subplot(gs[0]); ax.imshow(rgb(img)); S.no_axes(ax); ax.set_title("(a) Original painting")
    ax = fig.add_subplot(gs[1]); plot_hsv_hist(ax, img, "(b) Color histogram")
    ax = fig.add_subplot(gs[2]); ax.imshow(lbp, cmap="gray"); S.no_axes(ax); ax.set_title("(c) Local Binary Pattern")
    ax = fig.add_subplot(gs[3]); ax.imshow(sift_img); S.no_axes(ax); ax.set_title(f"(d) SIFT keypoints ({n_kp})")
    ax = fig.add_subplot(gs[4]); ax.imshow(hog_image(img), cmap="gray"); S.no_axes(ax); ax.set_title("(e) HOG gradients")
    fig.tight_layout()
    fig.savefig(OUT / "feature_overview.png"), plt.close(fig)


def fig_sift_keypoints(df):
    artists = ["Pierre-Auguste Renoir", "Giovanni Battista Piranesi", "Pablo Picasso", "Ivan Aivazovsky"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
    for j, a in enumerate(artists):
        img = example(df, a, 2)
        over, n = sift_overlay(img)
        axes[0, j].imshow(rgb(img)); S.no_axes(axes[0, j]); axes[0, j].set_title(a, fontsize=10)
        axes[1, j].imshow(over); S.no_axes(axes[1, j]); axes[1, j].set_title(f"{n} SIFT keypoints", fontsize=10)
    fig.suptitle("SIFT keypoints (circle size = scale, line = dominant orientation)", fontsize=13, weight="bold")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(OUT / "sift_keypoints.png"), plt.close(fig)


def fig_hog(df):
    artists = ["Gustave Dore", "Paul Cezanne", "Albrecht Durer", "Ivan Shishkin"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
    for j, a in enumerate(artists):
        img = example(df, a, 1)
        axes[0, j].imshow(rgb(img)); S.no_axes(axes[0, j]); axes[0, j].set_title(a, fontsize=10)
        axes[1, j].imshow(hog_image(img), cmap="gray"); S.no_axes(axes[1, j])
        axes[1, j].set_title("HOG (9 orientations, 16 px cells)", fontsize=10)
    fig.suptitle("Histogram of Oriented Gradients captures stroke direction and edge structure",
                 fontsize=13, weight="bold")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(OUT / "hog_examples.png"), plt.close(fig)


def fig_color_hist_comparison(df):
    """Why colour histograms separate Piranesi / Dore (monochrome etchings) from colourists."""
    artists = ["Giovanni Battista Piranesi", "Gustave Dore", "Marc Chagall", "Martiros Saryan"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for j, a in enumerate(artists):
        img = example(df, a, 0)
        axes[0, j].imshow(rgb(img)); S.no_axes(axes[0, j]); axes[0, j].set_title(a, fontsize=10)
        plot_hsv_hist(axes[1, j], img, "HSV histogram")
        axes[1, j].set_ylim(0, 1)
        if j:
            axes[1, j].set_ylabel(""), axes[1, j].get_legend().remove()
    fig.suptitle("Color histograms: monochrome engravers have near-zero saturation",
                 fontsize=13, weight="bold")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(OUT / "color_histograms_by_artist.png"), plt.close(fig)


def fig_mean_saturation(df):
    """Mean colour histogram per artist summarised as average saturation (all images)."""
    X = np.load(C.FEATURE_DIR / "color_hist.npy").reshape(-1, *C.COLOR_BINS)
    sat_bins = (np.arange(C.COLOR_BINS[1]) + 0.5) / C.COLOR_BINS[1]
    mean_sat = (X.sum(axis=(1, 3)) * sat_bins).sum(1)
    s = pd.Series(mean_sat).groupby(df.artist.to_numpy()).mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.6))
    colors = [S.ORANGE if a in ("Giovanni Battista Piranesi", "Gustave Dore") else S.BLUE for a in s.index]
    ax.barh(s.index, s.values, color=colors, height=0.65)
    for y, v in enumerate(s.values):
        ax.text(v + 0.005, y, f"{v:.2f}", va="center", fontsize=8.5, color=S.TEXT_2)
    ax.set_xlabel("mean saturation from the HSV colour histogram (0-1)"), ax.grid(axis="y", visible=False)
    ax.set_title("Average saturation per artist (orange = monochrome engravers)")
    fig.savefig(OUT / "mean_saturation_by_artist.png"), plt.close(fig)


def fig_gist(df):
    img = example(df, "Ivan Aivazovsky", 0)
    gray = F.to_gray(img)
    bank = gabor_bank(n=256)
    resp = gabor_responses(gray)

    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(4, 12, hspace=0.35, wspace=0.15)
    ax = fig.add_subplot(gs[:, :4]); ax.imshow(rgb(img)); S.no_axes(ax); ax.set_title("Input (Aivazovsky)")
    # 4 scales x 8 orientations of filter transfer functions (frequency domain)
    for s in range(4):
        for o in range(0, 8, 2):
            k = s * 8 + o
            ax = fig.add_subplot(gs[s, 4 + o // 2]); S.no_axes(ax)
            ax.imshow(np.fft.fftshift(bank[k]), cmap=S.SEQ_BLUE)
            if s == 0:
                ax.set_title(f"{int(180 * o / 8)}°", fontsize=9)
            if o == 0:
                ax.set_ylabel(f"scale {s + 1}", fontsize=9)
    # responses for the same filters
    for s in range(4):
        for o in range(0, 8, 2):
            k = s * 8 + o
            ax = fig.add_subplot(gs[s, 8 + o // 2]); S.no_axes(ax)
            ax.imshow(resp[k], cmap="gray")
            if s == 0:
                ax.set_title(f"{int(180 * o / 8)}°", fontsize=9)
    fig.text(0.50, 0.95, "Gabor filters (frequency domain)", ha="center", fontsize=11, weight="bold")
    fig.text(0.78, 0.95, "Filter responses |I * g|", ha="center", fontsize=11, weight="bold")
    fig.suptitle("GIST: 4 scales x 8 orientations, averaged on a 4x4 grid -> 512-d descriptor",
                 y=1.03, fontsize=13, weight="bold")
    fig.savefig(OUT / "gist_filters.png"), plt.close(fig)

    # Torralba-style polar "showGist" visualisation of the 512-d vector
    g = F.gist_features(img).reshape(32, 4, 4)  # filter, block-col, block-row
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    axes[0].imshow(rgb(img)); S.no_axes(axes[0]); axes[0].set_title("Input with 4x4 GIST grid")
    for t in (64, 128, 192):
        axes[0].axhline(t, color="white", lw=1); axes[0].axvline(t, color="white", lw=1)
    canvas = np.zeros((4 * 65, 4 * 65))
    yy, xx = np.mgrid[-32:33, -32:33]
    ang = (np.arctan2(yy, xx)) % np.pi
    rad = np.hypot(yy, xx)
    for bi in range(4):
        for bj in range(4):
            patch = np.zeros((65, 65))
            for s in range(4):
                ring = (rad >= 6 + 6.5 * (3 - s)) & (rad < 6 + 6.5 * (4 - s))
                for o in range(8):
                    wedge = np.abs(((ang - np.pi * o / 8 + np.pi / 2) % np.pi) - np.pi / 2) < np.pi / 16
                    patch[ring & wedge] = g[s * 8 + o, bj, bi]
            canvas[bi * 65:(bi + 1) * 65, bj * 65:(bj + 1) * 65] = patch
    axes[1].imshow(canvas, cmap=S.SEQ_BLUE); S.no_axes(axes[1])
    axes[1].set_title("GIST energy per block (rings = scale, wedges = orientation)")
    fig.tight_layout()
    fig.savefig(OUT / "gist_descriptor.png"), plt.close(fig)


def fig_hu_haralick(df):
    hu = np.load(C.FEATURE_DIR / "hu.npy")
    har = np.load(C.FEATURE_DIR / "haralick.npy")
    names = C.SHORT_NAMES
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, data, title in ((axes[0], hu[:, 0], "Hu moment h1 (log-scaled)"),
                            (axes[1], har[:, 1], "Haralick contrast"),
                            (axes[2], har[:, 8], "Haralick entropy")):
        groups = [data[(df.label == i).to_numpy()] for i in range(len(names))]
        bp = ax.boxplot(groups, orientation="horizontal", patch_artist=True, showfliers=False, widths=0.6,
                        medianprops=dict(color=S.TEXT, linewidth=1.5))
        for p in bp["boxes"]:
            p.set_facecolor("#cde2fb"), p.set_edgecolor(S.BLUE)
        ax.set_yticks(range(1, len(names) + 1), names)
        ax.set_title(title), ax.grid(axis="y", visible=False)
    fig.suptitle("Distribution of shape (Hu) and texture (Haralick) features per artist",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "hu_haralick_by_artist.png"), plt.close(fig)


def fig_feature_costs():
    t = pd.read_csv(C.RESULTS_DIR / "feature_extraction_times.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, title, fmt in ((axes[0], "dim", "Descriptor length", "{:.0f}"),
                                (axes[1], "ms_per_image", "Extraction time per image (ms, 1 CPU core)", "{:.1f}")):
        tt = t.sort_values(col)
        ax.barh(tt.label, tt[col], color=S.BLUE, height=0.6)
        for y, v in enumerate(tt[col]):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8.5, color=S.TEXT_2)
        ax.set_title(title), ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "feature_dims_and_times.png"), plt.close(fig)


def fig_pca_and_embedding(df):
    """Explained variance of PCA(100) and a 2-D t-SNE of the GIST+Hu+Color-hist PCA space."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import MinMaxScaler

    train = (df.split == "train").to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for combo, color in ((("gist",), S.SERIES[0]), (("gist", "hu", "color_hist"), S.SERIES[1]),
                         (("hog",), S.SERIES[2])):
        X = np.hstack([np.load(C.FEATURE_DIR / f"{n}.npy") for n in combo])
        Xs = MinMaxScaler().fit(X[train]).transform(X[train])
        p = PCA(n_components=min(200, X.shape[1]), random_state=C.SEED).fit(Xs)
        cum = np.cumsum(p.explained_variance_ratio_)
        lbl = " + ".join(C.FEATURE_LABELS[n] for n in combo)
        ax.plot(np.arange(1, len(cum) + 1), cum, color=color, label=f"{lbl} ({100 * cum[99]:.0f}% @100)")
    ax.axvline(100, color=S.TEXT_2, lw=1, ls="--")
    ax.text(102, 0.45, "100 components", color=S.TEXT_2, fontsize=9)
    ax.set_xlabel("number of principal components"), ax.set_ylabel("cumulative explained variance")
    ax.set_ylim(0, 1.02), ax.legend(loc="lower right")
    ax.set_title("PCA on Min-Max-scaled features (fit on training split)")
    fig.savefig(OUT / "pca_explained_variance.png"), plt.close(fig)

    X = np.hstack([np.load(C.FEATURE_DIR / f"{n}.npy") for n in ("gist", "hu", "color_hist")])
    sc = MinMaxScaler().fit(X[train])
    Z = PCA(100, random_state=C.SEED).fit(sc.transform(X[train])).transform(sc.transform(X))
    E = TSNE(2, random_state=C.SEED, init="pca", perplexity=40).fit_transform(Z)
    highlight = {"Giovanni Battista Piranesi": S.SERIES[0], "Pierre-Auguste Renoir": S.SERIES[1], "Ivan Aivazovsky": S.SERIES[2]}
    fig, ax = plt.subplots(figsize=(8, 7))
    other = ~df.artist.isin(highlight).to_numpy()
    ax.scatter(E[other, 0], E[other, 1], s=5, color=S.MUTED, alpha=0.45, label="other 12 artists", linewidths=0)
    for a, col in highlight.items():
        m = (df.artist == a).to_numpy()
        ax.scatter(E[m, 0], E[m, 1], s=11, color=col, label=a, edgecolors=S.SURFACE, linewidths=0.3)
    S.no_axes(ax)
    ax.legend(loc="upper right", markerscale=2)
    ax.set_title("t-SNE of PCA(100) features (GIST + Hu + Color hist)")
    fig.savefig(OUT / "tsne_features.png"), plt.close(fig)


if __name__ == "__main__":
    df = load_splits()
    steps = [fig_dataset_grid, fig_class_distribution, fig_feature_overview, fig_sift_keypoints, fig_hog,
             fig_color_hist_comparison, fig_gist]
    if (C.FEATURE_DIR / "gist.npy").exists():
        steps += [fig_mean_saturation, fig_hu_haralick, fig_feature_costs, fig_pca_and_embedding]
    for step in steps:
        print("->", step.__name__)
        step(df) if step.__code__.co_argcount else step()
    print(f"figures saved to {OUT}")
