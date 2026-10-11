"""Step 6 - model-evaluation figures (figures/models/) from the CSV/JSON results of steps 4-5."""
import _path  # noqa: F401

import json

import cv2
import numpy as np
import pandas as pd

from artist_id import config as C
from artist_id import plotstyle as S
from artist_id.data import load_image

plt = S.plt
OUT = C.FIG_DIR / "models"
OUT.mkdir(parents=True, exist_ok=True)
R = C.RESULTS_DIR

search = pd.read_csv(R / "combination_search.csv")
final = json.load(open(R / "final_models.json"))
svm_s = search[search.model == "svm"].sort_values("val_acc", ascending=False)
lr_s = search[search.model == "lr"].sort_values("val_acc", ascending=False)


def fig_top_combinations(n=15):
    top = svm_s.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    colors = [S.ORANGE if i == n - 1 else S.BLUE for i in range(n)]
    ax.barh(top.combo_label, 100 * top.val_acc, color=colors, height=0.65)
    for y, v in enumerate(100 * top.val_acc):
        ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=8.5, color=S.TEXT_2)
    ax.set_xlim(100 * top.val_acc.min() - 8, 100 * top.val_acc.max() + 4)
    ax.set_xlabel("validation accuracy (%)"), ax.grid(axis="y", visible=False)
    ax.set_title(f"RBF-SVM: top {n} of 63 feature combinations")
    fig.savefig(OUT / "top_feature_combinations.png"), plt.close(fig)


def fig_single_features():
    single = search[search.n_features == 1].copy()
    single["label"] = single.combo.map(C.FEATURE_LABELS)
    piv = single.pivot(index="label", columns="model", values="val_acc").sort_values("svm")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(piv))
    ax.barh(y + 0.18, 100 * piv.svm, height=0.34, color=S.BLUE, label="RBF SVM")
    ax.barh(y - 0.18, 100 * piv.lr, height=0.34, color=S.ORANGE, label="Logistic Regression")
    for yy, v in zip(y, 100 * piv.svm):
        ax.text(v + 0.4, yy + 0.18, f"{v:.1f}", va="center", fontsize=8, color=S.TEXT_2)
    for yy, v in zip(y, 100 * piv.lr):
        ax.text(v + 0.4, yy - 0.18, f"{v:.1f}", va="center", fontsize=8, color=S.TEXT_2)
    ax.axvline(100 / 15, color=S.TEXT_2, ls="--", lw=1)
    ax.text(100 / 15 + 0.5, len(piv) - 0.4, "chance 6.7%", fontsize=8, color=S.TEXT_2)
    ax.set_yticks(y, piv.index), ax.grid(axis="y", visible=False)
    ax.set_xlabel("validation accuracy (%)"), ax.legend(loc="lower right")
    ax.set_title("Each descriptor on its own")
    fig.savefig(OUT / "single_feature_accuracy.png"), plt.close(fig)


def fig_feature_contribution():
    """Mean change in SVM validation accuracy when a descriptor is added to a combination."""
    deltas = {}
    for f in C.FEATURES:
        with_f = svm_s[svm_s.combo.str.split("+").apply(lambda c: f in c)]
        diffs = []
        for _, r in with_f.iterrows():
            rest = "+".join(x for x in r.combo.split("+") if x != f)
            if rest:
                base = svm_s[svm_s.combo == rest]
                if len(base):
                    diffs.append(r.val_acc - base.val_acc.iloc[0])
        deltas[C.FEATURE_LABELS[f]] = 100 * np.mean(diffs)
    s = pd.Series(deltas).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(s.index, s.values, color=[S.SERIES[7] if v < 0 else S.BLUE for v in s.values], height=0.6)
    for y, v in enumerate(s.values):
        ax.text(v + (0.15 if v >= 0 else -0.15), y, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=S.TEXT_2)
    ax.axvline(0, color=S.TEXT_2, lw=1)
    ax.set_xlabel("avg. change in validation accuracy when added (percentage points)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Marginal value of each descriptor (SVM, over all 63 combinations)")
    fig.savefig(OUT / "feature_contribution.png"), plt.close(fig)


def fig_grid_heatmap():
    best = final["svm"]["combo"]
    g = pd.read_csv(R / "grid_search" / f"{best}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, col, title in ((axes[0], "mean_test_score", "3-fold CV accuracy"),
                           (axes[1], "mean_train_score", "training-fold accuracy")):
        piv = g.pivot(index="C", columns="gamma", values=col).sort_index(ascending=False)
        im = ax.imshow(100 * piv.values, cmap=S.SEQ_BLUE, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(piv.columns)), [f"1e{int(np.log10(v))}" for v in piv.columns], rotation=45)
        ax.set_yticks(range(len(piv.index)), [f"1e{int(np.log10(v))}" for v in piv.index])
        ax.set_xlabel("gamma (γ)"), ax.set_ylabel("C"), ax.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = 100 * piv.values[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 55 else S.TEXT)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="accuracy (%)")
    bi = g.mean_test_score.idxmax()
    fig.suptitle(f"Grid search over C and γ - {final['svm']['features']} "
                 f"(best C={g.C[bi]:g}, γ={g.gamma[bi]:g}, CV={100 * g.mean_test_score[bi]:.1f}%)",
                 fontsize=12.5, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "svm_grid_search_heatmap.png"), plt.close(fig)


def fig_accuracy_comparison():
    rows = [final["lr"]] + final["svm_candidates"]
    labels = [("LR (baseline)\n" if r["name"] == "LR" else "SVM\n") + r["features"].replace(" + ", "\n+ ") for r in rows]
    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(9, 2.1 * len(rows)), 5.4))
    for k, (key, name, col) in enumerate((("train_acc", "train", S.SERIES[0]), ("val_acc", "validation", S.SERIES[1]),
                                          ("test_acc", "test", S.SERIES[2]))):
        vals = [100 * r[key] for r in rows]
        ax.bar(x + (k - 1) * w, vals, width=w - 0.03, color=col, label=name)
        for xx, v in zip(x, vals):
            ax.text(xx + (k - 1) * w, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.5, color=S.TEXT_2)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("accuracy (%)"), ax.set_ylim(0, 118), ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=3)
    ax.set_title("Train / validation / test accuracy of the classical models", loc="left")
    fig.savefig(OUT / "accuracy_comparison.png"), plt.close(fig)


def fig_confusion(tag, title):
    cm = np.load(R / f"confusion_{tag}.npy")
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cmn, cmap=S.SEQ_BLUE, vmin=0, vmax=1)
    n = len(C.ARTISTS)
    ax.set_xticks(range(n), [f"{i} {s}" for i, s in enumerate(C.SHORT_NAMES)], rotation=60, ha="right")
    ax.set_yticks(range(n), [f"{i} {s}" for i, s in enumerate(C.SHORT_NAMES)])
    ax.set_xlabel("predicted artist"), ax.set_ylabel("true artist"), ax.grid(False)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7.5,
                        color="white" if cmn[i, j] > 0.5 else S.TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="row-normalised (recall)")
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title} - test accuracy {100 * acc:.1f}% (counts shown)")
    fig.savefig(OUT / f"confusion_matrix_{tag}.png"), plt.close(fig)


def fig_precision_recall(tag, title):
    rep = pd.read_csv(R / f"classification_report_{tag}.csv", index_col=0).loc[C.ARTISTS]
    rep = rep.iloc[::-1]
    y = np.arange(len(rep))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(y + 0.19, 100 * rep.precision, height=0.36, color=S.BLUE, label="precision")
    ax.barh(y - 0.19, 100 * rep.recall, height=0.36, color=S.ORANGE, label="recall")
    ax.set_yticks(y, rep.index), ax.set_xlim(0, 105)
    ax.set_xlabel("%"), ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=2, bbox_to_anchor=(1, -0.13))
    ax.set_title(title)
    fig.savefig(OUT / f"precision_recall_{tag}.png"), plt.close(fig)


def fig_time_vs_error():
    """Paper Fig. 8: every feature combination as a point (validation error vs inference time)."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for d, col, name in ((svm_s, S.BLUE, "RBF SVM"), (lr_s, S.ORANGE, "Logistic Regression")):
        ax.scatter(d.total_inference_ms, 100 * (1 - d.val_acc), s=34, color=col, label=name,
                   edgecolors=S.SURFACE, linewidths=0.8)
    b = svm_s.iloc[0]
    ax.annotate(b.combo_label, (b.total_inference_ms, 100 * (1 - b.val_acc)), xytext=(15, -25),
                textcoords="offset points", fontsize=8.5, color=S.TEXT,
                arrowprops=dict(arrowstyle="-", color=S.TEXT_2, lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("inference time per image incl. feature extraction (ms, log scale)")
    ax.set_ylabel("validation error (%)"), ax.legend(loc="upper right")
    ax.set_title("Inference time vs. error for all 63 feature combinations (closer to origin = better)")
    fig.savefig(OUT / "inference_time_vs_error.png"), plt.close(fig)


def fig_overfitting():
    fig, ax = plt.subplots(figsize=(7, 5.6))
    for d, col, name in ((svm_s, S.BLUE, "RBF SVM"), (lr_s, S.ORANGE, "Logistic Regression")):
        ax.scatter(100 * d.train_acc, 100 * d.val_acc, s=30, color=col, label=name,
                   edgecolors=S.SURFACE, linewidths=0.8)
    ax.plot([0, 100], [0, 100], color=S.TEXT_2, lw=1, ls="--")
    ax.text(70, 73, "train = validation", rotation=38, color=S.TEXT_2, fontsize=8.5)
    ax.set_xlim(0, 102), ax.set_ylim(0, 102)
    ax.set_xlabel("training accuracy (%)"), ax.set_ylabel("validation accuracy (%)")
    ax.legend(loc="upper left")
    ax.set_title("Generalisation gap: points far below the diagonal overfit")
    fig.savefig(OUT / "overfitting_train_vs_val.png"), plt.close(fig)


def fig_misclassified(tag="svm", n=8):
    p = pd.read_csv(R / f"predictions_{tag}.csv")
    wrong = p[p.true != p.pred].sample(min(n, (p.true != p.pred).sum()), random_state=C.SEED)
    fig, axes = plt.subplots(2, n // 2, figsize=(3.2 * n // 2, 7.4))
    for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
        ax.imshow(cv2.cvtColor(load_image(r.filename), cv2.COLOR_BGR2RGB)); S.no_axes(ax)
        ax.set_title(f"true: {r.true_artist.split()[-1]}\npred: {r.pred_artist.split()[-1]}", fontsize=10)
    fig.suptitle("Examples of SVM mistakes on the test set", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"misclassified_{tag}.png"), plt.close(fig)


if __name__ == "__main__":
    for f in (fig_top_combinations, fig_single_features, fig_feature_contribution, fig_grid_heatmap,
              fig_accuracy_comparison, fig_time_vs_error, fig_overfitting):
        print("->", f.__name__); f()
    fig_confusion("svm", "RBF SVM"); fig_confusion("lr", "Logistic Regression")
    fig_precision_recall("svm", f"RBF SVM - precision & recall per artist ({final['svm']['features']})")
    fig_precision_recall("lr", f"Logistic Regression - precision & recall per artist ({final['lr']['features']})")
    fig_misclassified("svm")
    print(f"figures saved to {OUT}")
