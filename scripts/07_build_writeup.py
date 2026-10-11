"""Step 7 - write Section 3 (Feature Extraction) and Section 4.1 (SVM) of the write-up.

Every number is read from results/, so re-running the pipeline keeps the text in sync.
Produces report/Section3_Section4.1_writeup.md and .docx (with figures and tables).
"""
import _path  # noqa: F401

import json
import re

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from artist_id import config as C
from artist_id.data import load_splits

R, FIG = C.RESULTS_DIR, C.FIG_DIR
OUT_DIR = C.ROOT / "report"
OUT_DIR.mkdir(exist_ok=True)

final = json.load(open(R / "final_models.json"))
svm, lr = final["svm"], final["lr"]
search = pd.read_csv(R / "combination_search.csv")
times = pd.read_csv(R / "feature_extraction_times.csv").set_index("feature")
table = pd.read_csv(R / "final_results.csv")
df = load_splits()
n_split = df.split.value_counts()
svm_s = search[search.model == "svm"].sort_values("val_acc", ascending=False)
single = search[(search.model == "svm") & (search.n_features == 1)].set_index("combo")


def pct(x):
    return f"{100 * x:.1f}%"


def g(x):
    return f"{x:g}"


PRIMAL = r"\mathrm{min}_{w,b,\xi}\ \frac{1}{2}\lVert w\rVert^{2}+C\sum_{i=1}^{n}\xi_{i}\text{   subject to   }y_{i}\left(w^{T}\phi(x_{i})+b\right)\geq 1-\xi_{i},\ \xi_{i}\geq 0"
DUAL = r"\mathrm{max}_{\alpha}\ \sum_{i}\alpha_{i}-\frac{1}{2}\sum_{i}\sum_{j}\alpha_{i}\alpha_{j}y_{i}y_{j}K(x_{i},x_{j})\text{   subject to   }0\leq\alpha_{i}\leq C,\ \sum_{i}\alpha_{i}y_{i}=0"
KERNEL = r"K(x,x')=\exp\left(-\gamma\lVert x-x'\rVert^{2}\right)"
DECISION = r"\hat{y}=\mathrm{argmax}_{k}\left(\sum_{i}\alpha_{i}^{(k)}y_{i}^{(k)}K(x_{i},x)+b_{k}\right)"


def explained_variance(combo):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    X = np.hstack([np.load(C.FEATURE_DIR / f"{n}.npy") for n in combo.split("+")])
    tr = (df.split == "train").to_numpy()
    p = PCA(min(100, X.shape[1]), random_state=C.SEED).fit(MinMaxScaler().fit_transform(X[tr]))
    return p.explained_variance_ratio_.sum()


ev = explained_variance(svm["combo"])
# Marginal value of each descriptor: mean change in SVM validation accuracy when it is added.
delta = {}
for f in C.FEATURES:
    diffs = []
    for _, r in svm_s.iterrows():
        parts = r.combo.split("+")
        if f in parts and len(parts) > 1:
            base = svm_s[svm_s.combo == "+".join(p for p in parts if p != f)]
            if len(base):
                diffs.append(r.val_acc - base.val_acc.iloc[0])
    delta[f] = 100 * np.mean(diffs)
delta = pd.Series(delta).sort_values()

cm = np.load(R / "confusion_svm.npy")
rep = pd.read_csv(R / "classification_report_svm.csv", index_col=0).loc[C.ARTISTS]
best_f1 = rep["f1-score"].sort_values(ascending=False)
off = cm.copy()
np.fill_diagonal(off, 0)
sym = np.triu(off + off.T, 1)  # errors between each unordered pair of artists
order = np.dstack(np.unravel_index(np.argsort(-sym, axis=None), sym.shape))[0]
pairs = [(C.ARTISTS[i], C.ARTISTS[j], int(sym[i, j])) for i, j in order[:2]]
per_artist = df.groupby("artist").size()


def short(name):
    return name.split()[-1]

# ------------------------------------------------------------------ content (markdown-ish blocks)
blocks = []
H = lambda level, text: blocks.append(("h", level, text))  # noqa: E731
P = lambda text: blocks.append(("p", text))  # noqa: E731
B = lambda items: blocks.append(("ul", items))  # noqa: E731
EQ = lambda text, latex: blocks.append(("eq", text, latex))  # noqa: E731
IMG = lambda path, width, cap: blocks.append(("img", path, width, cap))  # noqa: E731
TAB = lambda frame, cap: blocks.append(("table", frame, cap))  # noqa: E731

H(1, "3 Dataset and Features")
H(2, "3.1 Dataset")
P(f"We use the team's shared subset of the Kaggle \"Painters by Numbers\" collection (WikiArt): {len(df):,} paintings "
  f"by 15 artists ({per_artist.min()}-{per_artist.max()} each), with a fixed train/validation/test split of "
  f"{n_split['train']:,} / {n_split['val']} / {n_split['test']} images that the CNN uses as well. Every painting is "
  f"resized to 256 x 256 RGB, as in the paper. The classes are nearly balanced, so accuracy is a meaningful headline "
  f"metric (chance = 6.7%).")

H(2, "3.2 Feature Extraction")
P("Each painting is described by six hand-crafted descriptors, computed with OpenCV, Mahotas and scikit-image. "
  "Every one of the 63 non-empty combinations was evaluated by concatenating the descriptors horizontally.")
B([
    f"**GIST** ({int(times.loc['gist','dim'])}-d). The spatial-envelope descriptor of Oliva & Torralba. The grayscale image is "
    "log-transformed, whitened and contrast-normalised, then filtered by a bank of 4 scales x 8 orientations of Gabor filters "
    "built in the frequency domain. The mean response energy of each filter on a 4 x 4 grid gives 32 x 16 = 512 values, "
    "summarising scene properties such as naturalness, openness, roughness, expansion and ruggedness. "
    "We ported Torralba's reference implementation to NumPy because the C library (lear-gist) needs FFTW.",
    f"**Hu moments** ({int(times.loc['hu','dim'])}-d). Seven polynomial combinations of normalised central image moments "
    "(OpenCV), invariant to translation, scale and rotation. They are log-scaled (−sign(h)·log₁₀|h|) because raw values span "
    "about 20 orders of magnitude.",
    f"**Color histogram** ({int(times.loc['color_hist','dim'])}-d). HSV values quantised into 8 bins per channel; counting each "
    "of the 8³ quantised colours gives an L1-normalised joint histogram. It records only which colours appear, not where.",
    f"**SIFT** ({int(times.loc['sift','dim'])}-d). Up to {C.SIFT_MAX_KEYPOINTS} scale-invariant keypoints per image with 128-d "
    f"descriptors. These are encoded as a bag of visual words: a k-means vocabulary of {C.SIFT_VOCAB_SIZE} words is learnt "
    "from training-image descriptors only, and each image becomes a normalised word histogram.",
    f"**HOG** ({int(times.loc['hog','dim'])}-d). Histogram of oriented gradients (scikit-image) on a 128 x 128 grayscale "
    "image: 9 orientation bins, 16 x 16-pixel cells and 2 x 2-cell blocks with L2-Hys normalisation, capturing edge and "
    "brush-stroke direction.",
    f"**Haralick textures** ({int(times.loc['haralick','dim'])}-d). The 13 grey-level co-occurrence statistics (Mahotas), e.g. "
    "contrast, correlation, entropy and homogeneity, averaged over four directions.",
])
IMG(FIG / "features" / "feature_overview.png", 6.5,
    "Figure 1: Feature extraction on a Renoir painting: (a) input, (b) HSV colour histogram, (c) local binary pattern, "
    "(d) SIFT keypoints, (e) HOG gradients.")
P(f"**Preprocessing.** Each concatenated feature vector is Min-Max scaled to [0, 1] so that long descriptors (HOG, 1,764-d) "
  f"cannot dominate short ones (Hu, 7-d). PCA then keeps the first {C.PCA_COMPONENTS} principal components "
  f"({pct(ev)} of the variance for the final feature set). The scaler and PCA are fit on the training split only, and "
  "validation/test images are projected onto the same components. Both steps sit inside the scikit-learn pipeline, so they "
  "are also re-fit inside every cross-validation fold.")

best_single = single.sort_values("val_acc", ascending=False)
top3 = ", ".join(f"{C.FEATURE_LABELS[f]} ({pct(v)})" for f, v in best_single.val_acc.head(3).items())
weakest = best_single.index[-1]
hurts = [f for f, d in delta.items() if d < 0]
P(f"**Which features help.** On their own, the strongest descriptors are {top3} validation accuracy with the SVM, "
  f"while {C.FEATURE_LABELS[weakest]} alone is weakest ({pct(best_single.val_acc.iloc[-1])}). The best combination was "
  f"**{svm['features']}**. Averaged over all combinations, adding a descriptor changed SVM validation accuracy by "
  + ", ".join(f"{d:+.1f} pp ({C.FEATURE_LABELS[f]})" for f, d in delta[::-1].items()) + ". "
  + ((f"{' and '.join(C.FEATURE_LABELS[f] for f in hurts)} lower{'s' if len(hurts) == 1 else ''} accuracy on average"
      + (", most likely because HOG's 1,764 position-specific gradient values dominate the 100 principal components "
         "without generalising across paintings" if "hog" in hurts else "")
      + ". Like the paper, we find that only a subset of descriptors is useful"
      + (", although here Haralick helps rather than hurts. " if delta["haralick"] > 0 else ". "))
     if hurts else "Every descriptor helps on average. ")
  + f"Extraction cost is dominated by GIST ({times.loc['gist','ms_per_image']:.0f} ms/image on one CPU core), whereas Hu moments and "
  f"colour histograms cost about 1 ms.")

H(1, "4 Methods")
H(2, "4.1 SVM: Formulation & Optimization")
P("The classifier is a multi-class soft-margin SVM with a Gaussian (RBF) kernel, trained one-vs-rest: 15 binary SVMs, each "
  "separating one artist (yᵢ = +1) from the other fourteen (yᵢ = −1). With φ the feature map implied by the kernel, each "
  "binary SVM solves")
EQ("min ½‖w‖² + C Σ ξᵢ  s.t.  yᵢ(wᵀφ(xᵢ) + b) ≥ 1 − ξᵢ, ξᵢ ≥ 0", PRIMAL)
P("where the slack variables ξᵢ allow margin violations and C trades margin width against training error. In practice the "
  "dual problem is solved (libsvm's SMO algorithm, via scikit-learn's SVC):")
EQ("max Σ αᵢ − ½ ΣΣ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)  s.t.  0 ≤ αᵢ ≤ C, Σ αᵢyᵢ = 0", DUAL)
EQ("K(x,x′) = exp(−γ‖x − x′‖²)", KERNEL)
EQ("decision", DECISION)
P("A painting is assigned to the artist whose classifier returns the largest decision value (last expression). "
  "The input is the 100-d PCA vector, and γ sets how "
  "local the kernel is: large γ fits individual training points, small γ approaches a linear model.")
P(f"**Hyper-parameter optimisation.** The feature set, C and γ were chosen in two stages, using only the training and "
  f"validation splits. (1) *Screening*: for all 63 combinations, a 3-fold stratified cross-validated grid search over "
  f"C ∈ {{{', '.join(g(c) for c in C.COARSE_C_GRID)}}} and γ ∈ {{{', '.join(g(x) for x in C.COARSE_GAMMA_GRID)}}}. "
  f"This stage used libsvm's native one-vs-one multi-class scheme, which has the same kernel and C/γ meaning but is about 7x "
  f"faster. (2) *Full search*: the {len(final['svm_candidates'])} combinations with the best validation accuracy were "
  f"re-tuned with the paper's grid, C, γ ∈ {{10⁻⁵, 10⁻⁴, …, 10⁵}} (121 settings x 3 folds), using the one-vs-rest SVMs. "
  f"The final model is the candidate with the highest validation accuracy, refit on the full training split: "
  f"**{svm['features']}, C = {g(svm['C'])}, γ = {g(svm['gamma'])}** (CV accuracy {pct(svm['cv_acc'])}). "
  f"A multinomial Logistic Regression on the same pipeline, with its own best feature set ({lr['features']}) and C, is the baseline.")
IMG(FIG / "models" / "svm_grid_search_heatmap.png", 6.5,
    f"Figure 2: 3-fold CV (left) and training-fold (right) accuracy over the C x γ grid for {svm['features']}.")

tab = table[["Model", "Training Accuracy (%)", "Test Accuracy (%)", "Train Time (s)", "Inference Time (ms/img)",
             "Features Used"]].copy()
tab.columns = ["Model", "Train acc. (%)", "Test acc. (%)", "Train time (s)", "Inference (ms/img)", "Features"]
tab["Model"] = tab["Model"].replace({"LR*": "LR (baseline)"})
TAB(tab, "Table 1: Classical models on the held-out test set. Inference time includes feature extraction on one CPU core.")
P(f"**Results.** The tuned RBF SVM reaches **{pct(svm['test_acc'])} test accuracy** (macro-F1 {svm['macro_f1']:.3f}) against "
  f"{pct(lr['test_acc'])} for Logistic Regression (the paper reports 68.1% and 61.2% on a different set of 15 artists). Like the paper, the SVM overfits heavily, with "
  f"{pct(svm['train_acc'])} training accuracy. The grid shows that settings with a smaller train-validation gap "
  f"(small γ or C) also have lower validation accuracy, so regularisation alone does not fix this. Using fewer principal "
  f"components or more training data are the natural next steps. Best recognised are {best_f1.index[0]} "
  f"(F1 {best_f1.iloc[0]:.2f}) and {best_f1.index[1]} (F1 {best_f1.iloc[1]:.2f}); hardest is {best_f1.index[-1]} "
  f"(F1 {best_f1.iloc[-1]:.2f}). The most confused pairs are {pairs[0][0]} / {pairs[0][1]} ({pairs[0][2]} test errors) "
  f"and {pairs[1][0]} / {pairs[1][1]} ({pairs[1][2]}). Training "
  f"takes {svm['train_time_s']:.1f} s on a CPU, but inference costs {svm['total_inference_ms']:.0f} ms per image, "
  f"almost all of it feature extraction.")


full_blocks = blocks

# ------------------------------------------------------------------ condensed version (fits the 2-page write-up)
blocks = []
H(1, "3 Dataset and Features")
P(f"**Data.** The team's shared set of {len(df):,} Painters-by-Numbers paintings by 15 artists, resized to 256 x 256, "
  f"with a fixed split of {n_split['train']:,} / {n_split['val']} / {n_split['test']} (train / val / test), "
  f"the same split the CNN uses.")
P(f"**Feature extraction.** Six descriptors were extracted with OpenCV, Mahotas and scikit-image: "
  f"**GIST** (512-d; 4 scales x 8 orientations of Gabor energy on a 4 x 4 grid, describing the scene's spatial envelope), "
  f"**Hu moments** (7 log-scaled shape invariants), **colour histogram** (512-bin joint HSV, 8 bins per channel), "
  f"**SIFT** (bag of {C.SIFT_VOCAB_SIZE} visual words from a k-means codebook learnt on training images), "
  f"**HOG** (1,764-d; 9 orientations, 16 px cells) and **Haralick** textures (13 GLCM statistics). "
  f"All 63 combinations were concatenated and tested. Each vector is Min-Max scaled to [0, 1] and reduced by PCA to "
  f"{C.PCA_COMPONENTS} components ({pct(ev)} of the variance), both fit on the training split only. "
  f"Averaged over all combinations, adding a descriptor changed validation accuracy by "
  + ", ".join(f"{d:+.1f} pp ({C.FEATURE_LABELS[f]})" for f, d in delta[::-1].items())
  + f". The best set was **{svm['features']}**.")
H(1, "4.1 SVM: Formulation & Optimization")
P("We train 15 one-vs-rest soft-margin SVMs with an RBF kernel on the 100-d PCA features. Each solves")
EQ("primal", PRIMAL)
EQ("kernel", KERNEL)
EQ("decision", DECISION)
P(f"through its dual (libsvm SMO). C penalises margin violations ξᵢ, and γ sets the kernel width. C, γ ∈ {{10⁻⁵, …, 10⁵}} "
  f"were tuned by 3-fold stratified CV grid search (121 settings) on the {len(final['svm_candidates'])} best feature sets, "
  f"after a coarse screening of all 63. The best model, C = {g(svm['C'])} and γ = {g(svm['gamma'])} "
  f"(CV {pct(svm['cv_acc'])}), reaches **{pct(svm['test_acc'])} test accuracy** against {pct(lr['test_acc'])} for the "
  f"Logistic-Regression baseline (paper, on different artists: 68.1% / 61.2%). The {pct(svm['train_acc'])} training accuracy shows strong "
  f"overfitting, which tuning C does not remove. {short(best_f1.index[0])} and {short(best_f1.index[1])} are recognised "
  f"best (F1 ≥ {min(best_f1.iloc[0], best_f1.iloc[1]):.2f}), and {short(pairs[0][0])}/{short(pairs[0][1])} and "
  f"{short(pairs[1][0])}/{short(pairs[1][1])} are the most confused pairs.")
short_blocks = blocks

# ------------------------------------------------------------------ writers
def md_inline(t):
    return t


def write_md(path, blocks):
    lines = []
    for b in blocks:
        if b[0] == "h":
            lines += ["#" * (b[1] + 1) + " " + b[2], ""]
        elif b[0] == "p":
            lines += [b[1], ""]
        elif b[0] == "ul":
            lines += [f"- {i}" for i in b[1]] + [""]
        elif b[0] == "eq":
            lines += [f"$$\\text{{{b[1]}}}$$" if False else f"    {b[1]}", ""]
        elif b[0] == "img":
            rel = b[1].relative_to(C.ROOT).as_posix()
            lines += [f"![{b[3]}](../{rel})", f"*{b[3]}*", ""]
        elif b[0] == "table":
            fr = b[1]
            lines += ["| " + " | ".join(fr.columns) + " |", "|" + "---|" * len(fr.columns)]
            lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in fr.itertuples(index=False)]
            lines += [f"*{b[2]}*", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


_XSLT = None


def omml(latex):
    """LaTeX -> MathML -> native Word equation (Office's own MML2OMML.XSL)."""
    global _XSLT
    import latex2mathml.converter
    from lxml import etree

    if _XSLT is None:
        _XSLT = etree.XSLT(etree.parse(r"C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL"))
    return _XSLT(etree.fromstring(latex2mathml.converter.convert(latex))).getroot()


def add_rich(par, text, size=None):
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            run = par.add_run(part[2:-2]); run.bold = True
        elif part.startswith("*"):
            run = par.add_run(part[1:-1]); run.italic = True
        else:
            run = par.add_run(part)
        if size:
            run.font.size = Pt(size)


def write_docx(path, blocks):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(4)
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Inches(0.8)
        sec.top_margin = sec.bottom_margin = Inches(0.7)
    for b in blocks:
        if b[0] == "h":
            h = doc.add_heading(b[2], level=b[1])
            for r in h.runs:
                r.font.color.rgb = RGBColor(0x1B, 0x24, 0x30)
                r.font.name = "Times New Roman"
        elif b[0] == "p":
            p = doc.add_paragraph(); add_rich(p, b[1]); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif b[0] == "ul":
            for item in b[1]:
                p = doc.add_paragraph(style="List Bullet"); add_rich(p, item); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif b[0] == "eq":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p._p.append(omml(b[2]))
        elif b[0] == "img":
            doc.add_picture(str(b[1]), width=Inches(b[2]))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            p = doc.add_paragraph(); add_rich(p, b[3], 9); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif b[0] == "table":
            fr = b[1]
            t = doc.add_table(rows=1, cols=len(fr.columns))
            t.style = "Light Grid Accent 1"
            for i, c in enumerate(fr.columns):
                t.rows[0].cells[i].text = str(c)
            for row in fr.itertuples(index=False):
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = str(v)
            for row in t.rows:
                for cell in row.cells:
                    for par in cell.paragraphs:
                        for r in par.runs:
                            r.font.size = Pt(9)
            p = doc.add_paragraph(); add_rich(p, b[2], 9); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.save(path)


def write_summary(path):
    """Small JSON of derived numbers for the slide builder (slides/build_slides.js)."""
    path.write_text(json.dumps({
        "n_images": int(len(df)),
        "split": {k: int(v) for k, v in n_split.items()},
        "per_artist_min": int(per_artist.min()), "per_artist_max": int(per_artist.max()),
        "delta_pp": {C.FEATURE_LABELS[f]: round(float(d), 1) for f, d in delta[::-1].items()},
        "confused_pairs": [[short(a), short(b), n] for a, b, n in pairs],
        "best_f1": [[short(a), round(float(v), 2)] for a, v in best_f1.head(2).items()],
        "worst_f1": [short(best_f1.index[-1]), round(float(best_f1.iloc[-1]), 2)],
        "pca_variance": round(float(ev), 3),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    write_summary(R / "summary_stats.json")
    write_md(OUT_DIR / "Section3_Section4.1_writeup.md", full_blocks)
    write_docx(OUT_DIR / "Section3_Section4.1_writeup.docx", full_blocks)
    write_md(OUT_DIR / "Section3_Section4.1_condensed.md", short_blocks)
    write_docx(OUT_DIR / "Section3_Section4.1_condensed.docx", short_blocks)
    print("wrote", OUT_DIR / "Section3_Section4.1_writeup.docx")
