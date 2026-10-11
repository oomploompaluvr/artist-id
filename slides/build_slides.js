// Builds slides/Classical_ML_Feature_Engineering.pptx from the figures and results.
// Run from the project root:  node slides/build_slides.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const FIG = (p) => path.join(ROOT, "figures", p);
const RES = (p) => path.join(ROOT, "results", p);
const OUT = path.join(__dirname, "Classical_ML_Feature_Engineering.pptx");

// ------------------------------------------------------------------ data
const final = JSON.parse(fs.readFileSync(RES("final_models.json"), "utf8"));
const csv = (file) => {
  const [head, ...rows] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  return rows.map((r) => {
    const cells = r.match(/("([^"]|"")*"|[^,]*)(,|$)/g).map((c) => c.replace(/,$/, "").replace(/^"|"$/g, ""));
    return Object.fromEntries(cols.map((c, i) => [c, cells[i]]));
  });
};
const table = csv(RES("final_results.csv"));
const featTimes = csv(RES("feature_extraction_times.csv"));
const search = csv(RES("combination_search.csv"));
const pct = (v) => (100 * v).toFixed(1) + "%";
const svm = final.svm, lr = final.lr;
const stats = JSON.parse(fs.readFileSync(RES("summary_stats.json"), "utf8"));
const fmt = (n) => n.toLocaleString("en-US");
const deltas = Object.entries(stats.delta_pp);           // sorted best -> worst
const helps = deltas.filter(([, d]) => d > 0).slice(0, 2).map(([f]) => f);
const hurts = deltas.filter(([, d]) => d < 0).map(([f]) => f);
const cp = stats.confused_pairs;

// PNG size straight from the IHDR chunk, to keep aspect ratios
function pngSize(p) {
  const b = fs.readFileSync(p);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fitImage(slide, file, x, y, w, h, opts = {}) {
  const { w: iw, h: ih } = pngSize(file);
  const s = Math.min(w / iw, h / ih);
  const ww = iw * s, hh = ih * s;
  slide.addImage({ path: file, x: x + (w - ww) / 2, y: y + (h - hh) / 2, w: ww, h: hh, ...opts });
}

// ------------------------------------------------------------------ theme
const THEME = {
  name: "Gallery Ink",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2430", lt1: "FFFFFF", dk2: "3A4656", lt2: "F1F4F8",
    accent1: "C9772B", accent2: "2A78D6", accent3: "1BAF7A", accent4: "6B7A8F",
    accent5: "E34948", accent6: "4A3AA7", hlink: "2A78D6", folHlink: "4A3AA7",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Classical Machine Learning & Feature Engineering for Artist Identification";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const W = 13.333, H = 7.5, M = 0.6;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.2, w: W - 2 * M, h: 1.6,
      fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.95, w: W - 2 * M, h: 1.2,
      fontSize: 18, color: "D7DDE5", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.6, w: W - 2 * M, h: 1.2,
      fontFace: THEME.headFontFace, fontSize: 38, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.9, w: W - 2 * M, h: 0.9,
      fontSize: 18, color: "D7DDE5", valign: "top" }, text: "" } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.5, w: 0.5, h: 0.3, fontSize: 10, color: "9AA6B5" },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.85,
      fontFace: THEME.headFontFace, fontSize: 26, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Artist Identification  |  Classical ML & Feature Engineering",
      options: { x: M, y: H - 0.45, w: 7, h: 0.3, fontSize: 10, color: "8A94A3", margin: 0 } } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.45, w: 0.5, h: 0.3, fontSize: 10, color: "8A94A3" },
});

const TOP = 1.35, BOTTOM = H - 0.65; // content band for CONTENT slides

function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function bullets(slide, items, x, y, w, h, size = 16) {
  slide.addText(items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({ text: r.text, options: { ...r.options, bullet: j === 0 ? { indent: 18 } : undefined,
      breakLine: j === runs.length - 1 && i < items.length - 1 } }));
  }).flat(), { x, y, w, h, fontSize: size, color: C.text2, valign: "top", paraSpaceAfter: 8, isTextBox: true, margin: 0 });
}
function card(slide, x, y, w, h, fill = HEX.lt2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
}
function stat(slide, x, y, w, value, label, color = C.accent1) {
  slide.addText(value, { x, y, w, h: 0.8, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color, margin: 0, isTextBox: true });
  slide.addText(label, { x, y: y + 0.8, w, h: 0.5, fontSize: 13, color: C.text2, margin: 0, isTextBox: true });
}
function caption(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.35, fontSize: 11, italic: true, color: "6B7A8F", margin: 0, isTextBox: true });
}

// ================================================================== slides
// 1. Title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Classical Machine Learning & Feature Engineering", { placeholder: "title" });
s.addText("Artist identification from paintings with hand-crafted descriptors, PCA and an RBF Support Vector Machine",
  { placeholder: "body" });
s.addNotes("Person 1 section: feature extraction, preprocessing, Logistic Regression baseline and RBF-SVM with grid search.");

// 2. Pipeline
s = content("The classical pipeline: hand-crafted features, then a kernel classifier", "Introduction");
const steps = [
  ["Painting", "256 x 256 RGB"],
  ["Descriptors", "GIST, Hu, Color hist,\nSIFT, HOG, Haralick"],
  ["Concatenate", "63 feature\ncombinations"],
  ["Min-Max", "scale every\nfeature to [0, 1]"],
  ["PCA", "100 principal\ncomponents"],
  ["Classifier", "LR baseline &\nRBF SVM (OvR)"],
];
const bw = 1.78, gap = 0.27, x0 = (W - (steps.length * bw + (steps.length - 1) * gap)) / 2, by = 2.1;
steps.forEach(([t, d], i) => {
  const x = x0 + i * (bw + gap);
  card(s, x, by, bw, 1.9, i === steps.length - 1 ? "FBEBDD" : HEX.lt2);
  s.addText(String(i + 1), { x: x + 0.15, y: by + 0.15, w: 0.45, h: 0.45, shape: pres.shapes.OVAL, fill: { color: C.accent1 },
    color: C.background1, fontSize: 14, bold: true, align: "center", valign: "middle", margin: 0 });
  s.addText(t, { x: x + 0.15, y: by + 0.7, w: bw - 0.3, h: 0.45, fontSize: 17, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addText(d, { x: x + 0.15, y: by + 1.12, w: bw - 0.3, h: 0.7, fontSize: 12.5, color: C.text2, margin: 0, valign: "top", isTextBox: true });
  if (i < steps.length - 1)
    s.addShape(pres.shapes.RIGHT_TRIANGLE, { x: x + bw + 0.06, y: by + 0.83, w: 0.15, h: 0.24, rotate: 90,
      fill: { color: "B7C0CC" }, line: { color: "B7C0CC" } });
});
bullets(s, [
  "Scaler, PCA and SVM are one scikit-learn Pipeline, fit on the training split only, so no information leaks from validation/test data",
  "Hyper-parameters (feature set, C, γ) chosen with 3-fold cross-validation on the training split, then the validation split",
  "The held-out test split is touched exactly once, for the final numbers",
], M, 4.45, W - 2 * M, 2.2, 16);

// 3. Dataset
s = content(`Dataset: ${fmt(stats.n_images)} paintings by 15 artists from Painters by Numbers`, "Introduction");
fitImage(s, FIG("features/dataset_samples.png"), M, TOP, 8.1, BOTTOM - TOP);
stat(s, 9.2, 1.6, 3.5, fmt(stats.n_images), "paintings, resized to 256 x 256 RGB");
stat(s, 9.2, 3.0, 3.5, "15", `artists, ${stats.per_artist_min}-${stats.per_artist_max} works each`);
stat(s, 9.2, 4.4, 3.5, "80/10/10", "shared team split (same as the CNN)");
caption(s, `${fmt(stats.split.train)} train  ·  ${stats.split.val} validation  ·  ${stats.split.test} test`, 9.2, 5.85, 3.6);

// ------------------------------------------------------------------ Feature extraction
pres.addSection({ title: "Feature extraction" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Feature extraction" });
s.addText("Feature extraction", { placeholder: "title" });
s.addText("Six global & local descriptors with OpenCV, Mahotas and scikit-image", { placeholder: "body" });

s = content("Each descriptor looks at a different property of the painting", "Feature extraction");
fitImage(s, FIG("features/feature_overview.png"), M, TOP, W - 2 * M, 3.3);
const rowsF = [
  [{ text: "Descriptor", options: { bold: true } }, { text: "Library", options: { bold: true } },
   { text: "Dim", options: { bold: true, align: "right" } }, { text: "ms / image", options: { bold: true, align: "right" } },
   { text: "Captures", options: { bold: true } }],
];
const caps = { gist: "spatial envelope (scale x orientation energy)", hu: "global shape, rotation/scale invariant",
  color_hist: "joint HSV colour distribution", sift: "bag of 100 local keypoint 'visual words'",
  hog: "gradient orientation / brush strokes", haralick: "GLCM texture statistics", lbp: "micro-texture (extra)" };
const libs = { gist: "NumPy (own port)", hu: "OpenCV", color_hist: "OpenCV", sift: "OpenCV + k-means", hog: "scikit-image",
  haralick: "Mahotas", lbp: "scikit-image" };
featTimes.forEach((f) => rowsF.push([f.label, libs[f.feature], { text: f.dim, options: { align: "right" } },
  { text: Number(f.ms_per_image).toFixed(1), options: { align: "right" } }, caps[f.feature]]));
s.addTable(rowsF, { x: M, y: 4.75, w: W - 2 * M, colW: [1.9, 2.0, 0.9, 1.3, 6.033], fontSize: 11, color: C.text2,
  border: { type: "solid", pt: 0.5, color: "D9DEE5" }, rowH: 0.25, fill: { color: C.background1 } });

s = content("GIST: the \"shape of the scene\" from 32 Gabor filters", "Feature extraction");
fitImage(s, FIG("features/gist_filters.png"), M, TOP, 8.4, BOTTOM - TOP);
bullets(s, [
  "Pre-filter: log intensity, whitening and local contrast normalisation",
  "4 scales x 8 orientations of Gabor filters, applied in the frequency domain",
  "Mean energy on a 4 x 4 grid gives 32 x 16 = 512 numbers",
  "Encodes naturalness, openness, roughness, expansion and ruggedness (Oliva & Torralba, 2001)",
  "Ported from Torralba's MATLAB code to NumPy (lear-gist needs FFTW)",
], 9.3, 1.6, 3.45, 5.0, 14);

s = content("Color histograms separate monochrome engravers from colourists", "Feature extraction");
fitImage(s, FIG("features/color_histograms_by_artist.png"), M, TOP, 8.3, BOTTOM - TOP);
fitImage(s, FIG("features/mean_saturation_by_artist.png"), 9.05, TOP + 0.1, 3.7, 3.4);
bullets(s, [
  "HSV, 8 bins per channel, 512-bin joint histogram (L1-normalised)",
  "No spatial information, only which colours appear",
  "Piranesi & Doré have near-zero saturation, so they are easy to identify",
], 9.05, 4.75, 3.7, 2.0, 13);

s = content("SIFT keypoints and HOG capture local structure and brushwork", "Feature extraction");
fitImage(s, FIG("features/sift_keypoints.png"), M, TOP, 6.0, BOTTOM - TOP - 0.4);
fitImage(s, FIG("features/hog_examples.png"), 6.75, TOP, 6.0, BOTTOM - TOP - 0.4);
caption(s, "SIFT: up to 500 keypoints, 128-d descriptors quantised into a 100-word vocabulary (k-means on training images)", M, BOTTOM - 0.35, 6.0);
caption(s, "HOG: 128 x 128 grayscale, 9 orientations, 16 px cells, 2 x 2 blocks, 1,764-d", 6.75, BOTTOM - 0.35, 6.0);

s = content("Hu moments (shape) and Haralick textures", "Feature extraction");
fitImage(s, FIG("features/hu_haralick_by_artist.png"), M, TOP, W - 2 * M, 3.9);
bullets(s, [
  "Hu: 7 moment invariants (translation, scale and rotation invariant), log-scaled because raw values span ~20 orders of magnitude",
  "Haralick: 13 grey-level co-occurrence statistics (contrast, correlation, entropy, ...), averaged over 4 directions",
  "Aivazovsky's smooth seascapes have very low contrast; the engravers have the highest",
], M, 5.35, W - 2 * M, 1.5, 14);

s = content("Min-Max scaling and PCA compress every feature set to 100 dimensions", "Feature extraction");
fitImage(s, FIG("features/pca_explained_variance.png"), M, TOP, 6.1, 4.0);
fitImage(s, FIG("features/tsne_features.png"), 6.9, TOP, 5.85, BOTTOM - TOP);
bullets(s, [
  "Min-Max: each feature mapped to [0, 1], so a 1,764-d HOG block cannot swamp 7 Hu values",
  "PCA(100) fit on the training split; validation and test data are projected onto the same components",
  "The t-SNE map shows Piranesi and Aivazovsky in tight clusters, while Renoir is spread out",
], M, 5.45, 6.1, 1.5, 13);

// ------------------------------------------------------------------ Models
pres.addSection({ title: "Models" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Models" });
s.addText("Models & tuning", { placeholder: "title" });
s.addText("Logistic Regression baseline and a multi-class RBF Support Vector Machine", { placeholder: "body" });

s = content("Two classifiers on the same 100-d PCA features", "Models");
card(s, M, TOP + 0.1, 5.9, 5.1);
s.addText("Logistic Regression (baseline)", { x: M + 0.3, y: TOP + 0.3, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
bullets(s, [
  "Multinomial softmax, L2 penalty, solved with L-BFGS",
  "One linear decision boundary per artist in PCA space",
  "C tuned over {0.01 ... 1000} with 3-fold CV",
  "Fast to train and a reference point for the SVM",
], M + 0.3, TOP + 0.95, 5.3, 4.0, 15);
card(s, 6.83, TOP + 0.1, 5.9, 5.1, "FBEBDD");
s.addText("RBF Support Vector Machine", { x: 7.13, y: TOP + 0.3, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true });
s.addText([
  { text: "For each artist k (one-vs-rest, 15 SVMs):", options: { breakLine: true } },
  { text: "min  ½‖w‖² + C Σ ξᵢ", options: { fontFace: "Cambria Math", fontSize: 18, bold: true, color: C.text1, breakLine: true } },
  { text: "s.t.  yᵢ(wᵀφ(xᵢ) + b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0", options: { fontFace: "Cambria Math", fontSize: 16, color: C.text1, breakLine: true } },
  { text: "K(x, x′) = exp(−γ‖x − x′‖²)", options: { fontFace: "Cambria Math", fontSize: 16, color: C.text1, breakLine: true } },
  { text: "Predict the artist whose SVM gives the largest decision value.", options: {} },
], { x: 7.13, y: TOP + 0.95, w: 5.3, h: 2.6, fontSize: 15, color: C.text2, paraSpaceAfter: 8, valign: "top", margin: 0, isTextBox: true });
bullets(s, [
  "C: penalty on margin violations (large C = narrow margin, fits training data harder)",
  "γ: RBF kernel width (large γ = very local, wiggly boundaries)",
], 7.13, TOP + 3.0, 5.3, 1.8, 15);

s = content("3-fold cross-validated grid search over C and γ", "Models");
fitImage(s, FIG("models/svm_grid_search_heatmap.png"), M, TOP, 9.0, BOTTOM - TOP);
stat(s, 9.95, 1.6, 2.9, `C = ${Number(svm.C)}`, "best penalty");
stat(s, 9.95, 3.0, 2.9, `γ = ${Number(svm.gamma)}`, "best RBF width");
stat(s, 9.95, 4.4, 2.9, pct(svm.cv_acc), "3-fold CV accuracy", C.accent2);
caption(s, "C, γ ∈ {10⁻⁵, ..., 10⁵}: 121 settings x 3 folds per feature set", 9.95, 5.9, 2.9);

s = content("Which feature combination works best?", "Models");
fitImage(s, FIG("models/top_feature_combinations.png"), M, TOP, 6.4, BOTTOM - TOP);
fitImage(s, FIG("models/feature_contribution.png"), 7.1, TOP, 5.65, 3.2);
bullets(s, [
  "All 63 combinations screened with a coarse 3-fold grid (C ∈ {1, 10, 100}, γ ∈ {0.01 ... 10})",
  `Best SVM: ${svm.features}`,
  `The top ${final.svm_candidates.length} combinations got the full 11 x 11 grid with one-vs-rest SVMs`,
], 7.1, 4.7, 5.65, 2.0, 13);

// ------------------------------------------------------------------ Evaluation
pres.addSection({ title: "Evaluation" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Evaluation" });
s.addText("Evaluation", { placeholder: "title" });
s.addText("Accuracy, per-artist errors, overfitting and inference cost", { placeholder: "body" });

s = content(`RBF SVM reaches ${pct(svm.test_acc)} test accuracy vs ${pct(lr.test_acc)} for the baseline`, "Evaluation");
const head = ["Model", "Train acc", "Val acc", "Test acc", "Train time (s)", "Inference (ms/img)", "Features"];
const rowsT = [head.map((h) => ({ text: h, options: { bold: true, color: C.background1, fill: { color: C.text1 } } }))];
table.forEach((r) => rowsT.push([r["Model"], r["Training Accuracy (%)"] + "%", r["Validation Accuracy (%)"] + "%",
  { text: r["Test Accuracy (%)"] + "%", options: { bold: true } }, r["Train Time (s)"], r["Inference Time (ms/img)"], r["Features Used"]]));
s.addTable(rowsT, { x: M, y: TOP + 0.1, w: W - 2 * M, colW: [0.9, 1.15, 1.1, 1.1, 1.45, 1.75, 4.683], fontSize: 13,
  color: C.text2, border: { type: "solid", pt: 0.5, color: "D9DEE5" }, rowH: 0.42 });
caption(s, "* LR = Logistic-Regression baseline with its best feature combination. Inference time = feature extraction + model, per image on one CPU core.",
  M, TOP + 0.25 + 0.42 * rowsT.length, W - 2 * M);
stat(s, M, 4.9, 3.6, pct(svm.test_acc), "SVM test accuracy (paper, other artists: 68.1%)");
stat(s, 4.7, 4.9, 3.6, pct(lr.test_acc), "LR baseline test accuracy (paper, other artists: 61.2%)", C.accent4);
stat(s, 8.8, 4.9, 3.9, pct(svm.train_acc), "SVM training accuracy: heavy overfitting", C.accent5);

s = content(`${cp[0][0]}/${cp[0][1]} and ${cp[1][0]}/${cp[1][1]} are the most confused pairs`, "Evaluation");
fitImage(s, FIG("models/confusion_matrix_svm.png"), M, TOP, 6.6, BOTTOM - TOP);
fitImage(s, FIG("models/precision_recall_svm.png"), 7.35, TOP, 5.4, BOTTOM - TOP);

s = content(`The SVM still overfits: ${pct(svm.train_acc)} train vs ${pct(svm.test_acc)} test accuracy`, "Evaluation");
fitImage(s, FIG("models/overfitting_train_vs_val.png"), M, TOP, 5.9, BOTTOM - TOP);
fitImage(s, FIG("models/accuracy_comparison.png"), 6.7, TOP, 6.05, 3.6);
bullets(s, [
  "Larger C or γ push training accuracy to 100% with no gain in CV accuracy",
  "Lower C or γ narrow the gap but cost validation accuracy (see the grid heatmap)",
  "Logistic Regression generalises more evenly but underfits",
], 6.7, 5.1, 6.05, 1.7, 13);

s = content("Most of the inference cost is feature extraction", "Evaluation");
fitImage(s, FIG("models/inference_time_vs_error.png"), M, TOP, 7.3, BOTTOM - TOP);
fitImage(s, FIG("features/feature_dims_and_times.png"), 8.1, TOP + 0.2, 4.65, 2.2);
bullets(s, [
  `Final SVM: ${Number(svm.feature_ms_per_image).toFixed(0)} ms of feature extraction vs ${Number(svm.model_ms_per_image).toFixed(1)} ms of SVM prediction per image`,
  "GIST dominates the cost; Hu and colour histograms are almost free",
  "Feature extraction is embarrassingly parallel and could run on a GPU",
], 8.1, 3.75, 4.65, 3.0, 13);

s = content("Examples of SVM mistakes", "Evaluation");
fitImage(s, FIG("models/misclassified_svm.png"), M, TOP, W - 2 * M, BOTTOM - TOP);

// ------------------------------------------------------------------ Conclusion
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusion" });
s.addText("Takeaways", { placeholder: "title" });
s.addText([
  { text: `Best features: ${svm.features}. ${helps.join(" and ")} help most${hurts.length ? "; " + hurts.join(" and ") + " hurt" + (hurts.length === 1 ? "s" : "") : ""}`, options: { bullet: true, breakLine: true } },
  { text: `Tuned RBF SVM: ${pct(svm.test_acc)} test accuracy on 15 artists, vs ${pct(lr.test_acc)} for Logistic Regression`, options: { bullet: true, breakLine: true } },
  { text: "The SVM overfits heavily even after tuning C and γ: fewer PCA components or more data would help", options: { bullet: true, breakLine: true } },
  { text: "Classical features are cheap to train on a CPU but slow at inference; this is the baseline for the CNN comparison", options: { bullet: true } },
], { placeholder: "body", paraSpaceAfter: 10 });

pres.writeFile({ fileName: OUT }).then(async (f) => {
  try {
    const { applyTheme } = require("./apply_theme.js");
    await applyTheme(f, THEME);
  } catch (e) {
    console.warn("theme colours not applied:", e.message);
  }
  console.log("wrote", f);
});
