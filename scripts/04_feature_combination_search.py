"""Step 4 - screen all 63 combinations of the six descriptors.

For every combination (features concatenated horizontally):
  * RBF SVM: 3-fold CV grid search over a coarse C x gamma grid on the training split
    (libsvm's fast one-vs-one multiclass is used for this screening stage only; the
    full grid search and final models in step 5 use the paper's one-vs-rest SVMs),
  * Logistic Regression: 3-fold CV grid search over C,
then the best model is refit on the whole training split and scored on the
validation split. Results go to results/combination_search.csv (the script
resumes if interrupted).
"""
import _path  # noqa: F401

import argparse
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from artist_id import config as C
from artist_id.models import (LR_PARAM, SVM_OVO_PARAM, all_combinations, combo_name, feature_ms, get_splits,
                              make_logreg, make_svm, timed_predict)

OUT = C.RESULTS_DIR / "combination_search.csv"
LR_C_GRID = [0.01, 0.1, 1, 10, 100, 1000]


def run(model_name, model, grid, d):
    X_tr, y_tr = d["train"]
    X_va, y_va = d["val"]
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    gs = GridSearchCV(model, grid, cv=cv, n_jobs=C.N_JOBS, scoring="accuracy", refit=True)
    t0 = time.perf_counter()
    gs.fit(X_tr, y_tr)
    search_s = time.perf_counter() - t0
    best = gs.best_estimator_
    pred_va, ms = timed_predict(best, X_va)
    return {
        "model": model_name,
        "cv_acc": gs.best_score_,
        "train_acc": (best.predict(X_tr) == y_tr).mean(),
        "val_acc": (pred_va == y_va).mean(),
        "best_C": gs.best_params_.get(SVM_OVO_PARAM["C"], gs.best_params_.get(LR_PARAM["C"])),
        "best_gamma": gs.best_params_.get(SVM_OVO_PARAM["gamma"], ""),
        "train_time_s": gs.refit_time_,
        "search_time_s": search_s,
        "model_ms_per_image": ms,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="svm,lr")
    args = ap.parse_args()
    models = args.models.split(",")

    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["combo", "model"])
    have = set(zip(done.combo, done.model))
    rows = done.to_dict("records")

    combos = all_combinations()
    for i, combo in enumerate(combos, 1):
        key = "+".join(combo)
        d = get_splits(combo)
        dim = d["train"][0].shape[1]
        for m in models:
            if (key, m) in have:
                continue
            if m == "svm":
                grid = {SVM_OVO_PARAM["C"]: C.COARSE_C_GRID, SVM_OVO_PARAM["gamma"]: C.COARSE_GAMMA_GRID}
                r = run("svm", make_svm(dim, multiclass="ovo"), grid, d)
            else:
                r = run("lr", make_logreg(dim), {LR_PARAM["C"]: LR_C_GRID}, d)
            r.update(combo=key, combo_label=combo_name(combo), n_features=len(combo), dim=dim,
                     feature_ms_per_image=feature_ms(combo))
            r["total_inference_ms"] = r["feature_ms_per_image"] + r["model_ms_per_image"]
            rows.append(r)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"[{i:2d}/{len(combos)}] {m:3s} {combo_name(combo):<55} cv={r['cv_acc']:.3f} "
                  f"val={r['val_acc']:.3f} train={r['train_acc']:.3f} ({r['search_time_s']:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    for m in models:
        print(f"\nTop 10 {m.upper()} combinations by validation accuracy:")
        print(res[res.model == m].sort_values("val_acc", ascending=False)
              .head(10)[["combo_label", "cv_acc", "train_acc", "val_acc", "best_C", "best_gamma"]].to_string(index=False))
