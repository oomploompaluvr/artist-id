"""Step 5 - full C x gamma grid search on the best feature sets, then final test evaluation.

* takes the TOP_K SVM combinations from step 4 (by validation accuracy),
* runs the paper's 3-fold CV grid search with C, gamma in {1e-5, ..., 1e5} (11 x 11),
* picks the final SVM by validation accuracy, and the final Logistic-Regression
  baseline (best LR combination from step 4),
* refits on the training split, evaluates once on the held-out test split and
  stores metrics, predictions, confusion matrices and the fitted models.

Outputs (results/):
    grid_search/<combo>.csv     full CV table for each combination
    final_results.csv           poster-style table (train/test acc, times, features)
    final_models.json           metrics + chosen hyper-parameters
    predictions_<model>.csv     test-set predictions
    models/<model>.joblib       fitted pipelines
"""
import _path  # noqa: F401

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from artist_id import config as C
from artist_id.data import load_splits
from artist_id.models import (LR_PARAM, SVM_PARAM, combo_name, feature_ms, get_splits, make_logreg, make_svm,
                              timed_fit, timed_predict)

GRID_DIR = C.RESULTS_DIR / "grid_search"
GRID_DIR.mkdir(exist_ok=True)


def full_grid_search(combo):
    out = GRID_DIR / f"{'+'.join(combo)}.csv"
    d = get_splits(combo)
    X_tr, y_tr = d["train"]
    if out.exists():
        cvres = pd.read_csv(out)
    else:
        grid = {SVM_PARAM["C"]: C.FULL_C_GRID, SVM_PARAM["gamma"]: C.FULL_GAMMA_GRID}
        cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED)
        gs = GridSearchCV(make_svm(X_tr.shape[1]), grid, cv=cv, n_jobs=C.N_JOBS, scoring="accuracy",
                          refit=False, return_train_score=True)
        t0 = time.perf_counter()
        gs.fit(X_tr, y_tr)
        print(f"  grid search took {(time.perf_counter() - t0) / 60:.1f} min")
        cvres = pd.DataFrame(gs.cv_results_)
        cvres = cvres.rename(columns={f"param_{SVM_PARAM['C']}": "C", f"param_{SVM_PARAM['gamma']}": "gamma"})
        cvres = cvres[["C", "gamma", "mean_test_score", "std_test_score", "mean_train_score", "mean_fit_time"]]
        cvres.to_csv(out, index=False)
    best = cvres.loc[cvres.mean_test_score.idxmax()]
    return float(best.C), float(best.gamma), float(best.mean_test_score)


def evaluate(name, model, combo, extra):
    d = get_splits(combo)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = d["train"], d["val"], d["test"]
    train_s = timed_fit(model, X_tr, y_tr)
    pred_te, model_ms = timed_predict(model, X_te)
    res = {
        "name": name,
        "combo": "+".join(combo),
        "features": combo_name(combo),
        "train_acc": float((model.predict(X_tr) == y_tr).mean()),
        "val_acc": float((model.predict(X_va) == y_va).mean()),
        "test_acc": float((pred_te == y_te).mean()),
        "train_time_s": train_s,
        "model_ms_per_image": model_ms,
        "feature_ms_per_image": feature_ms(combo),
        **extra,
    }
    res["total_inference_ms"] = res["model_ms_per_image"] + res["feature_ms_per_image"]
    return res, model, pred_te, y_te


if __name__ == "__main__":
    search = pd.read_csv(C.RESULTS_DIR / "combination_search.csv")
    svm_rank = search[search.model == "svm"].sort_values("val_acc", ascending=False)
    lr_rank = search[search.model == "lr"].sort_values("val_acc", ascending=False)

    # ---------------------------------------------------------------- SVM: full grid on top-K combos
    candidates = []
    for key in svm_rank.combo.head(C.TOP_K_COMBOS_FOR_FULL_GRID):
        combo = tuple(key.split("+"))
        print(f"Full 11x11 grid search: {combo_name(combo)}")
        c, g, cv = full_grid_search(combo)
        model = make_svm(get_splits(combo)["train"][0].shape[1], c, g)
        res, model, pred, y_te = evaluate("SVM", model, combo, {"C": c, "gamma": g, "cv_acc": cv})
        print(f"  best C={c:g} gamma={g:g} cv={cv:.3f} | train={res['train_acc']:.3f} "
              f"val={res['val_acc']:.3f} test={res['test_acc']:.3f}")
        candidates.append((res, model, pred, y_te))

    # ---------------------------------------------------------------- LR baseline
    lr_row = lr_rank.iloc[0]
    lr_combo = tuple(lr_row.combo.split("+"))
    lr_res, lr_model, lr_pred, y_te = evaluate(
        "LR", make_logreg(get_splits(lr_combo)["train"][0].shape[1], float(lr_row.best_C)), lr_combo,
        {"C": float(lr_row.best_C), "gamma": None, "cv_acc": float(lr_row.cv_acc)})
    print(f"LR baseline ({lr_res['features']}): train={lr_res['train_acc']:.3f} test={lr_res['test_acc']:.3f}")

    # ---------------------------------------------------------------- pick final SVM by validation accuracy
    candidates.sort(key=lambda t: t[0]["val_acc"], reverse=True)
    best_res, best_model, best_pred, _ = candidates[0]

    df = load_splits()
    test_files = df[df.split == "test"].filename.to_numpy()
    summary = {"svm": best_res, "lr": lr_res, "svm_candidates": [c[0] for c in candidates]}
    for tag, res, model, pred in (("svm", best_res, best_model, best_pred), ("lr", lr_res, lr_model, lr_pred)):
        joblib.dump(model, C.MODEL_DIR / f"{tag}.joblib")
        pd.DataFrame({"filename": test_files, "true": y_te, "pred": pred,
                      "true_artist": np.array(C.ARTISTS)[y_te], "pred_artist": np.array(C.ARTISTS)[pred]}
                     ).to_csv(C.RESULTS_DIR / f"predictions_{tag}.csv", index=False)
        np.save(C.RESULTS_DIR / f"confusion_{tag}.npy", confusion_matrix(y_te, pred, labels=range(len(C.ARTISTS))))
        rep = classification_report(y_te, pred, target_names=C.ARTISTS, output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.to_csv(C.RESULTS_DIR / f"classification_report_{tag}.csv")
        summary[tag]["macro_f1"] = rep["macro avg"]["f1-score"]

    with open(C.RESULTS_DIR / "final_models.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # Poster-style table: LR baseline + every SVM candidate (sorted by test accuracy).
    rows = [lr_res] + [c[0] for c in candidates]
    table = pd.DataFrame([{
        "Model": ("LR*" if r["name"] == "LR" else "SVM"),
        "Training Accuracy (%)": round(100 * r["train_acc"], 1),
        "Validation Accuracy (%)": round(100 * r["val_acc"], 1),
        "Test Accuracy (%)": round(100 * r["test_acc"], 1),
        "Train Time (s)": round(r["train_time_s"], 2),
        "Inference Time (ms/img)": round(r["total_inference_ms"], 1),
        "Features Used": r["features"],
        "C": r["C"], "gamma": r["gamma"],
    } for r in rows])
    table.to_csv(C.RESULTS_DIR / "final_results.csv", index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nFinal SVM: {best_res['features']}  test accuracy = {100 * best_res['test_acc']:.1f}%")
