"""Train every model on both feature sets and three seeds.

    python -m src.train                    # all models, validation scores only
    python -m src.train --models tree      # just some models
    python -m src.train --final            # ALSO score the locked test set (done once, at the end)

Outputs
    results/val_metrics.csv   one row per (model, feature_set, seed, split, metric)
    results/metrics.csv       the same for the test set (only with --final)
    results/dt_cv.csv         decision-tree cross-validation table (depth curve)
    checkpoints/*.joblib      fitted models, git-ignored, used by src.evaluate
"""
import argparse
import time

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import data, models
from src.metrics import best_f1_threshold, score

RESULTS = data.ROOT / "results"
CHECKPOINTS = data.ROOT / "checkpoints"


# --- one function per model: fit on train (tune with CV or validation) -------
def fit_dummy(X_tr, y_tr, X_val, y_val, seed, fs):
    m = models.dummy()
    t = time.perf_counter()
    m.fit(X_tr, y_tr)
    return m, time.perf_counter() - t, {}


def fit_tree(X_tr, y_tr, X_val, y_val, seed, fs):
    """Depth and leaf size chosen by 5-fold stratified CV on the TRAINING part only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(models.tree(seed), models.TREE_GRID, scoring="average_precision",
                          cv=cv, n_jobs=-1, refit=True)
    t = time.perf_counter()
    search.fit(X_tr, y_tr)
    tune_time = time.perf_counter() - t

    cv_table = pd.DataFrame(search.cv_results_)[
        ["param_clf__max_depth", "param_clf__min_samples_leaf", "mean_test_score", "std_test_score"]]
    cv_table.columns = ["max_depth", "min_samples_leaf", "cv_pr_auc_mean", "cv_pr_auc_std"]
    cv_table.insert(0, "seed", seed)
    cv_table.insert(0, "feature_set", fs)

    best = search.best_params_
    info = {"best_max_depth": best["clf__max_depth"] if best["clf__max_depth"] is not None else -1,
            "best_min_samples_leaf": best["clf__min_samples_leaf"],
            "cv_pr_auc": search.best_score_, "tune_time_s": tune_time, "_cv_table": cv_table}
    return search.best_estimator_, search.refit_time_, info


MODELS = {
    "dummy": fit_dummy,
    "tree": fit_tree,
}


# --- bookkeeping --------------------------------------------------------------
def long_rows(model, fs, seed, split, values):
    return [{"model": model, "feature_set": fs, "seed": seed, "split": split,
             "metric": k, "value": float(v)} for k, v in values.items()]


def merge_into(path, new_rows, models_run):
    """Replace the rows of the models just run; keep rows from other models."""
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["model"].isin(models_run)], new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def main(model_names, final):
    RESULTS.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    X, y = data.load()
    val_rows, test_rows, cv_tables = [], [], []

    for name in model_names:
        for fs in data.FEATURE_SETS:
            Xf = data.feature_set(X, fs)
            for seed in data.SEEDS:
                X_tr, X_val, X_test, y_tr, y_val, y_test = data.split(Xf, y, seed)
                model, train_time, info = MODELS[name](X_tr, y_tr, X_val, y_val, seed, fs)
                if "_cv_table" in info:
                    cv_tables.append(info.pop("_cv_table"))

                # Threshold is chosen on validation and then frozen.
                p_val = model.predict_proba(X_val)[:, 1]
                thr = best_f1_threshold(y_val, p_val)
                val_scores = score(y_val, p_val, thr) | {"train_time_s": train_time} | info
                val_rows += long_rows(name, fs, seed, "val", val_scores)

                if final:
                    p_test = model.predict_proba(X_test)[:, 1]
                    test_rows += long_rows(name, fs, seed, "test",
                                           score(y_test, p_test, thr) | {"train_time_s": train_time})

                joblib.dump({"model": model, "threshold": thr},
                            CHECKPOINTS / f"{name}_{fs}_{seed}.joblib")
                print(f"{name:6s} {fs:10s} seed={seed}  val PR-AUC={val_scores['pr_auc']:.4f}  "
                      f"ROC-AUC={val_scores['roc_auc']:.4f}  fit={train_time:.2f}s"
                      + (f"  depth={info['best_max_depth']} leaf={info['best_min_samples_leaf']}"
                         if "best_max_depth" in info else ""))

    merge_into(RESULTS / "val_metrics.csv", val_rows, model_names)
    if cv_tables:
        pd.concat(cv_tables).to_csv(RESULTS / "dt_cv.csv", index=False)
    if final:
        merge_into(RESULTS / "metrics.csv", test_rows, model_names)
        print("test-set metrics written to results/metrics.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--final", action="store_true", help="also score the locked test set")
    args = ap.parse_args()
    main(args.models, args.final)
