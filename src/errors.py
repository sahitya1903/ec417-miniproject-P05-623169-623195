"""Error analysis of the chosen model on the test set (called from src.evaluate).

Outputs
    results/errors_top_fp.csv    20 most confident false positives (seed 42)
    results/errors_top_fn.csv    20 most confident false negatives (seed 42)
    results/errors_profile.csv   what TP / FP / FN / TN clients look like (seed 42)
    results/slices_test.csv      PR-AUC, precision, recall per slice (mean over seeds)

`duration` is shown in these tables for context only; the deployable model never sees it.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import data

RESULTS = data.ROOT / "results"
CHECKPOINTS = data.ROOT / "checkpoints"

AGE_BINS = [0, 29, 39, 49, 59, 200]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60+"]
SLICE_COLS = ["job", "age_band", "contact", "month", "poutcome"]
SHOW_COLS = ["age", "job", "marital", "education", "balance", "housing", "loan", "contact",
             "day", "month", "campaign", "pdays", "previous", "poutcome", "duration"]
MIN_POSITIVES = 30   # slices with fewer test positives are too noisy to judge


def test_scores(model, fs="DEPLOYABLE"):
    """Raw test rows (all columns) with each seed's score and decision."""
    X, y = data.load()
    Xf = data.feature_set(X, fs)
    out = None
    for seed in data.SEEDS:
        _, _, X_test, _, _, y_test = data.split(Xf, y, seed)
        # Safe to unpickle: written locally by src.train.
        ck = joblib.load(CHECKPOINTS / f"{model}_{fs}_{seed}.joblib")
        if out is None:   # the test rows are the same for every seed
            out = X.loc[X_test.index].copy()
            out["y"] = y_test
            out["age_band"] = pd.cut(out["age"], AGE_BINS, labels=AGE_LABELS)
        p = ck["model"].predict_proba(X_test)[:, 1]
        out[f"p_{seed}"] = p
        out[f"pred_{seed}"] = (p >= ck["threshold"]).astype(int)
    return out


def outcome(y, pred):
    return np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
                     ["TP", "FP", "FN"], "TN")


def profile(df):
    """Average client in each cell of the confusion matrix."""
    g = df.groupby("outcome", observed=True)
    t = pd.DataFrame({
        "n": g.size(),
        "mean_score": g["p_42"].mean(),
        "age_mean": g["age"].mean(),
        "balance_median": g["balance"].median(),
        "campaign_mean": g["campaign"].mean(),
        "pct_previously_contacted": g["previously_contacted"].mean() * 100,
        "pct_poutcome_success": g["poutcome"].apply(lambda s: (s == "success").mean() * 100),
        "pct_contact_unknown": g["contact"].apply(lambda s: (s == "unknown").mean() * 100),
        "pct_month_may": g["month"].apply(lambda s: (s == "may").mean() * 100),
        "pct_housing_loan": g["housing"].apply(lambda s: (s == "yes").mean() * 100),
        "duration_median_not_used": g["duration"].median(),
    }).reindex(["TP", "FP", "FN", "TN"])
    return t.round(2)


def slices(df):
    """Per-slice metrics, averaged over the three seeds' models."""
    rows = []
    for col in SLICE_COLS:
        for val, s in df.groupby(col, observed=True):
            pos = int(s["y"].sum())
            r = {"column": col, "value": str(val), "n": len(s), "positives": pos,
                 "positive_rate": s["y"].mean()}
            per_seed = []
            for seed in data.SEEDS:
                p, pred = s[f"p_{seed}"], s[f"pred_{seed}"]
                per_seed.append({
                    "pr_auc": average_precision_score(s["y"], p) if 0 < pos < len(s) else np.nan,
                    "recall": pred[s["y"] == 1].mean() if pos else np.nan,
                    "precision": s["y"][pred == 1].mean() if pred.sum() else np.nan,
                    "share_called": pred.mean()})
            r |= pd.DataFrame(per_seed).mean().to_dict()
            r["pr_auc_over_base_rate"] = r["pr_auc"] / r["positive_rate"] if pos else np.nan
            r["reliable"] = pos >= MIN_POSITIVES
            rows.append(r)
    return pd.DataFrame(rows)


def run(model, fs="DEPLOYABLE"):
    df = test_scores(model, fs)
    df["outcome"] = outcome(df["y"], df["pred_42"])

    cols = SHOW_COLS + ["p_42"]
    fp = df[df["outcome"] == "FP"].sort_values("p_42", ascending=False).head(20)[cols]
    fn = df[df["outcome"] == "FN"].sort_values("p_42").head(20)[cols]
    fp.to_csv(RESULTS / "errors_top_fp.csv", index_label="row")
    fn.to_csv(RESULTS / "errors_top_fn.csv", index_label="row")
    prof = profile(df)
    prof.to_csv(RESULTS / "errors_profile.csv")
    sl = slices(df)
    sl.to_csv(RESULTS / "slices_test.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nERROR ANALYSIS ({model}, test set)\n\nprofile by outcome (seed 42):\n")
    print(prof.to_string())
    for name, t in [("false positives", fp), ("false negatives", fn)]:
        print(f"\ntop-20 {name}: poutcome {t['poutcome'].value_counts().to_dict()}, "
              f"contact {t['contact'].value_counts().to_dict()}, "
              f"month {t['month'].value_counts().head(4).to_dict()}")
    print("\nslices (mean over seeds; only slices with >= 30 test positives):\n")
    print(sl[sl["reliable"]].round(3).to_string(index=False))
    print("\nsaved results/errors_top_fp.csv, errors_top_fn.csv, errors_profile.csv, slices_test.csv")
    return df
