"""Summaries and figures from the files written by src.train.

    python -m src.evaluate            # validation summary + figures available so far
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import data
from src import plotstyle as ps

RESULTS = data.ROOT / "results"
FIG = RESULTS / "figures"

MODEL_ORDER = ["dummy", "tree", "forest", "hgb", "hgb_unweighted"]
LABELS = {"dummy": "Trivial baseline (prior)", "tree": "Decision tree (depth by CV)",
          "forest": "Random forest", "hgb": "HGB, class-weighted",
          "hgb_unweighted": "HGB, unweighted (ablation)"}


def summary(path, metrics=("pr_auc", "roc_auc", "precision", "recall", "f1", "train_time_s")):
    """mean ± std over seeds for each (model, feature_set)."""
    df = pd.read_csv(path)
    df = df[df["metric"].isin(metrics)]
    g = df.groupby(["feature_set", "model", "metric"])["value"].agg(["mean", "std"]).reset_index()
    g["cell"] = g.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    t = g.pivot(index=["feature_set", "model"], columns="metric", values="cell")[list(metrics)]
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return t.sort_index(key=lambda idx: idx.map(order) if idx.name == "model" else idx)


def plot_tree_cv():
    """Depth curve: best CV PR-AUC at each max_depth (over leaf sizes), seed 42."""
    cv = pd.read_csv(RESULTS / "dt_cv.csv")
    cv = cv[cv["seed"] == 42].copy()
    cv["max_depth"] = cv["max_depth"].fillna(-1).astype(int)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, fs in enumerate(["DEPLOYABLE", "FULL"]):
        best = (cv[cv["feature_set"] == fs]
                .sort_values("cv_pr_auc_mean").groupby("max_depth").tail(1).set_index("max_depth"))
        depths = [d for d in sorted(best.index) if d != -1]
        x = depths + [17]  # plot "None" (unlimited) at the right edge
        yv = best.loc[depths + [-1], "cv_pr_auc_mean"].values
        sd = best.loc[depths + [-1], "cv_pr_auc_std"].values
        c = ps.SERIES[i]
        ax.plot(x, yv, marker="o", markersize=4, color=c, label=fs)
        ax.fill_between(x, yv - sd, yv + sd, color=c, alpha=0.15, linewidth=0)
        ax.annotate(f"leaf={best.loc[-1, 'min_samples_leaf']}", (17, yv[-1]),
                    textcoords="offset points", xytext=(6, -3), fontsize=8, color=ps.INK_2)
    ax.set_xticks(list(range(2, 16)) + [17])
    ax.set_xticklabels([str(d) for d in range(2, 16)] + ["None"])
    ax.set_xlabel("max_depth (each point = best min_samples_leaf for that depth)")
    ax.set_ylabel("5-fold CV PR-AUC")
    ax.legend(loc="lower right")
    ax.set_title("Decision tree: deeper is better once leaves are kept large")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "DT_cv_depth_curve.png")
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # so "±" prints on Windows consoles
    ps.apply()
    pd.set_option("display.width", 160)
    print("VALIDATION (mean ± std over seeds 42/43/44)\n")
    print(summary(RESULTS / "val_metrics.csv").to_string())
    if (RESULTS / "dt_cv.csv").exists():
        plot_tree_cv()
        print("\nsaved results/figures/DT_cv_depth_curve.png")


if __name__ == "__main__":
    main()
