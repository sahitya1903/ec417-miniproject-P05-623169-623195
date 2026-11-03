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


def duration_gap(path):
    """PR-AUC and ROC-AUC with (FULL) vs without (DEPLOYABLE) duration, mean over seeds."""
    df = pd.read_csv(path)
    df = df[df["metric"].isin(["pr_auc", "roc_auc"])]
    g = df.groupby(["model", "metric", "feature_set"])["value"].mean().unstack("feature_set")
    g["gap"] = g["FULL"] - g["DEPLOYABLE"]
    g["share_lost"] = g["gap"] / g["FULL"]
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return g.sort_index(key=lambda idx: idx.map(order) if idx.name == "model" else idx).round(3)


def plot_duration_gap(path, name="F2b_duration_gap.png", split="validation"):
    """F2b: PR-AUC of each model with and without duration (mean ± std over seeds)."""
    df = pd.read_csv(path)
    df = df[df["metric"] == "pr_auc"]
    g = df.groupby(["model", "feature_set"])["value"].agg(["mean", "std"])
    names = [m for m in MODEL_ORDER if m in g.index.get_level_values("model")]
    prevalence = g.loc[("dummy", "DEPLOYABLE"), "mean"] if "dummy" in names else None

    fig, ax = plt.subplots(figsize=(7, 3.8))
    width = 0.36
    for i, (fs, label) in enumerate([("DEPLOYABLE", "Without duration (deployable)"),
                                     ("FULL", "With duration (leaky)")]):
        xs = [k + (i - 0.5) * (width + 0.02) for k in range(len(names))]
        means = [g.loc[(m, fs), "mean"] for m in names]
        stds = [g.loc[(m, fs), "std"] for m in names]
        ax.bar(xs, means, width, yerr=stds, color=ps.SERIES[i], label=label,
               error_kw={"ecolor": ps.INK_2, "elinewidth": 1, "capsize": 3})
        for x, v, s in zip(xs, means, stds):
            ax.annotate(f"{v:.2f}", (x, v + s), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, color=ps.INK_2)
    if prevalence is not None:
        ax.axhline(prevalence, color=ps.NO_COLOR, linestyle="--", linewidth=1)
        ax.annotate(f"no skill = positive rate ({prevalence:.3f})", (len(names) - 0.5, prevalence),
                    textcoords="offset points", xytext=(0, 4), ha="right", fontsize=8, color=ps.INK_2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([LABELS[m] for m in names], fontsize=9)
    ax.set_ylabel(f"{split} PR-AUC (mean ± std, 3 seeds)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("The duration leak: PR-AUC with and without the call length")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # so "±" prints on Windows consoles
    ps.apply()
    pd.set_option("display.width", 160)
    print("VALIDATION (mean ± std over seeds 42/43/44)\n")
    print(summary(RESULTS / "val_metrics.csv").to_string())
    print("\nDURATION GAP on validation (FULL - DEPLOYABLE, mean over seeds)\n")
    print(duration_gap(RESULTS / "val_metrics.csv").to_string())
    plot_duration_gap(RESULTS / "val_metrics.csv")
    print("\nsaved results/figures/F2b_duration_gap.png")
    if (RESULTS / "dt_cv.csv").exists():
        plot_tree_cv()
        print("saved results/figures/DT_cv_depth_curve.png")


if __name__ == "__main__":
    main()
