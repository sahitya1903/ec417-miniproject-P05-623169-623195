"""Summaries and figures from the files written by src.train.

    python -m src.evaluate            # validation summary, and the test-set report once
                                      # results/metrics.csv exists (after src.train --final)

Test-set outputs (Phase 5)
    results/figures/F2b_duration_gap_test.png, F3 ... F9
    results/gains_test.csv, profit_test.csv, permutation_importance.csv
    results/errors_*.csv, slices_test.csv     (see src/errors.py)
"""
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from src import data, errors
from src import plotstyle as ps

RESULTS = data.ROOT / "results"
FIG = RESULTS / "figures"
CHECKPOINTS = data.ROOT / "checkpoints"

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
    # The class-weight ablation is about weighting, not the leak, so it is left out here.
    names = [m for m in MODEL_ORDER
             if m != "hgb_unweighted" and m in g.index.get_level_values("model")]
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
    ax.set_xticklabels([LABELS[m].replace(" (", "\n(").replace(", ", ",\n") for m in names], fontsize=9)
    ax.set_ylabel(f"{split} PR-AUC (mean ± std, 3 seeds)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("The duration leak: PR-AUC with and without the call length")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


# --- Phase 5: test-set report -----------------------------------------------
CURVE_MODELS = ["tree", "forest", "hgb", "hgb_unweighted"]
# The ablation shares the weighted model's colour, dashed: same model, weighting switched off.
STYLE = {"tree": (ps.SERIES[2], "-"), "forest": (ps.SERIES[0], "-"),
         "hgb": (ps.SERIES[1], "-"), "hgb_unweighted": (ps.SERIES[1], "--")}
COST, REWARDS = 5, (25, 50, 100)   # a call costs 5; a subscription earns 25/50/100 (ratio 5/10/20)
CURVE_NOTE = "Curves: seed 42. Numbers: mean ± std over seeds 42, 43, 44."


def load_checkpoint(model, fs, seed):
    # Safe to unpickle: these files are written locally by src.train and never downloaded.
    return joblib.load(CHECKPOINTS / f"{model}_{fs}_{seed}.joblib")


def best_model(path=RESULTS / "val_metrics.csv", fs="DEPLOYABLE"):
    """The model with the highest mean VALIDATION PR-AUC. The test set plays no part in
    this choice, and the ablation is not a candidate."""
    df = pd.read_csv(path)
    d = df[(df["metric"] == "pr_auc") & (df["feature_set"] == fs)
           & df["model"].isin(["tree", "forest", "hgb"])]
    return d.groupby("model")["value"].mean().idxmax()


def mean_std(path, model, metric, fs="DEPLOYABLE"):
    df = pd.read_csv(path)
    v = df[(df["model"] == model) & (df["feature_set"] == fs) & (df["metric"] == metric)]["value"]
    return v.mean(), v.std()


def predictions(fs="DEPLOYABLE", names=tuple(MODEL_ORDER)):
    """Validation and test scores of every saved model: {(model, seed): {...}}."""
    X, y = data.load()
    Xf = data.feature_set(X, fs)
    out = {}
    for seed in data.SEEDS:
        _, X_val, X_test, _, y_val, y_test = data.split(Xf, y, seed)
        for m in names:
            ck = load_checkpoint(m, fs, seed)
            out[(m, seed)] = {"y_val": y_val.to_numpy(), "p_val": ck["model"].predict_proba(X_val)[:, 1],
                              "y_test": y_test.to_numpy(), "p_test": ck["model"].predict_proba(X_test)[:, 1],
                              "threshold": ck["threshold"]}
    return out


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"saved results/figures/{name}")


def plot_pr_curves(pred, path):
    """F3: precision-recall curves on the test set (DEPLOYABLE)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for m in CURVE_MODELS:
        d = pred[(m, 42)]
        prec, rec, _ = precision_recall_curve(d["y_test"], d["p_test"])
        mu, sd = mean_std(path, m, "pr_auc")
        c, ls = STYLE[m]
        ax.plot(rec, prec, color=c, linestyle=ls, linewidth=1.8, label=f"{LABELS[m]}   {mu:.3f} ± {sd:.3f}")
    prev = pred[("dummy", 42)]["y_test"].mean()
    ax.axhline(prev, color=ps.NO_COLOR, linestyle="--", linewidth=1,
               label=f"Trivial baseline = positive rate   {prev:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall (share of all subscribers found)",
           ylabel="Precision (share of calls that subscribe)")
    ax.legend(title="Test PR-AUC", loc="upper right", fontsize=8, title_fontsize=8)
    ax.set_title("Precision-recall on the test set, without duration")
    fig.text(0.01, -0.02, CURVE_NOTE, fontsize=8, color=ps.INK_2)
    save(fig, "F3_pr_curves.png")


def plot_roc_curves(pred, path):
    """F4: ROC curves on the test set (DEPLOYABLE)."""
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    for m in CURVE_MODELS:
        d = pred[(m, 42)]
        fpr, tpr, _ = roc_curve(d["y_test"], d["p_test"])
        mu, sd = mean_std(path, m, "roc_auc")
        c, ls = STYLE[m]
        ax.plot(fpr, tpr, color=c, linestyle=ls, linewidth=1.8, label=f"{LABELS[m]}   {mu:.3f} ± {sd:.3f}")
    ax.plot([0, 1], [0, 1], color=ps.NO_COLOR, linestyle="--", linewidth=1,
            label="Trivial baseline   0.500")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate (recall)")
    ax.legend(title="Test ROC-AUC", loc="lower right", fontsize=8, title_fontsize=8)
    ax.set_title("ROC on the test set, without duration")
    fig.text(0.01, -0.02, CURVE_NOTE, fontsize=8, color=ps.INK_2)
    save(fig, "F4_roc_curves.png")


def gains(y, p, frac):
    """Share of all subscribers found when calling the top `frac` of clients, ranked by p."""
    order = np.argsort(-p, kind="stable")
    found = np.concatenate([[0.0], np.cumsum(y[order]) / y.sum()])
    return found[np.ceil(frac * len(y)).astype(int)]


def plot_gains(pred, best):
    """F5: cumulative gains and lift on the test set, mean over seeds (band = ±1 std)."""
    frac = np.linspace(0, 1, 201)
    rows = []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for m in ["tree", "forest", "hgb"]:
        g = np.array([gains(pred[(m, s)]["y_test"], pred[(m, s)]["p_test"], frac) for s in data.SEEDS])
        mu, sd = g.mean(0), g.std(0, ddof=1)
        c, ls = STYLE[m]
        lw = 2.4 if m == best else 1.5
        a1.plot(frac * 100, mu * 100, color=c, linestyle=ls, linewidth=lw, label=LABELS[m])
        a1.fill_between(frac * 100, (mu - sd) * 100, (mu + sd) * 100, color=c, alpha=0.15, linewidth=0)
        k = frac >= 0.05
        a2.plot(frac[k] * 100, mu[k] / frac[k], color=c, linestyle=ls, linewidth=lw, label=LABELS[m])
        for pct in (5, 10, 20, 30, 50):
            i = int(np.argmin(np.abs(frac - pct / 100)))
            rows.append({"model": m, "top_pct_called": pct, "share_found_mean": mu[i],
                         "share_found_std": sd[i], "lift": mu[i] / frac[i]})
    y = pred[(best, 42)]["y_test"]
    a1.plot([0, 100], [0, 100], color=ps.NO_COLOR, linestyle="--", linewidth=1, label="Random calling")
    a1.plot([0, y.mean() * 100, 100], [0, 100, 100], color=ps.NO_COLOR, linestyle=":", linewidth=1,
            label="Perfect ranking")
    table = pd.DataFrame(rows)
    at20 = table[(table["model"] == best) & (table["top_pct_called"] == 20)].iloc[0]
    a1.scatter([20], [at20["share_found_mean"] * 100], color=STYLE[best][0], zorder=5)
    a1.annotate(f"Call top 20% → find {at20['share_found_mean']:.0%} of subscribers",
                (20, at20["share_found_mean"] * 100), textcoords="offset points", xytext=(10, -14),
                fontsize=9, color=ps.INK)
    a1.set(xlim=(0, 100), ylim=(0, 101), xlabel="% of clients called (highest score first)",
           ylabel="% of all subscribers found")
    a1.legend(loc="lower right", fontsize=8)
    a1.set_title("Cumulative gains (test set, no duration)")
    a2.axhline(1, color=ps.NO_COLOR, linestyle="--", linewidth=1)
    a2.set(xlim=(0, 100), ylim=(0, None), xlabel="% of clients called",
           ylabel="Lift (subscriber rate vs random calling)")
    a2.set_title("Cumulative lift")
    fig.text(0.01, -0.03, "Lines: mean over seeds 42, 43, 44; band: ±1 std.", fontsize=8, color=ps.INK_2)
    save(fig, "F5_gains_lift.png")
    table.to_csv(RESULTS / "gains_test.csv", index=False)
    return table


def permutation_table(model, fs="DEPLOYABLE", n_repeats=10):
    """Drop in test PR-AUC when one column is shuffled (10 repeats x 3 seeds)."""
    X, y = data.load()
    Xf = data.feature_set(X, fs)
    rows = []
    for seed in data.SEEDS:
        _, _, X_test, _, _, y_test = data.split(Xf, y, seed)
        r = permutation_importance(load_checkpoint(model, fs, seed)["model"], X_test, y_test,
                                   scoring="average_precision", n_repeats=n_repeats,
                                   random_state=seed, n_jobs=1)
        for j, col in enumerate(X_test.columns):
            rows += [{"model": model, "seed": seed, "feature": col, "pr_auc_drop": v}
                     for v in r.importances[j]]
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "permutation_importance.csv", index=False)
    return t


def plot_importance(t, model):
    """F6: permutation importance of the chosen model on the test set."""
    g = t.groupby("feature")["pr_auc_drop"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.barh(g.index, g["mean"], xerr=g["std"], color=ps.SERIES[0],
            error_kw={"ecolor": ps.INK_2, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=ps.INK_2, linewidth=0.8)
    ax.set_xlabel("Drop in test PR-AUC when the column is shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"What the model uses: permutation importance ({LABELS[model]})", fontsize=11)
    fig.text(0.01, -0.02, "Mean ± std over 10 shuffles x 3 seeds, test set, no duration.",
             fontsize=8, color=ps.INK_2)
    save(fig, "F6_permutation_importance.png")
    return g


def plot_confusion(pred, best, seed=42):
    """F7: confusion matrix on the test set at the threshold chosen on validation."""
    d = pred[(best, seed)]
    cm = confusion_matrix(d["y_test"], (d["p_test"] >= d["threshold"]).astype(int))
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]:,}\n{share:.0%} of row", ha="center", va="center", fontsize=11,
                    color="white" if share > 0.6 else ps.INK)
    ax.set_xticks([0, 1], ["No call / predicted no", "Call / predicted yes"])
    ax.set_yticks([0, 1], ["Actual no", "Actual yes"])
    ax.grid(False)
    prec = cm[1, 1] / cm[:, 1].sum()
    rec = cm[1, 1] / cm[1].sum()
    ax.set_title(f"{LABELS[best]}: precision {prec:.2f}, recall {rec:.2f}", fontsize=11)
    fig.text(0.01, -0.02, f"Test set, seed {seed}, threshold {d['threshold']:.3f} chosen on validation (max F1).",
             fontsize=8, color=ps.INK_2)
    save(fig, "F7_confusion_matrix.png")
    return cm


def profit(y, p, t, cost, reward):
    """Every called client costs `cost`; every subscriber among them earns `reward`."""
    call = p >= t
    return reward * int((call & (y == 1)).sum()) - cost * int(call.sum())


def profit_table(pred):
    """Threshold chosen on validation to maximise profit, then reported on test."""
    grid = np.round(np.arange(0, 1.0001, 0.005), 3)
    rows = []
    for m in CURVE_MODELS:
        for s in data.SEEDS:
            d = pred[(m, s)]
            for r in REWARDS:
                val = [profit(d["y_val"], d["p_val"], t, COST, r) for t in grid]
                t_star = grid[int(np.argmax(val))]
                yt, pt = d["y_test"], d["p_test"]
                rows.append({"model": m, "seed": s, "reward_cost_ratio": r // COST, "threshold": t_star,
                             "c_over_r": COST / r, "test_profit": profit(yt, pt, t_star, COST, r),
                             "call_everyone": profit(yt, pt, 0.0, COST, r),
                             "share_called": float((pt >= t_star).mean()),
                             "share_subscribers_reached": float(yt[pt >= t_star].sum() / yt.sum())})
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "profit_test.csv", index=False)
    return t


def plot_profit(pred, table, best, seed=42):
    """F8: test profit per 1,000 clients vs threshold; dots mark the validation-chosen threshold."""
    d = pred[(best, seed)]
    grid = np.round(np.arange(0, 1.0001, 0.005), 3)
    n = len(d["y_test"])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, r in enumerate(REWARDS):
        c = ps.SERIES[[2, 0, 4][i]]
        curve = np.array([profit(d["y_test"], d["p_test"], t, COST, r) for t in grid]) / n * 1000
        ax.plot(grid, curve, color=c, label=f"reward/cost = {r // COST}")
        row = table[(table["model"] == best) & (table["seed"] == seed) & (table["reward_cost_ratio"] == r // COST)].iloc[0]
        ax.scatter([row["threshold"]], [row["test_profit"] / n * 1000], color=c, zorder=5, s=30)
        ax.axvline(COST / r, color=c, linestyle=":", linewidth=1)
    ax.axhline(0, color=ps.INK_2, linewidth=0.8)
    ax.set(xlim=(0, 1), xlabel="Threshold (call if score ≥ threshold)",
           ylabel="Test profit per 1,000 clients on the list")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Expected profit vs threshold ({LABELS[best]})", fontsize=11)
    fig.text(0.01, -0.03, f"Call cost {COST}. Dots: threshold chosen on validation. Dotted: cost/reward "
             f"(the optimum if scores were calibrated probabilities). Test set, seed {seed}.",
             fontsize=8, color=ps.INK_2)
    save(fig, "F8_profit_threshold.png")


def plot_reliability(pred, path, seed=42):
    """F9: calibration of weighted vs unweighted boosting (and the forest) on the test set."""
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot([0, 1], [0, 1], color=ps.NO_COLOR, linestyle="--", linewidth=1, label="Perfectly calibrated")
    for m in ["forest", "hgb", "hgb_unweighted"]:
        d = pred[(m, seed)]
        frac_pos, mean_pred = calibration_curve(d["y_test"], d["p_test"], n_bins=10, strategy="quantile")
        mu, _ = mean_std(path, m, "brier")
        c, ls = STYLE[m]
        ax.plot(mean_pred, frac_pos, color=c, linestyle=ls, marker="o", markersize=4,
                label=f"{LABELS[m]}   Brier {mu:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability (10 equal-size bins)",
           ylabel="Observed share of subscribers")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Class weighting inflates the probabilities", fontsize=11)
    fig.text(0.01, -0.02, f"Test set, seed {seed}. Brier: mean over 3 seeds (lower is better).",
             fontsize=8, color=ps.INK_2)
    save(fig, "F9_reliability.png")


def test_report(path):
    print("\n" + "=" * 70 + "\nTEST SET (mean ± std over seeds 42/43/44; thresholds from validation)\n")
    print(summary(path, metrics=("pr_auc", "roc_auc", "brier", "precision", "recall", "f1",
                                 "train_time_s")).to_string())
    print("\nDURATION GAP on test (FULL - DEPLOYABLE)\n")
    print(duration_gap(path).to_string())
    plot_duration_gap(path, name="F2b_duration_gap_test.png", split="test")
    print("saved results/figures/F2b_duration_gap_test.png")

    best = best_model()
    print(f"\nchosen model (highest mean validation PR-AUC, DEPLOYABLE): {best}\n")
    pred = predictions()
    plot_pr_curves(pred, path)
    plot_roc_curves(pred, path)
    g = plot_gains(pred, best)
    print(g.pivot(index="top_pct_called", columns="model", values="share_found_mean").round(3).to_string())
    cm = plot_confusion(pred, best)
    print(f"confusion matrix (seed 42): TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}")
    pt = profit_table(pred)
    plot_profit(pred, pt, best)
    print("\nPROFIT on test (cost 5 per call; mean over seeds)\n")
    print(pt.groupby(["reward_cost_ratio", "model"])[
        ["threshold", "test_profit", "call_everyone", "share_called", "share_subscribers_reached"]]
        .mean().round(3).to_string())
    plot_reliability(pred, path)
    imp = plot_importance(permutation_table(best), best)
    print("\nPERMUTATION IMPORTANCE (drop in test PR-AUC)\n")
    print(imp.sort_values("mean", ascending=False).round(4).to_string())
    errors.run(best)


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
    abl = summary(RESULTS / "val_metrics.csv",
                  metrics=("pr_auc", "roc_auc", "brier", "threshold", "precision", "recall", "n_iter"))
    abl = abl[abl.index.get_level_values("model").isin(["hgb", "hgb_unweighted"])]
    if len(abl):
        print("\nCLASS-WEIGHT ABLATION (same configuration, only class_weight differs)\n")
        print(abl.to_string())
    if (RESULTS / "dt_cv.csv").exists():
        plot_tree_cv()
        print("saved results/figures/DT_cv_depth_curve.png")
    if (RESULTS / "metrics.csv").exists():
        test_report(RESULTS / "metrics.csv")
    else:
        print("\nno results/metrics.csv yet: run `python -m src.train --final` for the test-set report")


if __name__ == "__main__":
    main()
