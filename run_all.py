"""
Why Did This Post Go Viral? Predicting Facebook Post Engagement Before Publishing
COMP 365 Individual Project - Final reproducible run
Enioluwa Olabode

One command reproduces every table and figure:

    python run_all.py

Part 1 (primary): time-ordered evaluation, as requested in the Checkpoint 2 feedback.
    Posts are ordered by publication time. Months 1-6 are used to fit, months 7-9
    to choose each model's feature set and decision threshold, and months 10-12
    are held out and evaluated once. The high-engagement cutoff is computed from
    training months only.

Part 2 (secondary): the Checkpoint 2 random-split tests, rerun with a stronger
    statistical treatment: a 200-permutation label-shuffle test and a corrected
    resampled t-test (repeated folds share posts, so they are not independent).

Every design choice below is declared before the test months are scored.
"""

import json
import os
import platform
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------- settings
DATA_PATH = "data/dataset_Facebook.csv"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
RANDOM_STATE = 42

PRE_PUBLICATION_FEATURES = ["Page total likes", "Type", "Category", "Post Month",
                            "Post Weekday", "Post Hour", "Paid"]
TARGET_SOURCE = "Total Interactions"
HIGH_ENGAGEMENT_QUANTILE = 0.75

# Time-ordered design (declared in advance)
FIT_MONTHS = range(1, 7)       # Jan-Jun: fit candidate models
VAL_MONTHS = range(7, 10)      # Jul-Sep: choose feature set + threshold per model
TEST_MONTHS = range(10, 13)    # Oct-Dec: scored once at the end
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
FEATURE_SETS = {
    "all 7 features": None,
    "no time-trend features": ["Page total likes", "Post Month"],
}
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 200

MODEL_NAMES = ["Baseline: majority class", "Baseline: rule of thumb",
               "Decision Tree", "Logistic Regression", "Random Forest"]
LEARNED = ["Decision Tree", "Logistic Regression", "Random Forest"]


class RuleOfThumb(BaseEstimator, ClassifierMixin):
    """Non-ML baseline: predict HIGH if the post is paid, a video, or a status update."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        rule = (X["Paid"] == 1) | (X.get("Type_Video", 0) == 1) | (X.get("Type_Status", 0) == 1)
        return np.asarray(rule).astype(int)


def make_model(name):
    if name == "Baseline: majority class":
        return DummyClassifier(strategy="most_frequent")
    if name == "Baseline: rule of thumb":
        return RuleOfThumb()
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=10,
                                      class_weight="balanced", random_state=RANDOM_STATE)
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=1000))
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                      class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
    raise ValueError(name)


def scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def log(text=""):
    print(text)
    with open(os.path.join(RESULTS_DIR, "output.txt"), "a") as fh:
        fh.write(text + "\n")


def header(title):
    log("\n" + "=" * 76)
    log(title)
    log("=" * 76)


# ---------------------------------------------------------------- data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    df = df[PRE_PUBLICATION_FEATURES + [TARGET_SOURCE]].dropna().copy()
    # The file is in reverse chronological order (December first). Reverse it so
    # row order = publication order; "time_rank" 0 is the earliest post.
    df = df.iloc[::-1].reset_index(drop=True)
    df["time_rank"] = np.arange(len(df))
    assert df["Post Month"].is_monotonic_increasing, "posts are not in time order"
    return df


def encode(df, drop=None):
    X = pd.get_dummies(df[PRE_PUBLICATION_FEATURES], columns=["Type"], prefix="Type").astype(float)
    for col in ["Type_Link", "Type_Photo", "Type_Status", "Type_Video"]:
        if col not in X:
            X[col] = 0.0
    X = X[sorted(X.columns)]
    return X.drop(columns=drop) if drop else X


def label(df, cutoff):
    return (df[TARGET_SOURCE] > cutoff).astype(int).to_numpy()


def predict(model, X, threshold):
    if threshold is None or not hasattr(model, "predict_proba"):
        return model.predict(X)
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


# ---------------------------------------------------------------- part 1
def part1_time_ordered(df):
    header("PART 1: TIME-ORDERED EVALUATION (primary result)")
    fit = df[df["Post Month"].isin(FIT_MONTHS)]
    val = df[df["Post Month"].isin(VAL_MONTHS)]
    dev = df[df["Post Month"].isin(list(FIT_MONTHS) + list(VAL_MONTHS))]
    test = df[df["Post Month"].isin(TEST_MONTHS)]

    # Stage 1: selection. Cutoff from the fit months only.
    cut_fit = fit[TARGET_SOURCE].quantile(HIGH_ENGAGEMENT_QUANTILE)
    y_fit, y_val = label(fit, cut_fit), label(val, cut_fit)
    log(f"Fit months 1-6:        {len(fit)} posts | cutoff (75th pct of fit) = {cut_fit:.0f} "
        f"| high in fit: {y_fit.sum()} ({y_fit.mean():.0%})")
    log(f"Validation months 7-9: {len(val)} posts | high by that cutoff: {y_val.sum()} ({y_val.mean():.0%})")

    selection_rows, chosen = [], {}
    for name in MODEL_NAMES:
        best = None
        for fs_name, drop in FEATURE_SETS.items():
            model = make_model(name).fit(encode(fit, drop), y_fit)
            if name in LEARNED:
                proba = model.predict_proba(encode(val, drop))[:, 1]
                f1s = [f1_score(y_val, (proba >= t).astype(int), zero_division=0) for t in THRESHOLD_GRID]
                i = int(np.argmax(f1s))
                cand = (f1s[i], fs_name, float(THRESHOLD_GRID[i]))
                default = f1_score(y_val, (proba >= 0.5).astype(int), zero_division=0)
            else:
                f1v = f1_score(y_val, model.predict(encode(val, drop)), zero_division=0)
                cand, default = (f1v, fs_name, None), f1v
            selection_rows.append({"model": name, "feature set": fs_name,
                                   "val F1 at 0.50": default, "best threshold": cand[2],
                                   "val F1 at best threshold": cand[0]})
            if best is None or cand[0] > best[0] + 1e-12:
                best = cand
        chosen[name] = {"feature set": best[1], "threshold": best[2], "val F1": best[0]}

    sel = pd.DataFrame(selection_rows)
    sel.to_csv(f"{RESULTS_DIR}/table_selection_validation.csv", index=False)
    log("\nValidation (months 7-9) - every candidate:")
    log(sel.round(3).to_string(index=False))
    log("\nChosen settings per model (by validation F1):")
    for name, c in chosen.items():
        thr = "n/a" if c["threshold"] is None else f"{c['threshold']:.2f}"
        log(f"  {name:<26} features: {c['feature set']:<24} threshold: {thr:<5} val F1: {c['val F1']:.3f}")
    overall = max(LEARNED, key=lambda n: chosen[n]["val F1"])
    log(f"\nModel selected for the paper's main claim (highest validation F1): {overall}")

    # Stage 2: refit on months 1-9 with the chosen settings, score months 10-12 once.
    cut_dev = dev[TARGET_SOURCE].quantile(HIGH_ENGAGEMENT_QUANTILE)
    y_dev, y_test = label(dev, cut_dev), label(test, cut_dev)
    log(f"\nRefit on months 1-9:   {len(dev)} posts | cutoff (75th pct of months 1-9) = {cut_dev:.0f}")
    log(f"Test months 10-12:     {len(test)} posts | high by that cutoff: {y_test.sum()} ({y_test.mean():.0%})")

    rows, preds, rng = [], {}, np.random.default_rng(RANDOM_STATE)
    boot_idx = [rng.integers(0, len(test), len(test)) for _ in range(N_BOOTSTRAP)]
    for name in MODEL_NAMES:
        c = chosen[name]
        drop = FEATURE_SETS[c["feature set"]]
        model = make_model(name).fit(encode(dev, drop), y_dev)
        p = predict(model, encode(test, drop), c["threshold"])
        preds[name] = p
        boot = [f1_score(y_test[i], p[i], zero_division=0) for i in boot_idx]
        rows.append({"model": name, "feature set": c["feature set"],
                     "threshold": c["threshold"], "predicted high": int(p.sum()), **scores(y_test, p),
                     "f1 95% CI low": np.percentile(boot, 2.5), "f1 95% CI high": np.percentile(boot, 97.5)})
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(f"{RESULTS_DIR}/table_time_test.csv")
    log("\nTEST RESULTS (months 10-12, scored once):")
    log(table.drop(columns=["feature set", "threshold"]).round(3).to_string())

    # POST HOC (added after the test months were scored; NOT used to choose anything):
    # the same refit models at the default 0.50 threshold, to see whether the
    # validation-tuned thresholds caused the test result.
    post = []
    for name in LEARNED:
        drop = FEATURE_SETS[chosen[name]["feature set"]]
        model = make_model(name).fit(encode(dev, drop), y_dev)
        p = predict(model, encode(test, drop), 0.5)
        post.append({"model": name, "threshold": 0.5, "predicted high": int(p.sum()), **scores(y_test, p)})
    post = pd.DataFrame(post).set_index("model")
    post.to_csv(f"{RESULTS_DIR}/table_time_test_posthoc_default_threshold.csv")
    log("\nPOST HOC sensitivity (not used for selection): same models at the default 0.50 threshold")
    log(post.round(3).to_string())

    # Paired bootstrap: selected model minus rule of thumb on the same resampled posts.
    diffs = np.array([f1_score(y_test[i], preds[overall][i], zero_division=0)
                      - f1_score(y_test[i], preds["Baseline: rule of thumb"][i], zero_division=0)
                      for i in boot_idx])
    paired = {"selected model": overall, "f1 difference": float(table.loc[overall, "f1"]
                                                                 - table.loc["Baseline: rule of thumb", "f1"]),
              "95% CI low": float(np.percentile(diffs, 2.5)), "95% CI high": float(np.percentile(diffs, 97.5)),
              "share of resamples where model > rule": float((diffs > 0).mean())}
    log(f"\nPaired bootstrap ({N_BOOTSTRAP} resamples of the test posts), {overall} minus rule of thumb:")
    log(f"  F1 difference {paired['f1 difference']:+.3f}, 95% CI [{paired['95% CI low']:+.3f}, "
        f"{paired['95% CI high']:+.3f}], model ahead in {paired['share of resamples where model > rule']:.0%} of resamples")

    # Weak cases on the test months for the selected model.
    t = test.copy()
    t["high"], t["pred"] = y_test, preds[overall]
    by_type = (t.groupby("Type").apply(lambda g: pd.Series({
        "test posts": len(g), "high posts": int(g["high"].sum()),
        "high found": int(((g["high"] == 1) & (g["pred"] == 1)).sum()),
        "false alarms": int(((g["high"] == 0) & (g["pred"] == 1)).sum()),
        "recall": (g.loc[g["high"] == 1, "pred"].mean() if g["high"].sum() else np.nan),
        "accuracy": (g["high"] == g["pred"]).mean()})))
    by_type.to_csv(f"{RESULTS_DIR}/table_time_test_by_type.csv")
    log(f"\nWEAK CASES: {overall} on months 10-12, by content type")
    log(by_type.round(3).to_string())
    missed = t[(t["high"] == 1) & (t["pred"] == 0)].sort_values(TARGET_SOURCE, ascending=False)
    missed[PRE_PUBLICATION_FEATURES + [TARGET_SOURCE]].to_csv(f"{RESULTS_DIR}/table_time_test_missed.csv")
    log(f"\nMissed high-engagement test posts: {len(missed)} of {int(y_test.sum())}. Five biggest:")
    log(missed[PRE_PUBLICATION_FEATURES + [TARGET_SOURCE]].head(5).to_string())

    monthly = df.groupby("Post Month").agg(posts=(TARGET_SOURCE, "size"),
                                           median_interactions=(TARGET_SOURCE, "median"),
                                           page_likes_end=("Page total likes", "max"))
    monthly.to_csv(f"{RESULTS_DIR}/table_monthly.csv")

    return {"cut_fit": cut_fit, "cut_dev": cut_dev, "chosen": chosen, "overall": overall,
            "table": table, "paired": paired, "by_type": by_type, "monthly": monthly,
            "selection": sel, "n_test_high": int(y_test.sum()), "n_test": len(test)}


# ---------------------------------------------------------------- part 2
def corrected_t_test(a, b, n_train, n_test, k):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / n + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(d.mean()), float(t), float(p)


def part2_random_cv(df):
    header("PART 2: RANDOM-SPLIT CROSS-VALIDATION (Checkpoint 2 rerun, secondary)")
    cutoff = df[TARGET_SOURCE].quantile(HIGH_ENGAGEMENT_QUANTILE)
    y = label(df, cutoff)
    X = encode(df)
    log(f"All {len(df)} posts | cutoff (75th pct of all posts, as in Checkpoint 2) = {cutoff:.0f} "
        f"| high: {y.sum()}")
    log("Note: here the cutoff uses all posts, so this part is for comparison with Checkpoint 2 only.")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

    rows, per_fold = [], {}
    for name in MODEL_NAMES:
        res = cross_validate(make_model(name), X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1"])
        per_fold[name] = res["test_f1"]
        rows.append({"model": name, "accuracy": res["test_accuracy"].mean(),
                     "precision": res["test_precision"].mean(), "recall": res["test_recall"].mean(),
                     "f1_mean": res["test_f1"].mean(), "f1_std": res["test_f1"].std()})
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(f"{RESULTS_DIR}/table_random_cv.csv")
    log("\nRepeated stratified 5-fold CV (5 repeats = 25 folds):")
    log(table.round(3).to_string())

    n_test = len(y) // 5
    tt = {}
    for name in LEARNED:
        tt[name] = corrected_t_test(per_fold[name], per_fold["Baseline: rule of thumb"],
                                    len(y) - n_test, n_test, 5)
    log("\nCorrected resampled t-test vs rule of thumb (Nadeau & Bengio 2003; folds share posts):")
    for name, (gain, t, p) in tt.items():
        log(f"  {name:<20} mean F1 gain {gain:+.3f}   t = {t:.2f}   p = {p:.3f}")

    obs, null, pval = permutation_test_score(make_model("Logistic Regression"), X, y, cv=cv, scoring="f1",
                                             n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE, n_jobs=1)
    log(f"\nLabel-permutation test, Logistic Regression ({N_PERMUTATIONS} shuffles, same 25 folds):")
    log(f"  real-label F1 {obs:.3f} | shuffled F1 mean {null.mean():.3f} (sd {null.std():.3f}, "
        f"max {null.max():.3f}) | p = {pval:.3f}")

    abl = []
    for label_, drop in [("all features", None), ("without Page total likes", ["Page total likes"]),
                         ("without Page total likes + Post Month", ["Page total likes", "Post Month"])]:
        f1 = cross_validate(make_model("Logistic Regression"), encode(df, drop), y, cv=cv, scoring="f1")["test_score"]
        abl.append({"features": label_, "f1_mean": f1.mean(), "f1_std": f1.std()})
    abl = pd.DataFrame(abl).set_index("features")
    abl.to_csv(f"{RESULTS_DIR}/table_random_cv_ablation.csv")
    log("\nAblation (Logistic Regression, same 25 folds):")
    log(abl.round(3).to_string())

    return {"table": table, "ttests": tt, "perm": (float(obs), null, float(pval)), "ablation": abl}


# ---------------------------------------------------------------- figures
def make_figures(p1, p2):
    m = p1["monthly"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colors = ["#c9d6e8"] * 6 + ["#f2d7a6"] * 3 + ["#e8b4a8"] * 3
    ax.bar(m.index, m["median_interactions"], color=colors, edgecolor="#555", linewidth=0.5)
    ax.axhline(p1["cut_dev"], color="#333", linestyle="--", linewidth=1)
    ax.text(0.6, p1["cut_dev"] + 6, f"high-engagement cutoff from months 1-9 (> {p1['cut_dev']:.0f})", fontsize=8)
    for x, n in zip(m.index, m["posts"]):
        ax.text(x, 4, f"n={n}", ha="center", fontsize=7, color="#333")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Post month (2014)")
    ax.set_ylabel("Median total interactions")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c9d6e8", label="fit (1-6)"), Patch(color="#f2d7a6", label="validation (7-9)"),
                       Patch(color="#e8b4a8", label="test (10-12)")], fontsize=8, loc="upper left", ncol=3)
    ax.set_ylim(0, max(m["median_interactions"].max(), p1["cut_dev"]) * 1.35)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig1_time_split.png", dpi=200)
    plt.close(fig)

    t = p1["table"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    names = [n.replace("Baseline: ", "Baseline:\n") for n in t.index]
    err = np.vstack([t["f1"] - t["f1 95% CI low"], t["f1 95% CI high"] - t["f1"]])
    cols = ["#b8b8b8", "#b8b8b8"] + ["#2f5d8a" if n == p1["overall"] else "#8fb0d0" for n in t.index[2:]]
    ax.bar(names, t["f1"], yerr=err, capsize=4, color=cols)
    for i, v in enumerate(t["f1"]):
        ax.text(i + 0.08, v + 0.01, f"{v:.3f}", fontsize=8, ha="left")
    ax.set_ylabel("F1 on months 10-12")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig2_time_test_f1.png", dpi=200)
    plt.close(fig)

    obs, null, pval = p2["perm"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.hist(null, bins=25, color="#b8b8b8", edgecolor="white")
    ax.axvline(obs, color="#2f5d8a", linewidth=2)
    ax.set_xlim(null.min() - 0.01, obs + 0.06)
    ax.text(obs + 0.004, ax.get_ylim()[1] * 0.75, f"real labels\nF1 = {obs:.3f}\np = {pval:.3f}", fontsize=8, color="#2f5d8a")
    ax.set_xlabel("Cross-validated F1 (Logistic Regression)")
    ax.set_ylabel(f"Count of {len(null)} label shuffles")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig3_permutation_test.png", dpi=200)
    plt.close(fig)

    bt = p1["by_type"]
    bt = bt[bt["high posts"] > 0]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    pos = np.arange(len(bt))
    ax.bar(pos - 0.2, bt["recall"], 0.4, color="#2f5d8a", label="Recall (high posts found)")
    ax.bar(pos + 0.2, bt["accuracy"], 0.4, color="#8fb0d0", label="Accuracy (all posts of that type)")
    for i, (_, r) in enumerate(bt.iterrows()):
        ax.text(i - 0.2, r["recall"] + 0.03, f"{int(r['high found'])}/{int(r['high posts'])}", ha="center", fontsize=8)
        ax.text(i + 0.2, r["accuracy"] + 0.03, f"n={int(r['test posts'])}", ha="center", fontsize=8)
    ax.set_xticks(pos)
    ax.set_xticklabels(bt.index)
    ax.set_xlabel("Content type")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.45)
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig4_test_by_type.png", dpi=200)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    open(os.path.join(RESULTS_DIR, "output.txt"), "w").close()
    log(f"Python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__} | "
        f"scikit-learn {sklearn.__version__} | scipy {scipy.__version__} | matplotlib {matplotlib.__version__}")
    df = load_data()
    log(f"Posts after dropping rows with missing values: {len(df)}")
    p1 = part1_time_ordered(df)
    p2 = part2_random_cv(df)
    make_figures(p1, p2)
    summary = {"time_test_selected_model": p1["overall"], "time_test_paired_bootstrap": p1["paired"],
               "time_test_f1": p1["table"]["f1"].round(4).to_dict(),
               "random_cv_f1": p2["table"]["f1_mean"].round(4).to_dict(),
               "permutation_p": p2["perm"][2]}
    with open(f"{RESULTS_DIR}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"\nTables and output written to {RESULTS_DIR}/, figures to {FIGURES_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
