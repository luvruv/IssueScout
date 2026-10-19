"""Learning curves (bias-variance), manual-label agreement (kappa) and label-threshold sensitivity."""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from features.build_features import add_features, add_labels, load, labelled, FEATURES
from features.labels import assign_labels
from ml.common import preprocessor, temporal_split, save_metrics

warnings.filterwarnings("ignore")


def main():
    df = labelled().sort_values("created_at").reset_index(drop=True)
    plt.figure(figsize=(7, 4.8))
    cv = GroupKFold(n_splits=min(5, df.repo.nunique()))
    for name, clf, col in [("DecisionTree(deep)", DecisionTreeClassifier(random_state=42), "tab:red"),
                           ("RandomForest", RandomForestClassifier(200, random_state=42, n_jobs=-1), "tab:blue"),
                           ("LogisticRegression", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3), "tab:green")]:
        pipe = Pipeline([("pre", preprocessor()), ("clf", clf)])
        sz, tr, va = learning_curve(pipe, df[FEATURES], df["difficulty"], groups=df["repo"], cv=cv, scoring="f1_macro", train_sizes=np.linspace(.25, 1, 5), error_score=np.nan)
        plt.plot(sz, np.nanmean(tr, 1), "--", c=col, label=f"{name} train"); plt.plot(sz, np.nanmean(va, 1), c=col, label=f"{name} validation")
    plt.xlabel("training size"); plt.ylabel("macro F1"); plt.legend(fontsize=7); plt.title("Learning curves (grouped CV): a big train/validation gap = overfitting")
    plt.tight_layout(); plt.savefig("report/learning_curves.png", dpi=140); plt.close()

    p = "data/processed/manual_label_sample.csv"
    if os.path.exists(p):
        m = pd.read_csv(p); m = m[m.my_difficulty.astype(str).isin(["Easy", "Medium", "Hard"])]
        if len(m) >= 20:
            cur = df.set_index("id").difficulty
            m = m[m["id"].isin(cur.index)]; auto = m["id"].map(cur)       # compare with the CURRENT label recipe
            res = {"n": len(m), "agreement": round(float((auto == m.my_difficulty).mean()), 3), "kappa": round(float(cohen_kappa_score(auto, m.my_difficulty)), 3),
                   "within_one_level": round(float((auto.map({"Easy": 0, "Medium": 1, "Hard": 2}) - m.my_difficulty.map({"Easy": 0, "Medium": 1, "Hard": 2})).abs().le(1).mean()), 3)}
            print("manual vs automatic label:", res); save_metrics("label_validation", res)
        else:
            print(f"only {len(m)} manual labels so far: label at least 100-200 with `python cli.py label`")
    sens = {}
    for eq, hq in [(0.35, 0.75), (0.40, 0.80), (0.45, 0.85)]:
        raw = add_features(load(state="closed")); raw = raw[(raw.pr_merged == 1) & raw.pr_files.notna()]
        d = assign_labels(raw, eq, hq).sort_values("created_at"); tr, te = temporal_split(d)
        pipe = Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3))]).fit(tr[FEATURES], tr.difficulty)
        sens[f"{eq}/{hq}"] = round(float(f1_score(te.difficulty, pipe.predict(te[FEATURES]), average="macro")), 3)
    print("label-threshold sensitivity (LR test macro-F1):", sens); save_metrics("label_sensitivity", sens)