import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import cohen_kappa_score
from features.build_features import load, add_features, add_labels
from ml.common import FEATURES, preprocessor


def main():
    df = add_labels(add_features(load(state="closed")))
    df = df[df.pr_merged == 1].sort_values("created_at")
    plt.figure(figsize=(7, 5))
    for name, clf in {"DecisionTree": DecisionTreeClassifier(random_state=42),
                      "RandomForest": RandomForestClassifier(200, random_state=42, n_jobs=-1)}.items():
        pipe = Pipeline([("pre", preprocessor()), ("clf", clf)])
        sizes, tr, va = learning_curve(pipe, df[FEATURES], df["difficulty"], cv=5,
                                       scoring="f1_macro", train_sizes=np.linspace(0.2, 1, 5))
        plt.plot(sizes, tr.mean(1), "--", label=f"{name} train")
        plt.plot(sizes, va.mean(1), label=f"{name} validation")
    plt.xlabel("training size"); plt.ylabel("macro F1"); plt.legend(); plt.title("Learning curves")
    plt.savefig("report/learning_curves.png", dpi=150)

    p = "data/processed/manual_label_sample.csv"
    if os.path.exists(p):
        m = pd.read_csv(p).dropna(subset=["my_difficulty"])
        m = m[m.my_difficulty.astype(str).str.strip() != ""]
        if len(m):
            agree = (m.difficulty == m.my_difficulty).mean()
            kappa = cohen_kappa_score(m.difficulty, m.my_difficulty)
            print(f"Manual vs auto label: agreement={agree:.2f}, kappa={kappa:.2f} on {len(m)} issues")