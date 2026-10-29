"""Which label definition can the model actually learn?   python -m ml.compare_labels"""
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import features.labels as L
from features.build_features import add_features, load, FEATURES
from ml.common import preprocessor, group_splits, temporal_split

warnings.filterwarnings("ignore")


def pipe():
    return Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3))])


def run(raw, shrink, name):
    L.SHRINK = shrink
    d = L.assign_labels(raw).sort_values("created_at").reset_index(drop=True)
    easy = (d.difficulty == "Easy").astype(int)
    f, a = [], []
    for tr, te in group_splits(d):
        f.append(f1_score(d.difficulty.iloc[te], pipe().fit(d.iloc[tr][FEATURES], d.difficulty.iloc[tr]).predict(d.iloc[te][FEATURES]), average="macro"))
        if easy.iloc[te].nunique() > 1:
            a.append(roc_auc_score(easy.iloc[te], pipe().fit(d.iloc[tr][FEATURES], easy.iloc[tr]).predict_proba(d.iloc[te][FEATURES])[:, 1]))
    tr, te = temporal_split(d)
    tf = f1_score(te.difficulty, pipe().fit(tr[FEATURES], tr.difficulty).predict(te[FEATURES]), average="macro")
    ta = roc_auc_score((te.difficulty == "Easy").astype(int), pipe().fit(tr[FEATURES], (tr.difficulty == "Easy").astype(int)).predict_proba(te[FEATURES])[:, 1])
    print(f"{name:36s} UNSEEN repos: F1 {np.mean(f):.3f} Easy-AUC {np.mean(a):.3f} | KNOWN repos: F1 {tf:.3f} Easy-AUC {ta:.3f}")


def main():
    raw = add_features(load(state="closed")); raw = raw[(raw.pr_merged == 1) & raw.pr_files.notna()].copy()
    print(f"{len(raw)} labelled issues, {raw.repo.nunique()} repos")
    run(raw, 10, "A: effort relative to its own repo")
    run(raw, 1e9, "B: absolute effort (all repos pooled)")


if __name__ == "__main__":
    main()