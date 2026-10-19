"""Will the PR for this issue be merged?  (only issues that have a linked closed PR are observable)"""
import warnings
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, PrecisionRecallDisplay
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from features.build_features import load, add_features, FEATURES
from ml.common import preprocessor, temporal_split, group_splits, save_metrics, write_table

warnings.filterwarnings("ignore")


def main():
    df = add_features(load(state="closed"))
    df = df[df.pr_number.notna()].copy().sort_values("created_at").reset_index(drop=True)
    df["y"] = df.pr_merged.astype(int)
    train, test = temporal_split(df)
    folds = group_splits(df)
    base = float(df.y.mean())
    print(f"{len(df)} PR-linked issues, merged rate {base:.3f} (imbalanced: judge by PR-AUC vs this base rate AND ROC-AUC)")
    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }
    rows, fig, ax = {}, *plt.subplots(figsize=(6, 5))
    for n, clf in models.items():
        p = lambda: Pipeline([("pre", preprocessor()), ("clf", clone(clf))])
        cv = cross_validate(p(), df[FEATURES], df["y"], cv=folds, scoring="average_precision", error_score=np.nan)
        m = p().fit(train[FEATURES], train["y"]); pr = m.predict_proba(test[FEATURES])[:, 1]
        rows[n] = {"cv_ap_grouped": float(np.nanmean(cv["test_score"])), "test_ap": average_precision_score(test.y, pr),
                   "test_roc_auc": roc_auc_score(test.y, pr) if test.y.nunique() > 1 else np.nan, "test_base_rate": float(test.y.mean())}
        PrecisionRecallDisplay.from_predictions(test.y, pr, name=n, ax=ax)
        print(f"  {n:20s} grouped-CV AP {rows[n]['cv_ap_grouped']:.3f} | test AP {rows[n]['test_ap']:.3f} (base {rows[n]['test_base_rate']:.3f}) | ROC-AUC {rows[n]['test_roc_auc']:.3f}")
    rows["baseline: merge rate"] = {"test_ap": float(test.y.mean())}
    ax.set_title("Precision-recall (test)"); fig.savefig("report/pr_merge.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    write_table(rows, "merge_comparison"); save_metrics("merge", {k: {a: round(b, 4) for a, b in v.items()} for k, v in rows.items()})
    best = max(models, key=lambda n: np.nan_to_num(rows[n]["cv_ap_grouped"], nan=-1))
    final = CalibratedClassifierCV(Pipeline([("pre", preprocessor()), ("clf", clone(models[best]))]), method="sigmoid", cv=3).fit(df[FEATURES], df["y"])
    joblib.dump(final, "models/merge.joblib")
    ok = rows[best]["test_roc_auc"] > 0.70
    print(f"saved models/merge.joblib ({best}). Useful ranking signal? {'yes' if ok else 'WEAK (ROC-AUC<=0.70): the plan leans on the repo-level Bayesian rate instead'}")