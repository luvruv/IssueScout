import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, PrecisionRecallDisplay
from sklearn.model_selection import cross_val_score
from features.build_features import load, add_features
from ml.common import FEATURES, preprocessor, temporal_split, save_metrics


def main():
    df = add_features(load(state="closed"))
    df = df[df.pr_number.notna()].copy()
    df["y"] = df.pr_merged.astype(int)
    train, test = temporal_split(df)
    Xtr, ytr, Xte, yte = train[FEATURES], train["y"], test[FEATURES], test["y"]
    print("merged rate:", round(df.y.mean(), 3), "(imbalanced -> use class weights + PR curve)")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(300, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }
    res, fitted, fig_ax = {}, {}, plt.subplots(figsize=(6, 5))
    ax = fig_ax[1]
    for name, clf in models.items():
        pipe = Pipeline([("pre", preprocessor()), ("clf", clf)])
        cv_ap = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="average_precision").mean()
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        res[name] = {"cv_ap": round(cv_ap, 3), "test_ap": round(average_precision_score(yte, p), 3),
                     "test_roc_auc": round(roc_auc_score(yte, p), 3)}
        PrecisionRecallDisplay.from_predictions(yte, p, name=name, ax=ax)
        fitted[name] = pipe
        print(name, res[name])
    res["baseline: merge rate"] = {"test_ap": round(float(yte.mean()), 3)}
    fig_ax[0].savefig("report/pr_merge.png", dpi=150, bbox_inches="tight")
    save_metrics("merge", res)

    best = max(models, key=lambda n: res[n]["cv_ap"])     # choose by CV, not by test
    final = fitted[best].fit(df[FEATURES], df["y"])
    joblib.dump(final, "models/merge.joblib")
    print("saved models/merge.joblib from", best)