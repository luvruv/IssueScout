import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, GridSearchCV
from features.build_features import load, add_features, add_labels
from ml.common import FEATURES, preprocessor, temporal_split, bootstrap_ci, save_metrics


def main():
    df = add_labels(add_features(load(state="closed")))
    df = df[df.pr_merged == 1]
    train, test = temporal_split(df)                       # old -> train, new -> test
    Xtr, ytr, Xte, yte = train[FEATURES], train["difficulty"], test[FEATURES], test["difficulty"]
    f1m = lambda a, b: f1_score(a, b, average="macro")
    majority = ytr.mode()[0]

    rows = {
        "baseline: majority": {"test_f1": round(f1m(yte, [majority] * len(yte)), 3)},
        "baseline: good-first-issue label": {"test_f1": round(f1m(yte, np.where(test.is_gfi == 1, "Easy", majority)), 3)},
    }
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "LinearSVM": LinearSVC(class_weight="balanced"),
    }
    for name, clf in models.items():
        pipe = Pipeline([("pre", preprocessor()), ("clf", clf)])
        cv = cross_val_score(pipe, Xtr, ytr, cv=5, scoring="f1_macro").mean()
        pipe.fit(Xtr, ytr)
        rows[name] = {"cv_f1": round(cv, 3), "test_f1": round(f1m(yte, pipe.predict(Xte)), 3)}
        print(name, rows[name])

    # Hyperparameter tuning on Random Forest
    pipe = Pipeline([("pre", preprocessor()),
                     ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1))])
    grid = GridSearchCV(pipe, {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 20],
                               "clf__min_samples_leaf": [1, 3]}, cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(Xtr, ytr)
    pred = grid.predict(Xte)
    print("Best params:", grid.best_params_)
    print(classification_report(yte, pred))
    rows["RandomForest (tuned)"] = {"test_f1": round(f1m(yte, pred), 3),
                                    "f1_95ci": bootstrap_ci(yte, pred, f1m), "params": grid.best_params_}

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(yte, pred, ax=ax)
    fig.savefig("report/confusion_difficulty.png", dpi=150, bbox_inches="tight")
    save_metrics("difficulty", rows)
    print(pd.DataFrame(rows).T)

    # Final model for deployment: refit on everything
    final = grid.best_estimator_.fit(df[FEATURES], df["difficulty"])
    joblib.dump(final, "models/difficulty.joblib")
    print("saved models/difficulty.joblib")