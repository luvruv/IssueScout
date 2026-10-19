"""Difficulty classification (Easy/Medium/Hard).  Every model sees the same data, the same folds and the same metrics.
  * temporal hold-out : train on the oldest 80%, test on the newest 20%  (realistic)
  * grouped CV by repo: every repo is entirely in train or test          (does it work on UNSEEN repos?)
Model choice uses the grouped-CV score, never the test score."""
import json, time, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from features.build_features import labelled, FEATURES, NUM
from features.labels import LABEL_VERSION, EASY_Q, HARD_Q
from ml.common import (preprocessor, temporal_split, group_splits, bootstrap_ci, save_metrics,
                       write_table, weighted_cost)

warnings.filterwarnings("ignore")
ORDER = ["Easy", "Medium", "Hard"]


def models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "SVM(linear)": CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3, method="sigmoid"),
        "KNN(cosine)": KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="distance"),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "MLP(64)": MLPClassifier((64,), alpha=1e-2, max_iter=500, random_state=42),
    }


GRIDS = {
    "LogisticRegression": {"clf__C": [0.1, 1, 10]},
    "DecisionTree": {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [3, 8]},
    "RandomForest": {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 3]},
    "KNN(cosine)": {"clf__n_neighbors": [7, 15, 25]},
    "GradientBoosting": {"clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [100, 200]},
}


def pipe(clf):
    return Pipeline([("pre", preprocessor()), ("clf", clf)])


def f1m(a, b):
    return f1_score(a, b, average="macro")


def ece(proba, y, classes, bins=8):
    conf, pred = proba.max(1), np.array(classes)[proba.argmax(1)]
    ok, e, edges = (pred == np.asarray(y)), 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def test_metrics(p, train, test):
    t = time.time(); p = clone(p).fit(train[FEATURES], train["difficulty"]); fit_s = time.time() - t
    t = time.time(); pred = p.predict(test[FEATURES]); inf_ms = (time.time() - t) / max(len(test), 1) * 1000
    y = test["difficulty"]
    r = {"test_accuracy": accuracy_score(y, pred), "test_precision_macro": precision_score(y, pred, average="macro", zero_division=0),
         "test_recall_macro": recall_score(y, pred, average="macro", zero_division=0), "test_f1_macro": f1m(y, pred),
         "test_f1_weighted": f1_score(y, pred, average="weighted"), "test_cost": weighted_cost(y, pred),
         "hard_called_easy": float(((y == "Hard") & (pred == "Easy")).sum() / max((y == "Hard").sum(), 1)),
         "train_s": fit_s, "infer_ms_each": inf_ms}
    proba = None
    if hasattr(p, "predict_proba"):
        proba = p.predict_proba(test[FEATURES])
        try:
            r["test_roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", labels=list(p.classes_))
        except ValueError:
            r["test_roc_auc_ovr"] = np.nan
    return r, p, pred, proba


def main(tune=True):
    df = labelled().sort_values("created_at").reset_index(drop=True)
    if len(df) < 60 or df.difficulty.nunique() < 3:
        raise SystemExit(f"Only {len(df)} labelled issues / {df.difficulty.nunique()} classes: collect more data first.")
    train, test = temporal_split(df)
    folds = group_splits(df)
    X, y = df[FEATURES], df["difficulty"]
    print(f"{len(df)} labelled issues, {df.repo.nunique()} repos | train {len(train)} / test {len(test)} | {len(folds)} grouped folds")
    print("class balance:", df.difficulty.value_counts(normalize=True).round(2).to_dict())

    maj = train["difficulty"].mode()[0]
    rows = {
        "baseline: majority": {"test_f1_macro": f1m(test.difficulty, [maj] * len(test)), "test_accuracy": float((test.difficulty == maj).mean())},
        "baseline: good-first-issue label": {"test_f1_macro": f1m(test.difficulty, np.where(test.is_gfi == 1, "Easy", maj)),
                                             "test_accuracy": float((test.difficulty == np.where(test.is_gfi == 1, "Easy", maj)).mean())},
    }
    fitted = {}
    for name, clf in models().items():
        t0 = time.time()
        cv = cross_validate(pipe(clf), X, y, cv=folds, scoring="f1_macro", return_train_score=True, error_score=np.nan)
        r, p, pred, proba = test_metrics(pipe(clf), train, test)
        r.update(cv_f1_grouped=float(np.nanmean(cv["test_score"])), cv_f1_std=float(np.nanstd(cv["test_score"])),
                 cv_train_f1=float(np.nanmean(cv["train_score"])))
        r["overfit_gap"] = r["cv_train_f1"] - r["cv_f1_grouped"]
        r["f1_95ci"] = str(bootstrap_ci(test.difficulty, pred, f1m, n=500))
        rows[name] = r; fitted[name] = clf
        print(f"  {name:20s} grouped-CV F1 {r['cv_f1_grouped']:.3f}  test F1 {r['test_f1_macro']:.3f}  gap {r['overfit_gap']:.2f}  ({time.time()-t0:.0f}s)")

    best = max(fitted, key=lambda n: np.nan_to_num(rows[n]["cv_f1_grouped"], nan=-1))
    best_clf = fitted[best]
    if tune and best in GRIDS:
        gs = GridSearchCV(pipe(best_clf), GRIDS[best], cv=folds, scoring="f1_macro", n_jobs=1, error_score=np.nan).fit(X, y)
        print(f"  tuned {best}: {gs.best_params_}  grouped-CV F1 {gs.best_score_:.3f}")
        if gs.best_score_ >= rows[best]["cv_f1_grouped"]:
            r, p, pred, proba = test_metrics(gs.best_estimator_, train, test)
            r.update(cv_f1_grouped=float(gs.best_score_), cv_f1_std=np.nan, cv_train_f1=np.nan, overfit_gap=np.nan,
                     f1_95ci=str(bootstrap_ci(test.difficulty, pred, f1m, n=500)), params=str(gs.best_params_))
            rows[best + " (tuned)"] = r
            best_pipe = gs.best_estimator_; best = best + " (tuned)"
        else:
            best_pipe = pipe(best_clf)
    else:
        best_pipe = pipe(best_clf)

    table = write_table(rows, "model_comparison")
    print("\n" + table[[c for c in ["cv_f1_grouped", "test_f1_macro", "test_f1_weighted", "test_roc_auc_ovr", "overfit_gap", "test_cost", "hard_called_easy"] if c in table]].round(3).to_string())

    # calibrated version of the winner, trained on the old data only, for honest test plots + ECE
    cal = CalibratedClassifierCV(clone(best_pipe), method="sigmoid", cv=3).fit(train[FEATURES], train["difficulty"])
    pp = cal.predict_proba(test[FEATURES])
    prior = train["difficulty"].value_counts(normalize=True).reindex(cal.classes_).values
    pc = np.array(cal.classes_)[(pp / prior).argmax(1)]      # prior-corrected rule: stops the model ignoring the rare Hard class
    print("\n" + classification_report(test.difficulty, pc, zero_division=0))
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ConfusionMatrixDisplay.from_predictions(test.difficulty, pc, labels=ORDER, ax=ax, normalize="true", values_format=".2f"); ax.set_title(f"{best}: normalised confusion (test)")
    fig.savefig("report/confusion_difficulty.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, c in enumerate(cal.classes_):
        yy = (test.difficulty.values == c)
        if yy.any() and (~yy).any():
            fp, tp, _ = roc_curve(yy, pp[:, i]); ax.plot(fp, tp, label=f"{c} (AUC {roc_auc_score(yy, pp[:, i]):.2f})")
    ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.legend(); ax.set_title("ROC one-vs-rest (test)")
    fig.savefig("report/roc_difficulty.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    conf, ok = pp.max(1), (pc == test.difficulty.values)
    bins = np.linspace(0, 1, 7); mid, acc = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 3: mid.append(conf[m].mean()); acc.append(ok[m].mean())
    fig, ax = plt.subplots(figsize=(4.5, 4)); ax.plot([0, 1], [0, 1], "k:"); ax.plot(mid, acc, "o-"); ax.set_xlabel("stated confidence"); ax.set_ylabel("actual accuracy"); ax.set_title("Reliability (test)")
    fig.savefig("report/calibration_difficulty.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    ece_v = ece(pp, test.difficulty.values, list(cal.classes_))

    # final deployed model: ALL labelled data, calibrated so predict_proba is trustworthy
    final = CalibratedClassifierCV(clone(best_pipe), method="sigmoid", cv=3).fit(X, y)
    joblib.dump(final, "models/difficulty.joblib")
    meta = {"chosen": best, "chosen_by": "grouped-CV macro-F1 (unseen repos)", "n_train": len(df), "n_repos": int(df.repo.nunique()),
            "label_version": LABEL_VERSION, "easy_q": EASY_Q, "hard_q": HARD_Q, "ece_test": round(ece_v, 3),
            "class_balance": df.difficulty.value_counts(normalize=True).round(3).to_dict(), "eval_plot_rule": "argmax(p / class_prior) for the confusion matrix; the app shows the most probable class plus all probabilities", "trained": pd.Timestamp.now().isoformat()}
    json.dump(meta, open("models/difficulty_meta.json", "w"), indent=2)
    save_metrics("difficulty", {k: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} for k, v in rows.items()})
    save_metrics("difficulty_meta", meta)
    print(f"\nsaved models/difficulty.joblib ({best}); ECE on test = {ece_v:.3f}")
    print("Decide with grouped-CV F1 and its std, NOT with one test number; chance level for 3 balanced classes is about 0.33.")