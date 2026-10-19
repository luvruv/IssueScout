"""Time-to-close regression (log hours).  Calendar time mostly measures WAITING, so this is expected to be hard:
we compare against honest baselines and only call it useful if it beats them."""
import warnings
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features.build_features import load, add_features, add_labels, FEATURES, NUM, CAT
from ml.common import temporal_split, group_splits, save_metrics, write_table

warnings.filterwarnings("ignore")


def pre():
    return ColumnTransformer([("txt", TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english", min_df=2), "text"),
                              ("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def main():
    df = add_labels(add_features(load(state="closed")))
    df = df[(df.pr_merged == 1) & (df.hours_to_close > 0)].sort_values("created_at").reset_index(drop=True)
    df["y"] = np.log1p(df.hours_to_close)
    train, test = temporal_split(df); folds = group_splits(df)
    med = train.y.median(); rmed = train.groupby("repo").y.median()
    base_pred = {"baseline: global median": np.full(len(test), med),
                 "baseline: repo median": test.repo.map(rmed).fillna(med).values}
    rows = {k: {"test_r2": r2_score(test.y, v), "test_mae_logh": mean_absolute_error(test.y, v)} for k, v in base_pred.items()}
    cfg = [("Linear", LinearRegression())] + [(f"Ridge a={a}", Ridge(alpha=a)) for a in [1, 10, 100, 300]] + [(f"Lasso a={a}", Lasso(alpha=a, max_iter=20000)) for a in [0.01, 0.03, 0.1]]
    fits = {}
    for n, reg in cfg:
        p = Pipeline([("pre", pre()), ("reg", reg)]).fit(train[FEATURES], train.y); pr = p.predict(test[FEATURES])
        coef = p.named_steps["reg"].coef_
        rows[n] = {"test_r2": r2_score(test.y, pr), "test_mae_logh": mean_absolute_error(test.y, pr), "test_rmse_logh": mean_squared_error(test.y, pr) ** .5,
                   "cv_r2_grouped": float(np.nanmean(cross_val_score(Pipeline([("pre", pre()), ("reg", reg)]), df[FEATURES], df.y, cv=folds, scoring="r2", error_score=np.nan))),
                   "nonzero_coefs": int((np.abs(coef) > 1e-8).sum()), "coef_l2": float(np.linalg.norm(coef))}
        fits[n] = p
    t = write_table(rows, "time_comparison"); print(t.round(3).to_string())
    ridge = {k: v for k, v in rows.items() if k.startswith("Ridge")}
    best = max(ridge, key=lambda k: ridge[k]["cv_r2_grouped"])
    useful = bool(rows[best]["test_mae_logh"] < rows["baseline: repo median"]["test_mae_logh"] and rows[best]["test_r2"] > 0)
    joblib.dump(fits[best], "models/time.joblib")
    save_metrics("time_to_close", {**{k: {a: round(b, 4) for a, b in v.items()} for k, v in rows.items()}, "best": best, "beats_repo_median": useful})
    ra = [(a, rows[f"Ridge a={a}"]) for a in [1, 10, 100, 300]]; la = [(a, rows[f"Lasso a={a}"]) for a in [0.01, 0.03, 0.1]]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot([a for a, _ in ra], [r["coef_l2"] for _, r in ra], "o-", label="Ridge ||w||"); ax[0].set_xscale("log"); ax[0].set_title("Ridge: coefficients shrink"); ax[0].set_xlabel("alpha")
    ax[1].plot([a for a, _ in la], [r["nonzero_coefs"] for _, r in la], "o-", c="tab:red"); ax[1].set_xscale("log"); ax[1].set_title("Lasso: non-zero coefficients"); ax[1].set_xlabel("alpha")
    fig.tight_layout(); fig.savefig("report/regularisation_effect.png", dpi=140); plt.close(fig)
    print(f"\nbest ridge {best}; beats repo-median baseline: {useful}. "
          + ("" if useful else "NEGATIVE RESULT: report it honestly; the plan uses difficulty-based effort hours instead."))