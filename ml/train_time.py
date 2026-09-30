import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score
from features.build_features import load, add_features, add_labels
from ml.common import FEATURES, preprocessor, temporal_split, save_metrics


def main():
    df = add_labels(add_features(load(state="closed")))
    df = df[(df.pr_merged == 1) & (df.hours_to_close > 0)].copy()
    df["y"] = np.log1p(df.hours_to_close)
    train, test = temporal_split(df)
    res = {}
    configs = [("Linear", LinearRegression())]
    configs += [(f"Ridge a={a}", Ridge(alpha=a)) for a in [0.1, 1, 10, 100]]
    configs += [(f"Lasso a={a}", Lasso(alpha=a, max_iter=10000)) for a in [0.001, 0.01, 0.1]]
    best, best_r2 = None, -9
    for name, reg in configs:
        pipe = Pipeline([("pre", preprocessor()), ("reg", reg)]).fit(train[FEATURES], train["y"])
        r2 = r2_score(test["y"], pipe.predict(test[FEATURES]))
        coef = pipe.named_steps["reg"].coef_
        res[name] = {"test_r2": round(r2, 3), "nonzero_coefs": int((np.abs(coef) > 1e-8).sum()),
                     "coef_l2": round(float(np.linalg.norm(coef)), 2)}
        print(name, res[name])
        if r2 > best_r2 and name.startswith("Ridge"):
            best, best_r2 = pipe, r2
    save_metrics("time_to_close", res)
    joblib.dump(best, "models/time.joblib")