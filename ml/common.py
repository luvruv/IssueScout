import json, os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from features.build_features import NUM, CAT, FEATURES  # noqa: F401


def preprocessor():
    return ColumnTransformer([
        ("txt", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english", min_df=2), "text"),
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])


def temporal_split(df, frac=0.8):
    df = df.sort_values("created_at")
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]


def bootstrap_ci(y_true, y_pred, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(fn(y_true[idx], y_pred[idx]))
    return [round(float(x), 3) for x in np.percentile(scores, [2.5, 97.5])]


def save_metrics(name, d):
    os.makedirs("report", exist_ok=True)
    path = "report/metrics.json"
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[name] = d
    json.dump(data, open(path, "w"), indent=2)