import hashlib, json, os, pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from features.build_features import NUM, CAT, FEATURES  # noqa: F401


class RepoPrior(BaseEstimator, TransformerMixin):
    """Smoothed per-repo class rates learned on TRAIN data only.
    Training rows use leave-one-out so a row never sees its own label. Unseen repos get the global prior."""
    def __init__(self, strength=8):
        self.strength = strength

    def _fit(self, X, y):
        repo, y = np.asarray(X).ravel().astype(str), np.asarray(y)
        self.classes_ = sorted(set(y.tolist()))
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        self.counts_ = {r: np.array([((y == c) & (repo == r)).sum() for c in self.classes_], float) for r in set(repo)}
        return repo, y

    def _enc(self, counts):
        return (counts + self.strength * self.prior_) / (counts.sum() + self.strength)

    def fit(self, X, y=None):
        self._fit(X, y); return self

    def fit_transform(self, X, y=None, **kw):
        repo, y = self._fit(X, y)
        out = np.zeros((len(repo), len(self.classes_)))
        for i, (r, yi) in enumerate(zip(repo, y)):
            c = self.counts_[r].copy(); c[self.classes_.index(yi)] -= 1
            out[i] = self._enc(c)
        return out

    def transform(self, X):
        repo = np.asarray(X).ravel().astype(str)
        return np.array([self._enc(self.counts_[r]) if r in self.counts_ else self.prior_ for r in repo])


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """OPTIONAL (pip install sentence-transformers; set USE_EMBEDDINGS=1). Embeddings are cached on disk."""
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        global _ST
        texts = [str(t)[:1500] for t in np.asarray(X).ravel()]
        path = "data/emb_cache.pkl"
        cache = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        miss = [i for i, k in enumerate(keys) if k not in cache]
        if miss:
            from sentence_transformers import SentenceTransformer
            if "_ST" not in globals():
                _ST = SentenceTransformer(self.model)
            vecs = _ST.encode([texts[i] for i in miss], batch_size=32, show_progress_bar=False)
            for i, v in zip(miss, vecs):
                cache[keys[i]] = v.astype("float32")
            pickle.dump(cache, open(path, "wb"))
        return np.vstack([cache[k] for k in keys])


def preprocessor(use_emb=None):
    use_emb = (os.getenv("USE_EMBEDDINGS") == "1") if use_emb is None else use_emb
    parts = [
        ("txt", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english", min_df=2), "text"),
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("repo", RepoPrior(), "repo"),
    ]
    if use_emb:
        parts.append(("emb", SentenceEmbedder(), "text"))
    return ColumnTransformer(parts)


def temporal_split(df, frac=0.8):
    df = df.sort_values("created_at")
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]


def group_splits(df, n=5):
    """Folds that keep every repo entirely in train OR test => measures generalisation to UNSEEN repos."""
    g = df["repo"].values
    k = max(2, min(n, len(set(g))))
    return list(GroupKFold(n_splits=k).split(df, groups=g))


def bootstrap_ci(y_true, y_pred, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(fn(y_true[idx], y_pred[idx]))
    return [round(float(x), 3) for x in np.percentile(scores, [2.5, 97.5])]


# cost of predicting `pred` when the truth is `true`: calling a Hard issue Easy is the worst mistake
COST = {("Hard", "Easy"): 3, ("Hard", "Medium"): 1, ("Medium", "Easy"): 1,
        ("Easy", "Hard"): 2, ("Easy", "Medium"): 1, ("Medium", "Hard"): 1}


def weighted_cost(y_true, y_pred):
    return float(np.mean([COST.get((t, p), 0) for t, p in zip(y_true, y_pred)]))


def save_metrics(name, d):
    os.makedirs("report", exist_ok=True)
    path = "report/metrics.json"
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[name] = d
    json.dump(data, open(path, "w"), indent=2, default=str)


def write_table(rows, stem):
    """rows: {model: {metric: value}} -> report/<stem>.csv and .md"""
    os.makedirs("report", exist_ok=True)
    t = pd.DataFrame(rows).T
    t.to_csv(f"report/{stem}.csv")
    cols = ["model"] + list(t.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for k, r in t.iterrows():
        md.append("| " + " | ".join([str(k)] + [("" if pd.isna(v) else f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r]) + " |")
    open(f"report/{stem}.md", "w").write("\n".join(md) + "\n")
    return t