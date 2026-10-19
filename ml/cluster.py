"""Repo clustering (K-Means, DBSCAN, BIRCH), anomaly detection (Isolation Forest), PCA and kernel PCA.
Cluster MEANINGS are not assumed: read report/cluster_profiles.csv and name the clusters yourself."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import Birch, DBSCAN, KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from collector.db import conn


def main(eps=None):
    import os; os.makedirs("report", exist_ok=True); os.makedirs("data/processed", exist_ok=True)
    c = conn()
    r = pd.read_sql("SELECT * FROM repos", c)
    iss = pd.read_sql("SELECT repo, state, pr_merged, first_response_hours FROM issues", c)
    st = iss.groupby("repo").agg(median_reply_h=("first_response_hours", "median"),
                                 merge_rate=("pr_merged", "mean"), n_issues=("state", "size"))
    r = r.join(st, on="full_name")
    now = pd.Timestamp.now(tz="UTC")
    r["days_since_push"] = (now - pd.to_datetime(r.pushed_at, utc=True)).dt.days
    r["age_years"] = (now - pd.to_datetime(r.created_at, utc=True)).dt.days / 365
    F = pd.DataFrame({"log_stars": np.log1p(r.stars), "log_forks": np.log1p(r.forks), "log_contrib": np.log1p(r.contributors),
                      "log_open": np.log1p(r.open_issues), "log_push": np.log1p(r.days_since_push), "age_years": r.age_years,
                      "log_reply_h": np.log1p(r.median_reply_h.fillna(r.median_reply_h.median())),
                      "merge_rate": r.merge_rate.fillna(r.merge_rate.median()), "has_contributing": r.has_contributing.fillna(0)})
    Z = StandardScaler().fit_transform(F)
    ks = range(2, min(8, len(r) - 1))
    sil = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=42).fit_predict(Z)) for k in ks}
    k = max(sil, key=sil.get)
    print("silhouette by k:", {a: round(b, 3) for a, b in sil.items()}, "-> k =", k)
    r["cluster"] = KMeans(k, n_init=10, random_state=42).fit_predict(Z)
    r["birch"] = Birch(n_clusters=k).fit_predict(Z)
    kd = np.sort(NearestNeighbors(n_neighbors=4).fit(Z).kneighbors(Z)[0][:, -1])
    eps = eps or float(np.percentile(kd, 90))
    r["dbscan"] = DBSCAN(eps=eps, min_samples=4).fit_predict(Z)
    print(f"DBSCAN eps={eps:.2f} (90th pct of 4-NN distance): clusters={len(set(r.dbscan)) - (-1 in set(r.dbscan))}, noise={int((r.dbscan == -1).sum())}")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(kd); ax[0].axhline(eps, c="r", ls=":"); ax[0].set_title("k-distance (eps at the elbow)")
    ax[1].plot(list(sil), list(sil.values()), "o-"); ax[1].set_title("silhouette vs k"); fig.tight_layout(); fig.savefig("report/kdistance.png", dpi=140); plt.close(fig)

    iso = IsolationForest(contamination=0.1, random_state=42).fit(Z)
    r["anomaly_score"] = -iso.score_samples(Z)
    r["anomaly"] = iso.predict(Z) == -1
    r["abandoned"] = (r.archived == 1) | (r.days_since_push > 365) | (r.anomaly & (r.days_since_push > 180))
    print(pd.crosstab(r.archived, r.anomaly, rownames=["archived"], colnames=["anomaly"]), "\nabandoned flagged:", int(r.abandoned.sum()))

    p = PCA(min(3, Z.shape[1])).fit(Z); pcs = p.transform(Z)
    print("PCA explained variance:", np.round(p.explained_variance_ratio_, 3))
    kp = KernelPCA(2, kernel="rbf", gamma=0.1).fit_transform(Z)
    r["pc1"], r["pc2"] = pcs[:, 0], pcs[:, 1]; r["kpc1"], r["kpc2"] = kp[:, 0], kp[:, 1]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (x, y, t, col) in zip(ax, [("pc1", "pc2", "PCA + K-Means", "cluster"), ("pc1", "pc2", "PCA + DBSCAN (-1 = noise)", "dbscan"), ("kpc1", "kpc2", "Kernel PCA (rbf) + BIRCH", "birch")]):
        a.scatter(r[x], r[y], c=r[col], cmap="tab10", s=40)
        for _, row in r[r.anomaly].iterrows(): a.annotate(row.full_name.split("/")[1], (row[x], row[y]), fontsize=7)
        a.set_title(t)
    fig.tight_layout(); fig.savefig("report/pca_repos.png", dpi=140); plt.close(fig)
    prof = pd.concat([F, r[["cluster"]]], axis=1).groupby("cluster").mean().round(2); prof["n_repos"] = r.cluster.value_counts().sort_index()
    prof.to_csv("report/cluster_profiles.csv"); print("\ncluster profiles (name the clusters yourself from this):\n", prof.to_string())
    r[["full_name", "cluster", "birch", "dbscan", "anomaly", "anomaly_score", "abandoned", "pc1", "pc2", "kpc1", "kpc2", "days_since_push"]].to_csv("data/processed/repo_flags.csv", index=False)
    print("saved data/processed/repo_flags.csv")