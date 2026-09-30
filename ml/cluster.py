import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from collector.db import conn


def main(eps=1.5):
    r = pd.read_sql("SELECT * FROM repos", conn())
    now = pd.Timestamp.now(tz="UTC")
    r["days_since_push"] = (now - pd.to_datetime(r.pushed_at, utc=True)).dt.days
    r["age_days"] = (now - pd.to_datetime(r.created_at, utc=True)).dt.days
    F = pd.DataFrame({
        "log_stars": np.log1p(r.stars), "log_forks": np.log1p(r.forks),
        "log_contrib": np.log1p(r.contributors), "log_open": np.log1p(r.open_issues),
        "log_push": np.log1p(r.days_since_push), "age_years": r.age_days / 365})
    Z = StandardScaler().fit_transform(F)

    r["cluster"] = KMeans(4, n_init=10, random_state=42).fit_predict(Z)

    d, _ = NearestNeighbors(n_neighbors=4).fit(Z).kneighbors(Z)      # k-distance plot -> pick eps
    plt.figure(); plt.plot(np.sort(d[:, -1])); plt.title("k-distance (choose eps at the elbow)")
    plt.savefig("report/kdistance.png", dpi=150)
    r["dbscan"] = DBSCAN(eps=eps, min_samples=4).fit_predict(Z)

    iso = IsolationForest(contamination=0.08, random_state=42).fit(Z)
    r["anomaly"] = iso.predict(Z) == -1
    r["abandoned"] = (r.archived == 1) | (r.days_since_push > 365) | (r.anomaly & (r.days_since_push > 180))
    print(pd.crosstab(r.archived, r.anomaly, rownames=["archived"], colnames=["anomaly"]))  # sanity check

    pcs = PCA(2).fit_transform(Z)
    r["pc1"], r["pc2"] = pcs[:, 0], pcs[:, 1]
    plt.figure(figsize=(7, 5))
    plt.scatter(r.pc1, r.pc2, c=r.cluster, cmap="tab10")
    plt.title("Repos in PCA space, coloured by K-Means cluster")
    plt.savefig("report/pca_repos.png", dpi=150)

    r[["full_name", "cluster", "dbscan", "anomaly", "abandoned", "pc1", "pc2", "days_since_push"]] \
        .to_csv("data/processed/repo_flags.csv", index=False)
    print("saved data/processed/repo_flags.csv")