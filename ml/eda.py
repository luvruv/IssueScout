"""Exploratory data analysis. Writes figures to report/eda/ and a summary to report/eda/summary.md"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from features.build_features import load, add_features, add_labels, NUM

OUT = "report/eda"
ORDER = ["Easy", "Medium", "Hard"]


def _save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=130); plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = load()
    df = add_labels(add_features(raw))
    lab = df[df["difficulty"].notna()].copy()
    S = []                                        # summary lines
    S.append(f"- issues in DB: {len(raw)} (closed {int((raw.state=='closed').sum())}, open {int((raw.state=='open').sum())})")
    S.append(f"- repos: {raw.repo.nunique()}; languages: {raw.drop_duplicates('repo').language.value_counts().to_dict()}")
    S.append(f"- labelled (merged PR) issues used for training: {len(lab)} in {lab.repo.nunique()} repos")
    S.append(f"- duplicate ids: {int(raw.id.duplicated().sum())}; duplicate titles within a repo: {int(raw.duplicated(['repo','title']).sum())}")
    miss = raw[["body", "labels", "first_response_hours", "pr_files", "pr_review_comments"]].isna().mean().round(3)
    S.append(f"- missing share: {miss.to_dict()}")
    cd = lab["difficulty"].value_counts(normalize=True).reindex(ORDER).round(3)
    S.append(f"- class distribution: {cd.to_dict()}  (imbalance ratio max/min = {cd.max()/max(cd.min(),1e-9):.2f})")
    S.append(f"- 'good first issue' share: open {raw[raw.state=='open'].labels.fillna('').str.lower().str.contains('good first issue').mean():.3f}, labelled {lab.is_gfi.mean():.3f}")
    S.append(f"- top repo holds {lab.repo.value_counts(normalize=True).iloc[0]:.1%} of labelled issues (repo skew)")

    plt.figure(figsize=(4.5, 3.5)); sns.countplot(x="difficulty", data=lab, order=ORDER); plt.title("Class distribution"); _save("01_class_distribution")
    top = lab.repo.value_counts().head(15)
    plt.figure(figsize=(7, 4)); top.iloc[::-1].plot.barh(); plt.title("Labelled issues per repo (top 15)"); _save("02_repo_distribution")
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for a, (c, t) in zip(ax.ravel(), [("body_len", "Description length"), ("comments", "Comments (EDA only, not a feature)"),
                                      ("title_len", "Title length"), ("log_stars", "log stars"),
                                      ("repo_age_days", "Repo age (days)"), ("n_code_blocks", "Code blocks")]):
        a.hist(np.log1p(df[c].clip(lower=0)) if c in ("body_len", "comments") else df[c], bins=30); a.set_title(t)
    _save("03_histograms")
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, c in zip(ax, ["body_len", "n_paths", "n_code_blocks", "log_stars"]):
        sns.boxplot(x="difficulty", y=c, data=lab, order=ORDER, ax=a); a.set_title(f"{c} by class")
    _save("04_feature_vs_class_box")
    pr = lab.assign(lines=lab.pr_additions.fillna(0) + lab.pr_deletions.fillna(0))
    plt.figure(figsize=(5, 3.8)); sns.boxplot(x="difficulty", y=np.log1p(pr["lines"]), data=pr, order=ORDER); plt.ylabel("log(lines changed)"); plt.title("Label sanity: PR size per class"); _save("05_label_vs_pr_size")
    corr = lab[NUM + ["effort"]].corr()
    plt.figure(figsize=(10, 8)); sns.heatmap(corr, cmap="coolwarm", center=0, annot=False); plt.title("Correlation (features + effort)"); _save("06_correlation")
    S.append("- features most correlated with effort: " + ", ".join(f"{k} ({v:+.2f})" for k, v in corr["effort"].drop("effort").abs().sort_values(ascending=False).head(5).items()))
    tags = raw.labels.fillna("").str.split("|").explode()
    tags = tags[tags != ""].value_counts().head(15)
    plt.figure(figsize=(7, 4)); tags.iloc[::-1].plot.barh(); plt.title("Most frequent GitHub labels"); _save("07_label_frequency")
    age = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(raw[raw.state == "open"].created_at, utc=True)).dt.days
    plt.figure(figsize=(5, 3.5)); plt.hist(age.clip(upper=2000), bins=40); plt.title("Age of open issues (days)"); _save("08_issue_age")
    rt = raw.first_response_hours.dropna()
    if len(rt):
        plt.figure(figsize=(5, 3.5)); plt.hist(np.log1p(rt), bins=30); plt.title("log(1+hours) to first maintainer reply"); _save("09_response_time")
        S.append(f"- maintainer first reply: median {rt.median():.1f} h, 90th pct {rt.quantile(.9):.0f} h (n={len(rt)})")
    q1, q3 = lab.body_len.quantile([.25, .75]); out = ((lab.body_len > q3 + 3 * (q3 - q1))).sum()
    S.append(f"- extreme body-length outliers (3xIQR): {int(out)}; kept (real issues), log-scaled in plots")
    gfi = lab.groupby("is_gfi")["difficulty"].value_counts(normalize=True).unstack().reindex(columns=ORDER).round(3)
    S.append("- difficulty split by 'good first issue' label:\n" + gfi.to_string())
    # top terms per class
    tf = TfidfVectorizer(max_features=4000, stop_words="english", min_df=3)
    Xt = tf.fit_transform(lab.text); terms = np.array(tf.get_feature_names_out())
    for c in ORDER:
        m = np.asarray(Xt[(lab.difficulty == c).values].mean(0)).ravel(); S.append(f"- top terms {c}: " + ", ".join(terms[m.argsort()[-10:][::-1]]))
    z = TruncatedSVD(2, random_state=0).fit_transform(Xt)
    plt.figure(figsize=(5.5, 4.5))
    for c, col in zip(ORDER, ["tab:green", "tab:orange", "tab:red"]):
        k = (lab.difficulty == c).values; plt.scatter(z[k, 0], z[k, 1], s=10, alpha=.6, c=col, label=c)
    plt.legend(); plt.title("Issue text (TF-IDF -> SVD 2D)"); _save("10_text_svd")
    open(f"{OUT}/summary.md", "w").write("# EDA summary\n\n" + "\n".join(S) + "\n")
    print("\n".join(S)); print(f"\nfigures in {OUT}/")
