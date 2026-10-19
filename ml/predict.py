"""Predict the difficulty of ANY GitHub issue:  python cli.py predict owner/repo#123   (or the issue URL)"""
import json, os, re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from features.build_features import load, add_features, FEATURES, labelled

_SIM = {}


def parse_ref(ref):
    m = re.search(r"github\.com/([\w.\-]+/[\w.\-]+)/issues/(\d+)", ref) or re.match(r"^([\w.\-]+/[\w.\-]+)#(\d+)$", ref.strip())
    if not m:
        raise ValueError("use owner/repo#123 or a github.com issue URL")
    return m.group(1), int(m.group(2))


def _row_from_db(repo, number):
    df = load()
    r = df[(df.repo == repo) & (df.number == number)]
    return r.head(1) if len(r) else None


def _row_from_api(repo, number):
    from collector.github_client import GitHubClient
    gh = GitHubClient()
    i = gh.get(f"/repos/{repo}/issues/{number}")["body"]; rp = gh.get(f"/repos/{repo}")["body"]
    if not i or not rp:
        raise ValueError("issue or repo not found (private, or wrong name)")
    has_c = any(gh.get(f"/repos/{repo}/contents/{p}")["body"] for p in ["CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md"])
    return pd.DataFrame([{
        "id": f"{repo}#{number}", "repo": repo, "number": number, "title": i["title"], "body": i.get("body") or "",
        "labels": "|".join(l["name"] for l in i["labels"]), "created_at": i["created_at"], "closed_at": i.get("closed_at"),
        "state": i["state"], "comments": i["comments"], "has_assignee": int(bool(i.get("assignee"))), "author_assoc": i.get("author_association"),
        "has_open_pr": 0, "stars": rp["stargazers_count"], "forks": rp["forks_count"], "language": rp["language"], "contributors": 0,
        "has_contributing": int(has_c), "archived": int(rp["archived"]), "repo_created": rp["created_at"], "pushed_at": rp["pushed_at"]}])


def similar_issues(text, k=3, exclude_id=None):
    """The k most similar SOLVED issues and how hard they turned out to be (explains the prediction)."""
    if "tf" not in _SIM:
        lab = labelled(); tf = TfidfVectorizer(max_features=5000, stop_words="english", min_df=1, ngram_range=(1, 2))
        _SIM.update(df=lab.reset_index(drop=True), tf=tf, X=tf.fit_transform(lab["text"]))
    s = (_SIM["X"] @ _SIM["tf"].transform([text]).T).toarray().ravel()
    out = []
    for i in np.argsort(-s):
        r = _SIM["df"].iloc[i]
        if r["id"] == exclude_id or s[i] <= 0:
            continue
        out.append({"id": r["id"], "title": r["title"][:90], "difficulty": r["difficulty"], "similarity": round(float(s[i]), 2),
                    "url": f"https://github.com/{r['repo']}/issues/{int(r['number'])}"})
        if len(out) == k:
            break
    return out


def predict_issue(ref, allow_api=True):
    repo, number = parse_ref(ref)
    row = _row_from_db(repo, number)
    src = "database"
    if row is None:
        if not allow_api:
            raise ValueError("issue not in the database")
        row, src = _row_from_api(repo, number), "GitHub API"
    row = add_features(row)
    m = joblib.load("models/difficulty.joblib")
    proba = m.predict_proba(row[FEATURES])[0]; classes = list(m.classes_)
    label = classes[int(proba.argmax())]                      # most probable class, consistent with the printed probabilities
    conf = float(proba.max())
    return {"issue": f"{repo}#{number}", "title": row.iloc[0]["title"], "source": src, "predicted": label,
            "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)}, "confidence": round(conf, 3),
            "uncertain": conf < 0.5, "similar_solved": similar_issues(row.iloc[0]["text"], exclude_id=f"{repo}#{number}"),
            "note": "Probabilities are calibrated, but the model is weak when trained on few repos: treat as a hint, read the issue."}
