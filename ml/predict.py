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
        raise ValueError("Invalid issue format. Use owner/repo#123 or a github.com issue URL.")
    return m.group(1), int(m.group(2))


def _row_from_db(repo, number):
    try:
        df = load()
        r = df[(df.repo == repo) & (df.number == number)]
        return r.head(1) if len(r) else None
    except Exception:
        return None


def _row_from_api(repo, number):
    from collector.github_client import GitHubClient
    try:
        gh = GitHubClient()
    except Exception as e:
        raise ConnectionError(f"GitHub client initialization failed: {e}. Check your GITHUB_TOKEN in .env.")

    try:
        res_i = gh.get(f"/repos/{repo}/issues/{number}")
        i = res_i.get("body")
        res_rp = gh.get(f"/repos/{repo}")
        rp = res_rp.get("body")
    except Exception as e:
        raise ConnectionError(f"Network error communicating with GitHub API: {e}")

    if not i or not rp:
        raise ValueError(
            f"Issue #{number} or repository '{repo}' was not found. "
            "Please verify the repository name and issue number (or check if it is private/rate-limited)."
        )

    # Check contributing guide with safe exception handling
    has_c = False
    for p in ["CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md"]:
        try:
            b = gh.get(f"/repos/{repo}/contents/{p}").get("body")
            if b:
                has_c = True
                break
        except Exception:
            continue

    labels_str = "|".join(l["name"] for l in i.get("labels", [])) if isinstance(i.get("labels"), list) else ""

    return pd.DataFrame([{
        "id": f"{repo}#{number}", "repo": repo, "number": number, "title": i.get("title", ""), "body": i.get("body") or "",
        "labels": labels_str, "created_at": i.get("created_at"), "closed_at": i.get("closed_at"),
        "state": i.get("state", "open"), "comments": i.get("comments", 0), "has_assignee": int(bool(i.get("assignee"))),
        "author_assoc": i.get("author_association", "NONE"), "has_open_pr": 0, "stars": rp.get("stargazers_count", 0),
        "forks": rp.get("forks_count", 0), "language": rp.get("language") or "Unknown", "contributors": 0,
        "has_contributing": int(has_c), "archived": int(rp.get("archived", False)),
        "repo_created": rp.get("created_at"), "pushed_at": rp.get("pushed_at")
    }])


def similar_issues(text, k=3, exclude_id=None):
    """The k most similar SOLVED issues and how hard they turned out to be (explains the prediction)."""
    try:
        if "tf" not in _SIM:
            lab = labelled()
            if lab.empty or "text" not in lab.columns:
                return []
            tf = TfidfVectorizer(max_features=5000, stop_words="english", min_df=1, ngram_range=(1, 2))
            _SIM.update(df=lab.reset_index(drop=True), tf=tf, X=tf.fit_transform(lab["text"]))
        
        s = (_SIM["X"] @ _SIM["tf"].transform([text or ""]).T).toarray().ravel()
        out = []
        for i in np.argsort(-s):
            r = _SIM["df"].iloc[i]
            if r["id"] == exclude_id or s[i] <= 0:
                continue
            out.append({
                "id": r["id"],
                "title": str(r["title"])[:90],
                "difficulty": r["difficulty"],
                "similarity": round(float(s[i]), 2),
                "url": f"https://github.com/{r['repo']}/issues/{int(r['number'])}"
            })
            if len(out) == k:
                break
        return out
    except Exception:
        return []


def predict_issue(ref, allow_api=True):
    repo, number = parse_ref(ref)
    row = _row_from_db(repo, number)
    src = "Local Database (Offline Cache)"
    
    if row is None:
        if not allow_api:
            raise ValueError(f"Issue {repo}#{number} is not in your local database and online fetching is disabled.")
        row = _row_from_api(repo, number)
        src = "GitHub REST API (Live)"

    if row is None or row.empty:
        raise ValueError(f"Unable to extract features for issue {repo}#{number}.")

    row = add_features(row)
    model_path = "models/difficulty.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run 'python cli.py train all' first.")

    m = joblib.load(model_path)
    
    # Verify expected feature columns exist
    missing_features = [f for f in FEATURES if f not in row.columns]
    for mf in missing_features:
        row[mf] = 0

    proba = m.predict_proba(row[FEATURES])[0]
    classes = list(m.classes_)
    label = classes[int(proba.argmax())]
    conf = float(proba.max())
    
    issue_text = row.iloc[0].get("text", row.iloc[0].get("title", ""))
    similar = similar_issues(issue_text, exclude_id=f"{repo}#{number}")

    return {
        "issue": f"{repo}#{number}",
        "title": row.iloc[0]["title"],
        "source": src,
        "predicted": label,
        "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
        "confidence": round(conf, 3),
        "uncertain": conf < 0.5,
        "similar_solved": similar,
        "note": "Probabilities are calibrated across open-source repositories; treat as a hint, read the issue before committing."
    }
