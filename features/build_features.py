import os
import numpy as np
import pandas as pd
from collector.db import conn
from features.text_features import add_struct, STRUCT
from features.labels import assign_labels

NUM = ["body_len", "title_len", "n_labels", "is_gfi", "is_help_wanted", "is_bug", "is_docs",
       "n_code_blocks", "has_link", "log_stars", "log_contributors", "has_contributing",
       "repo_age_days"] + STRUCT
CAT = ["language", "author_assoc"]
FEATURES = NUM + CAT + ["text", "repo"]
# LEAKAGE RULE: 'comments', 'has_assignee' and every pr_* column are NOT features.


def load(state=None):
    q = """SELECT i.*, r.stars, r.forks, r.language, r.contributors, r.has_contributing,
                  r.archived, r.created_at AS repo_created, r.pushed_at
           FROM issues i JOIN repos r ON i.repo = r.full_name"""
    if state:
        q += f" WHERE i.state = '{state}'"
    return pd.read_sql(q, conn())


def add_features(df):
    df = df.copy()
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    lab = df["labels"].fillna("").str.lower()
    df["text"] = df["title"] + " . " + df["body"].str.slice(0, 2000)
    df["body_len"] = df["body"].str.len()
    df["title_len"] = df["title"].str.len()
    df["n_labels"] = lab.apply(lambda s: len([x for x in s.split("|") if x]))
    df["is_gfi"] = lab.str.contains("good first issue|good-first-issue|beginner|easy").astype(int)
    df["is_help_wanted"] = lab.str.contains("help wanted|help-wanted").astype(int)
    df["is_bug"] = lab.str.contains("bug").astype(int)
    df["is_docs"] = lab.str.contains("doc").astype(int)
    df["n_code_blocks"] = df["body"].str.count("```") // 2
    df["has_link"] = df["body"].str.contains("http").astype(int)
    df["log_stars"] = np.log1p(df["stars"].fillna(0))
    df["log_contributors"] = np.log1p(df["contributors"].fillna(0))
    df["has_contributing"] = df["has_contributing"].fillna(0)
    df["repo_age_days"] = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(df["repo_created"], utc=True)).dt.days
    df["language"] = df["language"].fillna("unknown")
    df["author_assoc"] = df["author_assoc"].fillna("NONE")
    df = add_struct(df)
    df[NUM] = df[NUM].fillna(0)
    return df


def add_labels(df):
    """Labels come from the RESOLVED PR (merged only). Targets only, never inputs."""
    df = df.copy()
    df["hours_to_close"] = (pd.to_datetime(df["closed_at"], utc=True)
                            - pd.to_datetime(df["created_at"], utc=True)).dt.total_seconds() / 3600
    df["difficulty"] = pd.Series([None] * len(df), index=df.index, dtype=object)   # object dtype: pandas 3 is strict
    df["effort"] = np.nan
    m = (df["pr_merged"] == 1) & df["pr_files"].notna()
    if m.sum() >= 10:
        lab = assign_labels(df[m])
        df.loc[m, "difficulty"] = lab["difficulty"]
        df.loc[m, "effort"] = lab["effort"]
    return df


def labelled():
    """The training table: closed issues whose merged PR gives a difficulty label."""
    df = add_labels(add_features(load(state="closed")))
    return df[df["difficulty"].notna()].copy()


def main(n_manual=200):
    df = labelled()
    print(f"Labelled issues: {len(df)} from {df.repo.nunique()} repos")
    print("Difficulty distribution:\n", df["difficulty"].value_counts(normalize=True).round(3))
    path = "data/processed/manual_label_sample.csv"
    os.makedirs("data/processed", exist_ok=True)
    keep = pd.DataFrame()
    if os.path.exists(path):                         # never throw away labels you already typed
        old = pd.read_csv(path)
        keep = old[old["my_difficulty"].astype(str).str.strip().isin(["Easy", "Medium", "Hard"])]
    rest = df[~df["id"].isin(keep["id"] if len(keep) else [])]
    need = max(0, n_manual - len(keep))
    new = rest.sample(min(need, len(rest)), random_state=1)[["id", "repo", "number", "title", "difficulty"]].copy()
    new["url"] = "https://github.com/" + new["repo"] + "/issues/" + new["number"].astype(str)
    new["my_difficulty"] = ""
    if len(keep):
        k = keep[["id", "my_difficulty"]].merge(df[["id", "repo", "number", "title", "difficulty"]], on="id")
        k["url"] = "https://github.com/" + k["repo"] + "/issues/" + k["number"].astype(str)
        new = pd.concat([k[new.columns], new])
    new.to_csv(path, index=False)
    print(f"Wrote {path}: {len(new)} rows, {int((new.my_difficulty.astype(str).str.strip() != '').sum())} already labelled by you.")
    print("Label them with:  python cli.py label")