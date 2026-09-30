import numpy as np
import pandas as pd
from collector.db import conn

NUM = ["body_len", "title_len", "n_labels", "is_gfi", "is_help_wanted", "is_bug", "is_docs",
       "n_code_blocks", "has_link", "log_stars", "log_contributors", "has_contributing",
       "repo_age_days"]
CAT = ["language", "author_assoc"]
FEATURES = NUM + CAT + ["text"]
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
    df[NUM] = df[NUM].fillna(0)
    return df


def add_labels(df):
    """Labels come from the RESOLVED PR. Use them only as targets, never as inputs."""
    df = df.copy()
    lines = df["pr_additions"].fillna(0) + df["pr_deletions"].fillna(0)
    files = df["pr_files"].fillna(0)
    df["difficulty"] = np.where((lines <= 30) & (files <= 2), "Easy",
                        np.where((lines > 200) | (files > 6), "Hard", "Medium"))
    df["hours_to_close"] = (pd.to_datetime(df["closed_at"], utc=True)
                            - pd.to_datetime(df["created_at"], utc=True)).dt.total_seconds() / 3600
    return df


def main():
    df = add_labels(add_features(load(state="closed")))
    df = df[df.pr_merged == 1]
    print("Difficulty distribution:\n", df["difficulty"].value_counts(normalize=True).round(3))
    print("Tune the thresholds in add_labels() if one class is under 15%.")
    sample = df.sample(min(200, len(df)), random_state=1)[["id", "repo", "number", "title", "difficulty"]].copy()
    sample["url"] = "https://github.com/" + sample["repo"] + "/issues/" + sample["number"].astype(str)
    sample["my_difficulty"] = ""
    sample.to_csv("data/processed/manual_label_sample.csv", index=False)
    print("Wrote data/processed/manual_label_sample.csv. Open it, read each issue, and fill my_difficulty with Easy/Medium/Hard.")