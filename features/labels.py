"""
Difficulty label = how much EFFORT the resolving pull request took, relative to its own repository.

effort components (all from the merged PR):  log(lines changed), files changed, review comments
1. each component -> percentile rank INSIDE its repo (shrunk toward the global rank for small repos)
2. effort = mean of the available components
3. Easy = lowest EASY_Q share, Hard = top (1-HARD_Q) share, Medium = the rest
4. guards: a 1-file / <=5-line PR is always Easy, a >1500-line / >40-file PR is always Hard

These are used ONLY as the target y. They are never model inputs. Freeze EASY_Q/HARD_Q after EDA.
"""
import numpy as np
import pandas as pd

EASY_Q, HARD_Q = 0.40, 0.80
SHRINK = 10 ** 9          # a repo with n labelled issues gets weight n/(n+SHRINK) on its own ranking
LABEL_VERSION = "effort-absolute-v2"


def _pct(df, col):
    g = df[col].groupby(df["repo"])
    within = g.rank(pct=True, method="average")
    glob = df[col].rank(pct=True, method="average")
    n = g.transform("count")
    w = n / (n + SHRINK)
    return w * within + (1 - w) * glob


def effort_score(df):
    comp = pd.DataFrame(index=df.index)
    comp["lines"] = np.log1p(df["pr_additions"].fillna(0) + df["pr_deletions"].fillna(0))
    comp["files"] = df["pr_files"].astype(float)
    if "pr_review_comments" in df and df["pr_review_comments"].notna().mean() > 0.5:
        comp["review"] = df["pr_review_comments"].astype(float)
    work = comp.copy()
    work["repo"] = df["repo"].values
    ranks = pd.DataFrame({c: _pct(work, c) for c in comp.columns})
    return ranks.mean(axis=1, skipna=True)


def assign_labels(df, easy_q=EASY_Q, hard_q=HARD_Q):
    """df must contain merged-PR rows only. Returns df with 'effort' and 'difficulty'."""
    df = df.copy()
    df["effort"] = effort_score(df)
    lo, hi = df["effort"].quantile(easy_q), df["effort"].quantile(hard_q)
    lab = np.where(df["effort"] <= lo, "Easy", np.where(df["effort"] > hi, "Hard", "Medium"))
    lines = df["pr_additions"].fillna(0) + df["pr_deletions"].fillna(0)
    files = df["pr_files"].fillna(0)
    lab = np.where((lines <= 5) & (files <= 1), "Easy", lab)
    lab = np.where((lines > 1500) | (files > 40), "Hard", lab)
    df["difficulty"] = lab
    return df
