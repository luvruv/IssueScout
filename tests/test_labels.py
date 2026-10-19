import numpy as np, pandas as pd
from features.labels import assign_labels
from features.text_features import struct_features


def df():
    r = np.random.default_rng(0); n = 120
    return pd.DataFrame({"repo": ["a"] * 60 + ["b"] * 60, "pr_additions": r.integers(1, 400, n), "pr_deletions": r.integers(0, 100, n),
                         "pr_files": r.integers(1, 12, n), "pr_review_comments": r.integers(0, 10, n)})


def test_label_shares_and_order():
    d = assign_labels(df()); s = d.difficulty.value_counts(normalize=True)
    assert abs(s["Easy"] - 0.4) < 0.1 and abs(s["Hard"] - 0.2) < 0.1
    assert d[d.difficulty == "Easy"].effort.max() <= d[d.difficulty == "Hard"].effort.min()


def test_guards():
    d = df(); d.loc[0, ["pr_additions", "pr_deletions", "pr_files"]] = [2, 1, 1]; d.loc[1, ["pr_additions", "pr_deletions", "pr_files"]] = [3000, 10, 50]
    d = assign_labels(d); assert d.difficulty[0] == "Easy" and d.difficulty[1] == "Hard"


def test_label_does_not_use_issue_text():
    d = df(); a = assign_labels(d).difficulty.tolist(); d["title"] = "x"; assert assign_labels(d).difficulty.tolist() == a


def test_struct_features_are_creation_time_only():
    f = struct_features("Fix typo in README", "see docs/guide/setup.md")
    assert f["kw_small"] == 1 and f["n_paths"] == 1 and f["has_stack_trace"] == 0
    assert struct_features("crash", "Traceback (most recent call last)")["has_stack_trace"] == 1
