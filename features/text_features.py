"""Structural features that exist the moment an issue is opened (no leakage)."""
import re
import pandas as pd

RE_TRACE = re.compile(r"Traceback \(most recent|panic:|goroutine \d+|Exception in|\bat [\w.$]+\([\w.]+:\d+\)|segmentation fault|stack trace", re.I)
RE_PATH = re.compile(r"(?:[\w.\-]+/)+[\w.\-]+\.(?:go|py|js|ts|tsx|rs|java|md|yaml|yml|json|toml|c|cc|cpp|h|sh)\b")
RE_REPRO = re.compile(r"steps to reproduce|to reproduce|expected (?:behavio|result)|actual (?:behavio|result)", re.I)
RE_SMALL = re.compile(r"\b(typo|spelling|grammar|rename|wording|comment|docstring|readme|broken link|dead link|error message)\b", re.I)
RE_BIG = re.compile(r"\b(refactor|redesign|rewrite|architecture|proposal|design|migrate|migration|rfc|epic|umbrella|tracking|roadmap|support for|new feature)\b", re.I)
RE_LINT = re.compile(r"\b(lint|gofmt|format|deprecat|upgrade|bump|version|dependency)\b", re.I)

STRUCT = ["has_stack_trace", "n_paths", "n_checklist", "has_repro", "kw_small", "kw_big", "kw_lint", "n_words"]


def struct_features(title, body):
    t, b = title or "", body or ""
    both = f"{t}\n{b}"
    return {
        "has_stack_trace": int(bool(RE_TRACE.search(both))),
        "n_paths": min(len(set(RE_PATH.findall(both))), 20),
        "n_checklist": b.count("- [ ]") + b.count("- [x]"),
        "has_repro": int(bool(RE_REPRO.search(b))),
        "kw_small": int(bool(RE_SMALL.search(both))),
        "kw_big": int(bool(RE_BIG.search(both))),
        "kw_lint": int(bool(RE_LINT.search(both))),
        "n_words": min(len(b.split()), 3000),
    }


def add_struct(df):
    rows = [struct_features(t, b) for t, b in zip(df["title"], df["body"])]
    extra = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.drop(columns=[c for c in STRUCT if c in df.columns]), extra], axis=1)
