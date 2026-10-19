"""Outcome tracking: the real success metric is 'did a recommended issue become a merged PR?'"""
import json, os
from datetime import datetime

PATH = "data/outcomes.jsonl"
STATUSES = ["started", "pr_opened", "merged", "closed_unmerged", "abandoned", "too_hard"]


def add(issue_id, status, hours=None, notes=""):
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    os.makedirs("data", exist_ok=True)
    rec = {"issue_id": issue_id, "status": status, "hours": hours, "notes": notes, "at": datetime.utcnow().isoformat()}
    with open(PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load():
    return [json.loads(l) for l in open(PATH)] if os.path.exists(PATH) else []


def latest():
    d = {}
    for r in load():
        d[r["issue_id"]] = r
    return d


def summary():
    L = latest(); n = len(L)
    cnt = {s: sum(1 for r in L.values() if r["status"] == s) for s in STATUSES}
    hrs = [r["hours"] for r in L.values() if r.get("hours")]
    return {"issues_tried": n, **cnt, "merge_rate_of_tried": round(cnt["merged"] / n, 2) if n else None,
            "avg_hours_logged": round(sum(hrs) / len(hrs), 1) if hrs else None}
