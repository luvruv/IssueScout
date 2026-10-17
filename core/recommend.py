import json, os
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from core.profile import Profile
from features.build_features import load, add_features, FEATURES
from ai.ontology import missing, skills_to_learn, prereqs, closure
from ai.rules import run_rules
from ai.csp import solve
from ai.bayes import posterior
from llm.extract_skills import extract_skills
from llm.explain import explain

LEVEL = {"beginner": 0, "intermediate": 1, "advanced": 2}
RANK = {"Easy": 0, "Medium": 1, "Hard": 2}
BASE_HOURS = {"Easy": 3, "Medium": 8, "Hard": 20}


def repo_table():
    df = load(state="closed")
    g = df.groupby("repo").agg(n=("pr_merged", "size"), merged=("pr_merged", "sum"),
                               reply_h=("first_response_hours", "median"))
    g["reply_days"] = g["reply_h"] / 24
    prior = float(g["merged"].sum() / max(g["n"].sum(), 1))
    g["merge_rate"] = [posterior(m, n, prior)[0] for m, n in zip(g["merged"], g["n"])]
    return g, prior


def score_open():
    df = load(state="open")
    if df.empty:
        return df
    df = add_features(df)
    stats, prior = repo_table()
    df = df.join(stats[["reply_days", "merge_rate"]], on="repo")
    df["merge_rate"] = df["merge_rate"].fillna(prior)

    X = df[FEATURES]
    m = joblib.load("models/difficulty.joblib")
    proba, classes = m.predict_proba(X), list(m.classes_)
    df["difficulty"] = [classes[i] for i in proba.argmax(1)]
    df["diff_conf"] = proba.max(1)
    df["est_hours"] = proba @ np.array([BASE_HOURS[c] for c in classes])
    df["p_easy"] = proba[:, classes.index("Easy")] if "Easy" in classes else 0.0

    if os.path.exists("models/merge.joblib"):
        pm = joblib.load("models/merge.joblib").predict_proba(X)[:, 1]
        df["p_merge"] = 0.7 * pm + 0.3 * df["merge_rate"]          # blend model with Bayesian repo estimate
    else:
        df["p_merge"] = df["merge_rate"]
    df["p_merge"] = df["p_merge"].clip(0.01, 0.99)

    df["req"] = [extract_skills(t, b, l) for t, b, l in zip(df.title, df.body, df.language)]
    bad = set()
    if os.path.exists("data/processed/repo_flags.csv"):
        fl = pd.read_csv("data/processed/repo_flags.csv")
        bad = set(fl[fl.abandoned].full_name)
    df["abandoned"] = df.repo.isin(bad) | (df.archived == 1)
    return df


def recommend(p: Profile):
    df = score_open()
    if df is None or df.empty:
        return {"plan": [], "note": "No open issues collected yet."}
    budget, lvl = p.hours_per_week * p.weeks, LEVEL[p.level]
    avoid = [a.lower() for a in p.avoid]
    why, cands = Counter(), []
    now = pd.Timestamp.now(tz="UTC")

    for r in df.to_dict("records"):
        text = f"{r['title']} {r['body']}".lower()
        if r["has_assignee"]:                       why["already assigned"] += 1; continue
        if r["abandoned"]:                          why["repo abandoned"] += 1; continue
        if RANK[r["difficulty"]] > lvl:             why["too hard for level"] += 1; continue
        if any(a in text for a in avoid):           why["matches avoid list"] += 1; continue
        if r["est_hours"] > budget:                 why["exceeds time budget"] += 1; continue
        if missing(p.skills, r["req"]):             why["missing skills"] += 1; continue
        rd = None if pd.isna(r["reply_days"]) else float(r["reply_days"])
        facts = {"has_open_pr": bool(r["has_open_pr"]), "reply_days": rd,
                 "help_wanted": bool(r["is_help_wanted"]), "unassigned": True,
                 "has_contributing": bool(r["has_contributing"]),
                 "age_days": (now - pd.to_datetime(r["created_at"], utc=True)).days,
                 "comments": r["comments"]}
        st = run_rules(facts)
        if st["excluded"]:                          why["open PR exists"] += 1; continue
        cands.append({**r, "reply_days": rd, "rules": st["trace"],
                      "score": r["p_merge"] + 0.3 * r["p_easy"] + st["priority"]})

    cands.sort(key=lambda c: -c["score"])
    short = cands[:30]
    if p.use_llm:                                    # LLM re-checks skills on the shortlist only
        keep = []
        for c in short:
            req = extract_skills(c["title"], c["body"], c["language"], use_llm=True)
            if not missing(p.skills, req):
                c["req"] = req; keep.append(c)
        short = keep
    pool = [dict(c, idx=i, hours=c["est_hours"]) for i, c in enumerate(short)]

    chosen, nodes, nodes_plain = None, 0, None
    for kk in range(p.k, 0, -1):
        chosen, nodes = solve(pool, kk, budget)
        if chosen:
            _, nodes_plain = solve(pool, kk, budget, fc=False, mrv=False, max_nodes=50000)
            break
    if not chosen:
        return {"plan": [], "rejected": dict(why), "note": "No feasible plan. Try more hours or more skills."}

    plan = []
    for rank, c in enumerate(chosen, 1):
        item = {
            "rank": rank, "id": c["id"], "repo": c["repo"], "number": int(c["number"]), "title": c["title"],
            "url": f"https://github.com/{c['repo']}/issues/{int(c['number'])}",
            "difficulty": c["difficulty"], "diff_conf": round(float(c["diff_conf"]), 2),
            "p_merge": round(float(c["p_merge"]), 2), "est_hours": round(float(c["est_hours"]), 1),
            "reply_days": None if c["reply_days"] is None else round(c["reply_days"], 1),
            "skills": sorted(c["req"]), "rules": c["rules"], "snippet": c["body"][:800],
            "csp_check": {"skills": True, "time": True, "unclaimed": True, "no_open_pr": True,
                          "max_2_per_repo": True, "not_abandoned": True},
        }
        item["explanation"] = explain(item, p.use_llm)
        plan.append(item)

    out = {"plan": plan, "budget_hours": budget, "planned_hours": round(sum(i["est_hours"] for i in plan), 1),
           "rejected": dict(why), "csp_nodes_fc_mrv": nodes, "csp_nodes_plain": nodes_plain}
    json.dump(out, open("data/last_plan.json", "w"), indent=2)
    return out


def learning_path(p: Profile, target: str):
    """target = 'owner/repo#123'. A* over stepping-stone issues."""
    from ai.astar import astar
    df = score_open()
    repo, num = target.split("#")
    row = df[(df.repo == repo) & (df.number == int(num))]
    if row.empty:
        return {"error": "Target issue is not in the collected open issues."}
    goal = skills_to_learn(p.skills, row.iloc[0]["req"])
    if not goal:
        return {"ready": True, "note": "You already have the skills for this issue."}
    steps = []
    for r in df.to_dict("records"):
        if r["has_assignee"] or r["has_open_pr"] or r["abandoned"] or r["difficulty"] == "Hard":
            continue
        teaches = set(r["req"]) & goal
        if not teaches:
            continue
        needs = set().union(*[prereqs(s) for s in teaches]) - teaches
        steps.append({"id": r["id"], "title": r["title"], "teaches": frozenset(teaches),
                      "needs": frozenset(needs), "hours": max(float(r["est_hours"]), 1.0)})
    steps = sorted(steps, key=lambda s: s["hours"])[:60]
    if not steps:
        return {"error": "No stepping-stone issues found for the missing skills."}
    path, hours, expanded = astar(closure(p.skills), goal, steps)
    if path is None:
        return {"error": "No path found with the collected issues.", "goal": sorted(goal)}
    return {"goal": sorted(goal), "expanded_nodes": expanded, "total_hours": round(hours, 1),
            "path": [{"id": s["id"], "title": s["title"], "teaches": sorted(s["teaches"]), "hours": round(s["hours"], 1)} for s in path]}