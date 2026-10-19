"""LLM evaluation (syllabus: evaluation metrics and benchmarks): JSON validity, agreement with keywords, guardrail pass rate, latency."""
import json, os, random, time
from collector.db import conn
from llm.client import provider, complete, last_error
from llm.extract_skills import parse_skills, extract_skills_meta
from llm.explain import explain_checked
from ai.ontology import extract_keywords, G
from ml.common import save_metrics


def main(n=30):
    if provider() == "none":
        raise SystemExit("Set LLM_PROVIDER in .env (openai / anthropic / mock) first.")
    rows = conn().execute("SELECT title, body, (SELECT language FROM repos WHERE full_name=repo) FROM issues WHERE state='open'").fetchall()
    random.Random(0).shuffle(rows); rows = rows[:n]
    valid = jac = fell = 0; lat = []
    for t, b, lang in rows:
        kw = extract_keywords(f"{t} {b}", lang); t0 = time.time()
        prompt = ('Return ONLY JSON like {"skills": ["go"]}. Choose the skills needed to solve this GitHub issue '
                  f"from this list only: {sorted(G.nodes)}\nTitle: {t}\nBody: {(b or '')[:1500]}")
        out = complete(prompt, 200); lat.append(time.time() - t0)
        try:
            got = parse_skills(out); valid += 1
            jac += len(got & kw) / max(len(got | kw), 1)
        except Exception:
            fell += 1
    res = {"issues": len(rows), "provider": provider(), "model": os.getenv("LLM_MODEL"), "json_valid_rate": round(valid / max(len(rows), 1), 3),
           "mean_jaccard_vs_keywords (agreement, not accuracy)": round(jac / max(valid, 1), 3), "mean_latency_s": round(sum(lat) / max(len(lat), 1), 2),
           "last_error": last_error()}
    if os.path.exists("data/last_plan.json"):
        plan = json.load(open("data/last_plan.json"))["plan"]
        metas = [explain_checked(i, True)[1] for i in plan]
        res["explanations"] = {"n": len(plan), "llm_passed_guardrail": sum(m["source"] == "llm" for m in metas),
                               "fell_back_to_template": sum(m["source"] == "template" for m in metas), "reasons": [m.get("reason") for m in metas if m.get("reason")]}
    print(json.dumps(res, indent=2)); save_metrics("llm_eval", res)
    return res
