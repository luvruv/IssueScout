"""Grounded explanations. The LLM may only restate model/rule outputs; a post-check rejects invented numbers."""
import json, re
from llm.client import complete

FACT_KEYS = ["repo", "number", "title", "difficulty", "diff_conf", "p_merge", "est_hours", "reply_days", "skills", "rules"]


def template(item):
    s = (f"Predicted {item['difficulty']} ({item['diff_conf']:.0%} confidence), about {item['est_hours']}h, "
         f"merge probability {item['p_merge']:.0%}. Needs: {', '.join(item['skills']) or 'no special skills'}. "
         f"Rules: {'; '.join(item['rules']) or 'none fired'}.")
    if item.get("uncertain"):
        s += " The model is UNSURE about this one: read the issue yourself before starting."
    return s


def _allowed(item):
    nums = {float(x) for x in [item["diff_conf"] * 100, item["p_merge"] * 100, item["est_hours"], item["number"]] if x is not None}
    nums |= {round(v) for v in list(nums)}
    if item.get("reply_days") is not None:
        nums.add(float(item["reply_days"]))
    for k in ("Easy", "Medium", "Hard"):
        if item.get("probs"):
            nums.add(round(item["probs"][k] * 100))
    text = f"{item['title']} {item.get('snippet', '')} {item['repo']}"
    nums |= {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)}
    return nums


def check(text, item):
    """Every number the LLM wrote must come from the facts (small numbers 0-5 are allowed for step lists)."""
    ok = _allowed(item)
    bad = []
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        v = float(tok)
        if v <= 5 or v in ok or round(v) in ok or any(abs(v - a) < 0.051 for a in ok):
            continue
        bad.append(tok)
    return (not bad), bad


def explain_checked(item, use_llm=False):
    fallback = template(item)
    if not use_llm:
        return fallback, {"source": "template"}
    facts = {k: item[k] for k in FACT_KEYS if k in item}
    out = complete("You explain GitHub issue recommendations to a beginner. Use ONLY these facts and invent nothing "
                   "(no new numbers, no new file names). Write 3 sentences on why it fits, then 'Starter steps:' with 3 short bullets "
                   f"based on the issue text.\nFACTS: {json.dumps(facts)}\nISSUE TEXT: {item.get('snippet', '')}", 450)
    if not out:
        return fallback, {"source": "template", "reason": "llm failed"}
    ok, bad = check(out, item)
    if not ok:
        return fallback, {"source": "template", "reason": f"guardrail rejected invented numbers {bad}"}
    return out, {"source": "llm", "guardrail": "passed"}


def explain(item, use_llm=False):
    return explain_checked(item, use_llm)[0]