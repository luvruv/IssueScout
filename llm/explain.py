import json
from llm.client import complete


def explain(item, use_llm=False):
    fallback = (f"Predicted {item['difficulty']} ({item['diff_conf']:.0%} confidence), about {item['est_hours']}h, "
                f"merge probability {item['p_merge']:.0%}. Needs: {', '.join(item['skills']) or 'no special skills'}. "
                f"Rules: {'; '.join(item['rules']) or 'none fired'}.")
    if not use_llm:
        return fallback
    facts = {k: item[k] for k in ["repo", "number", "title", "difficulty", "diff_conf", "p_merge",
                                  "est_hours", "reply_days", "skills", "rules"]}
    out = complete("You explain GitHub issue recommendations to a beginner. Use ONLY these facts and invent nothing. "
                   "Write 3 sentences on why it fits, then 'Starter steps:' with 3 short bullets based on the issue text.\n"
                   f"FACTS: {json.dumps(facts)}\nISSUE TEXT: {item['snippet']}")
    return out or fallback