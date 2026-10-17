import json, re
from ai.ontology import G, extract_keywords
from llm.client import complete


def extract_skills(title, body, language, use_llm=False):
    kw = extract_keywords(f"{title} {body}", language)
    if not use_llm:
        return kw
    prompt = ('Return ONLY JSON like {"skills": ["go"]}. Choose the skills needed to solve this GitHub issue '
              f"from this list only: {sorted(G.nodes)}\nTitle: {title}\nBody: {(body or '')[:1500]}")
    out = complete(prompt, 200)
    try:
        data = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
        return {s for s in data["skills"] if s in G} or kw
    except Exception:
        return kw