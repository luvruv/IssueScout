import json, re
from ai.ontology import G, extract_keywords
from llm.client import complete


def parse_skills(out):
    """Strict parse: JSON object with a list under 'skills', every item must be a known ontology skill."""
    m = re.search(r"\{.*\}", out or "", re.S)
    if not m:
        raise ValueError("no JSON object")
    data = json.loads(m.group(0))
    if not isinstance(data.get("skills"), list):
        raise ValueError("'skills' is not a list")
    return {str(s).lower() for s in data["skills"] if str(s).lower() in G}


def extract_skills_meta(title, body, language, use_llm=False):
    kw = extract_keywords(f"{title} {body}", language)
    if not use_llm:
        return kw, "keywords"
    prompt = ('Return ONLY JSON like {"skills": ["go"]}. Choose the skills needed to solve this GitHub issue '
              f"from this list only: {sorted(G.nodes)}\nTitle: {title}\nBody: {(body or '')[:1500]}")
    try:
        got = parse_skills(complete(prompt, 200))
        return (got or kw), ("llm" if got else "keywords(empty llm)")
    except Exception:
        return kw, "keywords(llm invalid)"


def extract_skills(title, body, language, use_llm=False):
    return extract_skills_meta(title, body, language, use_llm)[0]