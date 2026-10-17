import re
import networkx as nx

G = nx.DiGraph()
LIGHT = {"client-go", "cobra", "gin", "pytest", "fastapi", "flask", "yaml", "markdown", "bash", "git"}
# "light" skills can be picked up on the spot if you already know their prerequisites.


def edge(a, b, rel):
    G.add_edge(a, b, rel=rel)


for s in ["go", "python", "javascript", "typescript", "java", "rust", "docker", "kubernetes", "react",
          "client-go", "cobra", "gin", "fastapi", "flask", "pytest", "yaml", "markdown", "bash", "git"]:
    G.add_node(s)

edge("kubernetes", "go", "REQUIRES"); edge("kubernetes", "docker", "REQUIRES"); edge("kubernetes", "yaml", "REQUIRES")
edge("client-go", "go", "REQUIRES"); edge("cobra", "go", "REQUIRES"); edge("gin", "go", "REQUIRES")
edge("fastapi", "python", "REQUIRES"); edge("flask", "python", "REQUIRES"); edge("pytest", "python", "REQUIRES")
edge("client-go", "kubernetes", "PART_OF")
edge("react", "javascript", "IMPLIES"); edge("typescript", "javascript", "IMPLIES")

PATTERNS = {
    "go": r"\bgolang\b|\.go\b|go\.mod|\bgo (?:test|mod|build|func|module)",
    "python": r"\bpython\b|\.py\b|\bpip\b", "javascript": r"\bjavascript\b|\.js\b|\bnpm\b",
    "typescript": r"\btypescript\b|\.tsx?\b", "react": r"\breact\b|\.jsx\b",
    "docker": r"\bdocker(?:file)?\b|container image", "kubernetes": r"\bkubernetes\b|\bkubectl\b|\bk8s\b|\bcrd\b",
    "client-go": r"client-go|clientset|informer", "cobra": r"\bcobra\b", "gin": r"\bgin\b",
    "fastapi": r"\bfastapi\b", "flask": r"\bflask\b", "pytest": r"\bpytest\b",
    "yaml": r"\byaml\b|\.ya?ml\b", "markdown": r"\bmarkdown\b|\.md\b|readme|\bdocs?\b|documentation|typo",
    "bash": r"\bbash\b|\.sh\b|shell script", "java": r"\bjava\b", "rust": r"\brust\b|\.rs\b",
}


def prereqs(s):
    return {t for _, t, d in G.out_edges(s, data=True) if d["rel"] == "REQUIRES"} if s in G else set()


def closure(skills):
    """Skills the user effectively has (follows IMPLIES: React => JavaScript)."""
    have = {s.lower() for s in skills}
    stack = list(have)
    while stack:
        s = stack.pop()
        if s in G:
            for _, t, d in G.out_edges(s, data=True):
                if d["rel"] == "IMPLIES" and t not in have:
                    have.add(t); stack.append(t)
    return have


def missing(user_skills, required):
    have = closure(user_skills)
    out = set()
    for s in required:
        if s in have:
            continue
        if s in LIGHT and prereqs(s) <= have:
            continue                      # client-go is fine for someone who knows Go
        out.add(s)
    return out


def skills_to_learn(user_skills, required):
    have = closure(user_skills)
    out, stack = set(), list(missing(user_skills, required))
    while stack:
        s = stack.pop()
        if s in out or s in have:
            continue
        out.add(s)
        stack.extend(prereqs(s))
    return out


def extract_keywords(text, language):
    t = text.lower()
    found = {k for k, p in PATTERNS.items() if re.search(p, t)}
    lang = (language or "").lower()
    if lang in G and not (found and found <= {"markdown", "yaml"}):
        found.add(lang)
    return found