"""Skill knowledge graph (NetworkX). Edges: REQUIRES (needs first), PART_OF, IMPLIES (knowing A means knowing B)."""
import re
import networkx as nx

G = nx.DiGraph()
# "light" skills can be picked up on the spot if the user already has their prerequisites
LIGHT = {"client-go", "cobra", "gin", "pytest", "fastapi", "flask", "yaml", "json", "markdown", "bash", "git", "make",
         "github-actions", "protobuf", "grpc", "http", "rest-api", "sqlalchemy", "asyncio", "numpy", "express", "helm", "prometheus"}

NODES = ["go", "python", "javascript", "typescript", "java", "rust", "sql", "bash", "docker", "kubernetes", "helm", "terraform",
         "yaml", "json", "make", "git", "github-actions", "linux", "grpc", "protobuf", "prometheus", "etcd", "http", "rest-api",
         "postgresql", "redis", "client-go", "cobra", "gin", "fastapi", "flask", "django", "pytest", "pandas", "numpy",
         "scikit-learn", "sqlalchemy", "asyncio", "react", "vue", "node", "express", "tokio", "markdown"]
for n in NODES:
    G.add_node(n)

REQ = [("kubernetes", "go"), ("kubernetes", "docker"), ("kubernetes", "yaml"), ("helm", "kubernetes"), ("helm", "yaml"),
       ("client-go", "go"), ("cobra", "go"), ("gin", "go"), ("etcd", "go"), ("grpc", "protobuf"), ("prometheus", "go"),
       ("fastapi", "python"), ("flask", "python"), ("django", "python"), ("pytest", "python"), ("pandas", "python"),
       ("numpy", "python"), ("scikit-learn", "python"), ("sqlalchemy", "python"), ("sqlalchemy", "sql"), ("asyncio", "python"),
       ("github-actions", "yaml"), ("express", "node"), ("tokio", "rust"), ("postgresql", "sql"), ("rest-api", "http"),
       ("docker", "linux"), ("bash", "linux")]
for a, b in REQ:
    G.add_edge(a, b, rel="REQUIRES")
G.add_edge("client-go", "kubernetes", rel="PART_OF")
for a, b in [("react", "javascript"), ("typescript", "javascript"), ("vue", "javascript"), ("node", "javascript")]:
    G.add_edge(a, b, rel="IMPLIES")
for n in ["react", "vue", "node"]:
    G.add_node(n)

PATTERNS = {
    "go": r"\bgolang\b|\.go\b|go\.mod|\bgo (?:test|mod|build|func|module|vet|run|install)\b",
    "python": r"\bpython\b|\.py\b|\bpip\b|\bpypi\b", "javascript": r"\bjavascript\b|\.js\b|\bnpm\b",
    "typescript": r"\btypescript\b|\.tsx?\b", "java": r"\bjava\b|\.java\b|\bmaven\b|\bgradle\b", "rust": r"\brust\b|\.rs\b|\bcargo\b",
    "sql": r"\bsql\b|\bquery\b.*\bdatabase\b|\bsqlite\b", "bash": r"\bbash\b|\.sh\b|shell script", "docker": r"\bdocker(?:file)?\b|container image",
    "kubernetes": r"\bkubernetes\b|\bkubectl\b|\bk8s\b|\bcrd\b|\bkubelet\b", "helm": r"\bhelm\b|\bchart\.yaml\b", "terraform": r"\bterraform\b|\.tf\b",
    "yaml": r"\byaml\b|\.ya?ml\b", "json": r"\bjson\b", "make": r"\bmakefile\b|\bmake (?:test|build|lint)\b", "git": r"\bgit (?:rebase|merge|clone|commit)\b",
    "github-actions": r"github actions|\.github/workflows|\bworkflow\b", "linux": r"\blinux\b|\bsystemd\b|\bcgroup", "grpc": r"\bgrpc\b",
    "protobuf": r"\bprotobuf\b|\.proto\b", "prometheus": r"\bprometheus\b|\bpromql\b", "etcd": r"\betcd\b", "http": r"\bhttp\b|\bhttps\b request|status code",
    "rest-api": r"\brest\b|\bapi endpoint\b", "postgresql": r"\bpostgres(?:ql)?\b", "redis": r"\bredis\b", "client-go": r"client-go|clientset|informer",
    "cobra": r"\bcobra\b", "gin": r"\bgin-gonic\b|\bgin\b", "fastapi": r"\bfastapi\b", "flask": r"\bflask\b", "django": r"\bdjango\b", "pytest": r"\bpytest\b",
    "pandas": r"\bpandas\b|\bdataframe\b", "numpy": r"\bnumpy\b|\bndarray\b", "scikit-learn": r"\bsklearn\b|scikit-learn", "sqlalchemy": r"\bsqlalchemy\b",
    "asyncio": r"\basyncio\b|\basync def\b|\bawait\b", "react": r"\breact\b|\.jsx\b", "vue": r"\bvue\b", "node": r"\bnode\.?js\b|\bnode_modules\b",
    "express": r"\bexpress\b", "tokio": r"\btokio\b", "markdown": r"\bmarkdown\b|\.md\b|readme|\bdocs?\b|documentation|typo",
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


_COMPILED = {k: re.compile(p) for k, p in PATTERNS.items()}      # compile once: this runs for every issue


def extract_keywords(text, language):
    t = text[:4000].lower()
    found = {k for k, rx in _COMPILED.items() if rx.search(t)}
    lang = (language or "").lower()
    if lang in G and not (found and found <= {"markdown", "yaml", "json"}):
        found.add(lang)
    return found