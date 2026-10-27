"""RAG over a repo's contributor docs: CONTRIBUTING, README, dev guides, PR template, Makefile targets.

Retrieval backends
  tfidf     (default) scikit-learn only: works offline, nothing to download
  langchain (optional) RecursiveCharacterTextSplitter + Chroma + HuggingFaceEmbeddings; pip install -r requirements-rag.txt
Generation: LangChain chain if installed and a provider is configured, else llm.client.complete, else an EXTRACTIVE
answer (best sentences from the docs), so the feature still works with no LLM at all.
Every answer carries sources and a groundedness score."""
import base64, json, os, re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from llm.client import complete, provider

DOC_PATHS = ["CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md", "docs/contributing.md", "CONTRIBUTING.rst",
             "README.md", "DEVELOPMENT.md", "docs/development.md", "docs/DEVELOPMENT.md", "docs/dev/README.md", "HACKING.md",
             ".github/PULL_REQUEST_TEMPLATE.md", ".github/pull_request_template.md", "PULL_REQUEST_TEMPLATE.md", "Makefile"]
NOT_FOUND = "NOT FOUND IN DOCS"
GENERIC = {"run", "use", "using", "file", "files", "project", "code", "help", "need", "make", "work", "does", "used", "exist", "install", "build"}
CMD = re.compile(r"^\s*\$|`[^`]+`|\b(?:tox|pip|pytest|make|git|npm|cargo|go test)\b|scripts/")
MIN_COVERAGE = 0.34      # refuse when under a third of the question's content words appear in the retrieved text (tuned on 11 questions: re-check on yours)


def coverage(question, text):
    """Share of the question's content words found in the retrieved text (crude, language-independent refusal signal)."""
    words = [w for w in re.findall(r"[a-z][a-z\-]+", question.lower()) if w not in ENGLISH_STOP_WORDS and w not in GENERIC and len(w) > 2]
    if not words:
        return 1.0
    toks = {w[:5] for w in re.findall(r"[a-z][a-z\-]+", text.lower())}
    return sum(1 for w in words if w[:5] in toks) / len(words)


def fetch_docs(repo, refresh=False):
    cache = f"data/docs/{repo.replace('/', '__')}.json"
    if os.path.exists(cache) and not refresh:
        return json.load(open(cache))
    from collector.github_client import GitHubClient
    gh = GitHubClient(); docs = []
    for p in DOC_PATHS:
        b = gh.get(f"/repos/{repo}/contents/{p}")["body"]
        if not isinstance(b, dict) or not b.get("content"):
            continue
        text = base64.b64decode(b["content"]).decode("utf-8", "ignore")
        if p == "Makefile":
            targets = sorted(set(re.findall(r"^([A-Za-z0-9_.\-]+):(?!=)", text, re.M)))[:60]
            text = "Makefile targets: " + ", ".join(targets)
        docs.append({"path": p, "text": text[:60000]})
    os.makedirs("data/docs", exist_ok=True)
    json.dump(docs, open(cache, "w"))
    return docs


def chunk_docs(docs, size=900):
    """Split by markdown heading, then by paragraph; every chunk remembers its file and heading."""
    out = []
    for d in docs:
        heading, buf = "(top)", ""
        def flush():
            nonlocal buf
            if buf.strip():
                out.append({"path": d["path"], "heading": heading, "text": buf.strip()})
            buf = ""
        for para in re.split(r"\n\s*\n", d["text"]):
            if re.match(r"^\s{0,3}#{1,4}\s", para):
                flush(); heading = para.strip().split("\n")[0].lstrip("# ").strip()[:80]
            if len(buf) + len(para) > size:
                flush()
            buf += para + "\n\n"
        flush()
    return out


class TfidfRetriever:
    name = "tfidf"
    def __init__(self, chunks):
        self.chunks = chunks
        self.tf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.X = self.tf.fit_transform([f"{c['heading']} {c['text']}" for c in chunks])

    def search(self, q, k):
        s = (self.X @ self.tf.transform([q]).T).toarray().ravel()
        return [dict(self.chunks[i], score=float(s[i])) for i in np.argsort(-s)[:k] if s[i] > 0]


class LangChainRetriever:
    name = "langchain"
    def __init__(self, chunks):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        docs = [Document(page_content=p, metadata={"path": c["path"], "heading": c["heading"]}) for c in chunks for p in splitter.split_text(c["text"])]
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = Chroma.from_documents(docs, emb)

    def search(self, q, k):
        return [{"path": d.metadata["path"], "heading": d.metadata["heading"], "text": d.page_content, "score": float(1 / (1 + s))}
                for d, s in self.db.similarity_search_with_score(q, k=k)]


def _terms(s):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9_\-./]+", s.lower()) if len(w) > 2}


def grounded_score(answer, context):
    """Share of the answer's checkable tokens (`code`, paths, numbers, ACRONYMS, flags) that appear in the retrieved text."""
    toks = set(re.findall(r"`([^`]{2,60})`", answer)) | set(re.findall(r"\b[A-Z]{2,6}\b", answer)) | set(re.findall(r"\b\d+(?:\.\d+)?\b", answer)) \
        | set(re.findall(r"(?<!\w)--?[a-z][\w-]+", answer)) | set(re.findall(r"\b[\w.-]+/[\w./-]+\b", answer))
    toks = {t for t in toks if t not in {"NOT", "FOUND", "IN", "DOCS", "OK"}}
    if not toks:
        return 1.0, 0
    c = context.lower()
    return round(sum(1 for t in toks if t.lower() in c) / len(toks), 2), len(toks)


class RepoRAG:
    def __init__(self, repo, docs=None, backend="auto"):
        self.repo = repo
        self.docs = docs if docs is not None else fetch_docs(repo)
        self.chunks = chunk_docs(self.docs)
        self.backend = "tfidf"
        self.r = None
        if backend in ("auto", "langchain") and self.chunks:
            try:
                self.r = LangChainRetriever(self.chunks); self.backend = "langchain"
            except Exception as e:
                if backend == "langchain":
                    raise
        if self.r is None and self.chunks:
            self.r = TfidfRetriever(self.chunks)

    def retrieve(self, q, k=4):
        return self.r.search(q, k) if self.r else []

    def _extractive(self, q, hits):
        qt = _terms(q); sents = []
        for h in hits[:3]:
            for s in re.split(r"(?<=[.!?])\s+|\n", h["text"]):
                s = s.strip()
                if len(s) > 12:
                    ov = len(qt & _terms(s))
                    sents.append((ov + (2 if ov and CMD.search(s) else 0), s, h))      # commands that match the question rank higher
        sents.sort(key=lambda t: -t[0])
        return " ".join(s for _, s, _ in sents[:4]) or NOT_FOUND

    def ask(self, question, k=4, use_llm=True):
        hits = self.retrieve(question, k)
        if not hits:
            return {"answer": NOT_FOUND, "sources": [], "grounded": 1.0, "mode": "none", "backend": self.backend}
        cov = coverage(question, " ".join(h["text"] for h in hits))
        if cov < MIN_COVERAGE:
            return {"answer": NOT_FOUND, "sources": [], "grounded": 1.0, "mode": "refused (low coverage)", "backend": self.backend, "coverage": round(cov, 2)}
        ctx = "\n".join(f"[{i}] ({h['path']} > {h['heading']})\n{h['text']}" for i, h in enumerate(hits, 1))
        ans, mode = None, "extractive"
        if use_llm and provider() != "none":
            ans = complete("Answer the question ONLY from these excerpts. If the answer is not in them, reply exactly: NOT FOUND IN DOCS. "
                           "Quote commands exactly as written and cite sources like [1]. Be brief and practical.\n"
                           f"EXCERPTS:\n{ctx}\nQUESTION: {question}", 500)
            mode = "llm" if ans else "extractive (llm failed)"
        if not ans:
            ans = self._extractive(question, hits)
        g, n = grounded_score(ans, ctx)
        return {"answer": ans.strip(), "sources": [{"path": h["path"], "heading": h["heading"], "score": round(h["score"], 3), "snippet": h["text"][:240], "text": h["text"]} for h in hits],
                "coverage": round(cov, 2), "grounded": g, "checked_tokens": n, "mode": mode, "backend": self.backend,
                "warning": "low groundedness: verify against the linked docs" if g < 0.7 else None}


_CACHE = {}


def ask(repo, question, use_llm=True, docs=None):
    if repo not in _CACHE or docs is not None:
        _CACHE[repo] = RepoRAG(repo, docs)
    return _CACHE[repo].ask(question, use_llm=use_llm)