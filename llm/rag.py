import base64
import chromadb
from collector.github_client import GitHubClient
from llm.client import complete

gh = GitHubClient()


def fetch_contributing(repo):
    for p in ["CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md"]:
        b = gh.get(f"/repos/{repo}/contents/{p}")["body"]
        if b:
            return base64.b64decode(b["content"]).decode("utf-8", "ignore")


def chunk(text, size=800):
    out, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) > size and cur:
            out.append(cur); cur = ""
        cur += para + "\n\n"
    return out + [cur] if cur.strip() else out


def ask(repo, question):
    text = fetch_contributing(repo)
    if not text:
        return "This repo has no CONTRIBUTING.md."
    col = chromadb.Client().get_or_create_collection(repo.replace("/", "_"))
    parts = chunk(text)
    col.upsert(documents=parts, ids=[str(i) for i in range(len(parts))])
    hits = col.query(query_texts=[question], n_results=3)["documents"][0]
    return complete("Answer ONLY from these excerpts. If missing, say so.\n\n" + "\n---\n".join(hits) + f"\n\nQuestion: {question}")