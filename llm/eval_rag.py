"""Score the RAG on a hand-checked question set:  python cli.py eval-rag data/eval/rag_questions.json
File format (see data/eval/rag_questions.template.json):
  [{"repo": "owner/name", "question": "...", "must_include": ["make test"], "answerable": true}, ...]
Pass = every must_include string appears in the answer or in the retrieved sources.  answerable=false => must say NOT FOUND."""
import json, os, time
from llm.rag import RepoRAG, fetch_docs, NOT_FOUND
from ml.common import save_metrics


PLACEHOLDERS = {"owner/name", "real/repo"}


def main(path="data/eval/rag_questions.json", use_llm=True):
    qs = [q for q in json.load(open(path)) if not q.get("template")]
    fake = [q["question"] for q in qs if q["repo"].lower() in PLACEHOLDERS or q["question"].strip().lower() == "your question"
            or any("exact text" in m.lower() for m in q.get("must_include", []))]
    if fake:
        raise SystemExit(f"{path} still contains template placeholders (e.g. {fake[0]!r}). Replace them with real repos and questions first.")
    if not qs:
        raise SystemExit("No real questions yet: copy rag_questions.template.json to rag_questions.json and fill it in.")
    cache, rows = {}, []
    for q in qs:
        if q["repo"] not in cache:
            cache[q["repo"]] = RepoRAG(q["repo"], fetch_docs(q["repo"]))
        t = time.time(); r = cache[q["repo"]].ask(q["question"], use_llm=use_llm); dt = time.time() - t
        retrieved = " ".join(s.get("text", s["snippet"]) for s in r["sources"]).lower()
        if q.get("answerable", True):
            ret_hit = all(m.lower() in retrieved for m in q.get("must_include", []))
            ok = all(m.lower() in r["answer"].lower() for m in q.get("must_include", [])) and r["answer"] != NOT_FOUND
        else:
            ret_hit, ok = None, NOT_FOUND in r["answer"]
        rows.append({"repo": q["repo"], "q": q["question"], "pass": bool(ok), "retrieval_hit": ret_hit, "grounded": r["grounded"], "mode": r["mode"], "sec": round(dt, 2)})
        print(("PASS " if ok else "FAIL ") + f"{q['repo']}: {q['question'][:60]}  (grounded {r['grounded']})")
    ans = [r for r, q in zip(rows, qs) if q.get("answerable", True)]
    summ = {"n": len(rows), "accuracy": round(sum(r["pass"] for r in rows) / len(rows), 3),
            "retrieval_recall@4": round(sum(bool(r["retrieval_hit"]) for r in ans) / max(len(ans), 1), 3),
            "mean_groundedness": round(sum(r["grounded"] for r in ans) / max(len(ans), 1), 3),
            "refusal_correct": round(sum(r["pass"] for r, q in zip(rows, qs) if not q.get("answerable", True)) / max(sum(1 for q in qs if not q.get("answerable", True)), 1), 3),
            "backend": cache[qs[0]["repo"]].backend, "mode": rows[0]["mode"], "mean_seconds": round(sum(r["sec"] for r in rows) / len(rows), 2)}
    key = "rag_eval_heldout" if "heldout" in os.path.basename(path).lower() else "rag_eval"
    print(summ); save_metrics(key, summ)
    return summ