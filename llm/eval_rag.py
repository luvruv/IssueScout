"""Score the RAG on a hand-checked question set:  python cli.py eval-rag data/eval/rag_questions.json
File format (see data/eval/rag_questions.template.json):
  [{"repo": "owner/name", "question": "...", "must_include": ["make test"], "answerable": true}, ...]
Pass = every must_include string appears in the answer or in the retrieved sources.  answerable=false => must say NOT FOUND."""
import json, time
from llm.rag import RepoRAG, fetch_docs, NOT_FOUND
from ml.common import save_metrics


def main(path="data/eval/rag_questions.json", use_llm=True):
    qs = [q for q in json.load(open(path)) if not q.get("template")]
    if not qs:
        raise SystemExit("No real questions yet: copy rag_questions.template.json to rag_questions.json and fill it in.")
    cache, rows = {}, []
    for q in qs:
        if q["repo"] not in cache:
            cache[q["repo"]] = RepoRAG(q["repo"], fetch_docs(q["repo"]))
        t = time.time(); r = cache[q["repo"]].ask(q["question"], use_llm=use_llm); dt = time.time() - t
        blob = (r["answer"] + " " + " ".join(s["snippet"] for s in r["sources"])).lower()
        if q.get("answerable", True):
            hit = all(m.lower() in blob for m in q.get("must_include", []))
            ok = hit and r["answer"] != NOT_FOUND
        else:
            ok = NOT_FOUND in r["answer"]
        rows.append({"repo": q["repo"], "q": q["question"], "pass": bool(ok), "grounded": r["grounded"], "mode": r["mode"], "sec": round(dt, 2)})
        print(("PASS " if ok else "FAIL ") + f"{q['repo']}: {q['question'][:60]}  (grounded {r['grounded']})")
    ans = [r for r, q in zip(rows, qs) if q.get("answerable", True)]
    summ = {"n": len(rows), "accuracy": round(sum(r["pass"] for r in rows) / len(rows), 3),
            "mean_groundedness": round(sum(r["grounded"] for r in ans) / max(len(ans), 1), 3),
            "refusal_correct": round(sum(r["pass"] for r, q in zip(rows, qs) if not q.get("answerable", True)) / max(sum(1 for q in qs if not q.get("answerable", True)), 1), 3),
            "backend": cache[qs[0]["repo"]].backend, "mode": rows[0]["mode"], "mean_seconds": round(sum(r["sec"] for r in rows) / len(rows), 2)}
    print(summ); save_metrics("rag_eval", summ)
    return summ
