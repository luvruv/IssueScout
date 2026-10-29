"""Catch mistakes in your RAG test questions BEFORE you run eval-rag:   python -m tools.check_questions
For every answerable question it checks that each must_include string really appears in that repo's cached docs."""
import glob, json, os, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/eval/rag_questions.json"


def main():
    qs = [q for q in json.load(open(PATH)) if not q.get("template")]
    bad = 0
    for q in qs:
        if q["repo"].lower() in ("owner/name", "real/repo") or q["question"].strip().lower() == "your question":
            print(f"PLACEHOLDER (replace with a real repo/question): {q['repo']} | {q['question'][:50]}"); bad += 1; continue
        f = f"data/docs/{q['repo'].replace('/', '__')}.json"
        if not os.path.exists(f):
            print(f"NO DOCS CACHED for {q['repo']}: run  python cli.py ask {q['repo']} \"How do I run the tests?\"  once"); bad += 1; continue
        text = "\n".join(d["text"] for d in json.load(open(f))).lower()
        if q.get("answerable", True) and not q.get("must_include"):
            print(f"NO must_include for: {q['question']}"); bad += 1
        for m in q.get("must_include", []):
            if m.lower() not in text:
                print(f"NOT FOUND in {q['repo']} docs: {m!r}   (question: {q['question'][:50]})"); bad += 1
    print(f"\n{len(qs)} questions, {sum(1 for q in qs if q.get('answerable', True))} answerable, {sum(1 for q in qs if not q.get('answerable', True))} unanswerable, {bad} problems")


if __name__ == "__main__":
    main()