import json, os, re, sys

CMD = re.compile(r"^\s*\$|`[^`]+`|\b(?:tox|pip|pytest|make|git|npm|cargo|go (?:test|build|run|vet)|scripts/|pre-commit|nox|uv)\b")


def main():
    repo = sys.argv[1]
    f = f"data/docs/{repo.replace('/', '__')}.json"
    if not os.path.exists(f):
        from llm.rag import fetch_docs
        docs = fetch_docs(repo)
    else:
        docs = json.load(open(f))
    if not docs:
        print("no docs found for", repo); return
    for d in docs:
        print(f"\n=== {d['path']} ({len(d['text'])} chars)")
        for ln in d["text"].splitlines():
            s = ln.strip()
            if re.match(r"^#{1,4}\s", s):
                print("\n  #", s.lstrip("# ")[:90])
            elif s and CMD.search(s) and len(s) < 160 and "http" not in s:
                print("     ", s)


if __name__ == "__main__":
    main()