"""First-PR kit: everything a newcomer needs for ONE issue. It drafts text; it never posts anything to GitHub."""
import os, re
from features.text_features import RE_PATH
from llm.rag import RepoRAG, fetch_docs

CLA = re.compile(r"Contributor License Agreement|\bCLA\b|easycla|cla-assistant", re.I)
DCO = re.compile(r"Developer Certificate of Origin|\bDCO\b|Signed-off-by|git commit (?:-s|--signoff)", re.I)
TEST = re.compile(r"(make (?:test|check|lint|verify|unit)[\w-]*|go test[^\n`]*|pytest[^\n`]*|npm (?:run )?test|cargo test|tox|nox|yarn test)")
SETUP = re.compile(r"(git clone[^\n`]*|make (?:build|install|setup|deps|bootstrap)[\w-]*|pip install[^\n`]*|go build[^\n`]*|npm install|yarn install|cargo build|poetry install|pre-commit install)")
ASSIGN = re.compile(r"(?m)^[^\n]*?(/assign|/claim|/take)\b[^\n]*")
CONV = re.compile(r"conventional commits|\b(?:feat|fix|docs|chore)(?:\([\w-]+\))?:", re.I)
QUESTIONS = {"setup": "How do I set up a development environment and build the project?",
             "tests": "How do I run the tests and linters before submitting?",
             "process": "What is the pull request process, including sign-off or review requirements?"}


def _uniq(rx, text, n=6):
    seen = []
    for m in rx.finditer(text):
        s = m.group(1).strip().rstrip(".,;)")
        if s and s not in seen:
            seen.append(s)
    return seen[:n]


def build_kit(ref, docs=None, issue=None, use_llm=False):
    from ml.predict import parse_ref, _row_from_db, _row_from_api
    repo, number = parse_ref(ref)
    if issue is None:
        row = _row_from_db(repo, number)
        row = row if row is not None else _row_from_api(repo, number)
        issue = row.iloc[0].to_dict()
    docs = docs if docs is not None else fetch_docs(repo)
    allt = "\n".join(d["text"] for d in docs)
    title, body = issue["title"], issue.get("body") or ""
    files = sorted(set(RE_PATH.findall(f"{title}\n{body}")))[:8]
    idents = sorted(set(re.findall(r"`([A-Za-z_][\w.]{3,40})`", body)))[:6]
    tmpl = next((d["text"] for d in docs if "pull_request_template" in d["path"].lower()), "")
    sections = [l.strip("# ").strip() for l in tmpl.splitlines() if l.startswith("#")][:8]
    checklist = [l.strip() for l in tmpl.splitlines() if re.match(r"\s*[-*] \[[ x]\]", l)][:8]
    mk = next((d["text"] for d in docs if d["path"] == "Makefile"), "")
    mk_targets = [t.strip() for t in mk.replace("Makefile targets:", "").split(",") if t.strip() and not t.strip().startswith(".")]
    kit = {
        "issue": f"{repo}#{number}", "title": title, "url": f"https://github.com/{repo}/issues/{number}",
        "requires_cla": bool(CLA.search(allt)), "requires_dco_signoff": bool(DCO.search(allt)),
        "claim_style": (ASSIGN.search(allt).group(1) if ASSIGN.search(allt) else "comment on the issue and wait for a maintainer to reply"),
        "setup_commands": _uniq(SETUP, allt), "test_commands": _uniq(TEST, allt) or [f"make {t}" for t in mk_targets if t in ("test", "check", "lint", "ci")][:3],
        "commit_style": "Conventional Commits (feat:, fix:, docs:)" if CONV.search(allt) else "follow recent commit messages in `git log`",
        "pr_template_sections": sections, "pr_checklist": checklist, "docs_found": [d["path"] for d in docs],
        "files_mentioned": files, "search_hints": [f'git grep -n "{i}"' for i in idents] or ([f'git grep -n "{w}"' for w in re.findall(r"\b[a-z]+[A-Z]\w+|\b\w+_\w+\b", title)[:3]]),
    }
    if docs:
        rag = RepoRAG(repo, docs)
        kit["doc_answers"] = {k: rag.ask(q, use_llm=use_llm) for k, q in QUESTIONS.items()}
    signoff = "git commit -s -m \"<message>\"" if kit["requires_dco_signoff"] else "git commit -m \"<message>\""
    kit["steps"] = [
        f"Read the whole issue and its comments; check nobody else is working on it: {kit['url']}",
        f"Claim it politely ({kit['claim_style']}) and WAIT for a reply or an assignment before coding.",
        "Fork the repo, clone your fork, add the upstream remote, create a branch: `git checkout -b fix-" + str(number) + "`",
        "Set up: " + ("; ".join(kit["setup_commands"][:3]) or "see the README / CONTRIBUTING (no commands detected)"),
        "Reproduce the problem first. Look at: " + (", ".join(files) or "(no file named in the issue: use search hints)"),
        "Make the smallest change that fixes it, add or update a test.",
        "Run: " + ("; ".join(kit["test_commands"][:3]) or "the project's test command (see docs)"),
        ("Sign the CLA first (the repo mentions one). " if kit["requires_cla"] else "") + f"Commit: `{signoff}`",
        "Open the PR with the repo's template and `Fixes #" + str(number) + "` in the description.",
    ]
    kit["draft_claim_comment"] = (f"Hi! I'd like to work on this. My understanding is: {title}. "
                                  f"I plan to {'start from ' + files[0] if files else 'locate the relevant code first'} and add a test. "
                                  "Is it still available, and is there anything I should know before starting?")
    kit["draft_pr_description"] = ("## What\n<one or two sentences>\n\n## Why\nFixes #" + str(number) + "\n\n## How I tested\n<commands you ran>\n"
                                   + ("\n## Checklist (from the repo template)\n" + "\n".join(checklist) if checklist else ""))
    kit["warning"] = "These are DRAFTS. Review and post them yourself; IssueScout never comments on GitHub for you."
    return kit


def render_markdown(kit):
    L = [f"# First-PR kit: {kit['issue']}", f"**{kit['title']}**  \n{kit['url']}", "",
         f"- CLA required: **{kit['requires_cla']}**  |  DCO sign-off required: **{kit['requires_dco_signoff']}**",
         f"- How to claim: {kit['claim_style']}", f"- Commit style: {kit['commit_style']}", f"- Docs read: {', '.join(kit['docs_found']) or 'none found'}", "", "## Steps"]
    L += [f"{i}. {s}" for i, s in enumerate(kit["steps"], 1)]
    for k, a in kit.get("doc_answers", {}).items():
        L += ["", f"## From the docs: {k}", a["answer"], f"_sources: {', '.join(s['path'] for s in a['sources'][:3])} | grounded {a['grounded']} | {a['mode']}_"]
    L += ["", "## Draft comment to claim the issue", f"> {kit['draft_claim_comment']}", "", "## Draft PR description", "```", kit["draft_pr_description"], "```", "", f"**{kit['warning']}**"]
    return "\n".join(L)


def save(kit):
    os.makedirs("data/kits", exist_ok=True)
    p = f"data/kits/{kit['issue'].replace('/', '__').replace('#', '_')}.md"
    open(p, "w", encoding="utf-8").write(render_markdown(kit))
    return p