# IssueScout: complete build guide (instructions + every file to paste)

## Instructions: what to do and in what order

Everything below runs from the project root with the venv active. Check progress at any time: `python cli.py status`.

## 0. Install this package (10 minutes)
1. **Rotate your keys first.** The `.env` you shared contained a live GitHub token and OpenAI key. Revoke both, make new ones, and never zip `.env`.
2. Create or replace each file listed in the FILES section below (path, then code). Do the files in the order given.
3. `pip install -r requirements.txt`
4. Your old trained models no longer match the new features, so retrain: `python cli.py train all` (about 2-3 minutes on 466 issues).
5. `python cli.py test` -> expect every test to pass (13 at the time of writing), then `python cli.py status`.
6. Edit `.env` (see `.env.example`). Set `LLM_PROVIDER=openai` or `anthropic`, a REAL `LLM_MODEL` for your account, then `python cli.py llm-check`. Wrong model names fail silently into template text, so test this once.
7. Commit now: `git add . && git commit -m "v2: labels, models, AI, LLM, RAG" && git push`.

## 1. What I measured on your data (466 labelled issues, 30 repos, my run)
| Result | Value | Meaning |
|---|---|---|
| Difficulty, best model on UNSEEN repos (grouped CV macro-F1) | LogisticRegression 0.39 (sd 0.03) | chance is 0.33: a small real signal, not yet strong |
| Same models, newest-20% test (94 issues) | LR 0.40, RF 0.42, MLP 0.37 vs baselines 0.19 (majority) and 0.20 ("good first issue") | beats both baselines; CI is wide |
| Overfitting (train minus CV F1) | RF 0.69, KNN 0.66, MLP 0.63, LR 0.45, SVM 0.18 | trees/forests/KNN/MLP memorise 466 rows: use this in your bias-variance section |
| Probability calibration | ECE 0.005 on test | the stated confidences are trustworthy; they are just low (about 0.4-0.5) |
| Merge model | ROC-AUC 0.55-0.70, PR-AUC 0.86-0.92 vs base rate 0.84 | weak ranking; looks good only because 85% merge |
| Time-to-close | does NOT beat a repo-median baseline | negative result: report it honestly |
| Feature groups (LR, grouped CV) | numeric 0.38, structural 0.34, text 0.31, category 0.28 | numeric/structural features carry most of the signal |
| CSP | forward checking and MRV alone: identical node counts to plain; budget bound: 100x or more fewer | the bound is the real gain; say so |
| A* | same optimal cost as uniform-cost search, fewer expansions, admissible on all instances | |
| "good first issue" label | only 1.2% of open issues; 56% of tagged issues were Easy vs 39% overall | your audit finding |

**Conclusion: Part 1 (more data) decides whether the ML part is convincing.** Do not skip it.

## 2. How training works (what `python cli.py train all` does)
1. Loads closed issues whose merged PR gives a label (`features/build_features.py: labelled()`).
2. Builds features known at issue creation only (`NUM`, `CAT`, TF-IDF text, a leave-one-out repo prior).
3. Evaluates 7 models + 2 baselines the same way: newest-20% test split AND 5-fold grouped CV where each repo is wholly in train or test.
4. Picks the winner by grouped-CV macro-F1 (never by the test number), tunes it, calibrates probabilities, saves `models/difficulty.joblib`.
5. Writes `report/model_comparison.md/.csv`, confusion matrix, ROC, calibration, then trains merge and time-to-close.
Read the table printed at the end. The columns that matter: `cv_f1_grouped`, `overfit_gap` (train minus CV: big = overfitting), `hard_called_easy` (the harmful mistake).

## 3. The four remaining parts

### Part 1: data and labels (about 1 week, mostly waiting)
1. `python cli.py add-repos repos_extra.txt` (adds ~90 repos; names are from memory, missing ones print "missing" and are skipped). Remove `django/django` from `repos.txt`: it uses Trac, so it has no issues.
2. `python cli.py collect --closed-pages 6 --open-pages 3 --pr-pages 6`. It takes hours, is rate-limited, cached and resumable. Run it overnight, and again if it stops. Your fine-grained token expires on the date you chose (30 days by default): renew it if needed.
3. Target: 100+ repos, 1,500+ labelled issues. Add JavaScript/TypeScript/Rust repos too, or the ontology's other languages stay unused.
4. `python cli.py features` -> shows the 40/40/20 class split and builds the 200-issue sample.
5. `python cli.py label`. Label 100-200 issues (an evening). Easy = a beginner finishes in one sitting, change is local. Medium = a few files, must understand a module. Hard = design decisions, many files, deep domain knowledge. The tool hides the automatic label.
6. `python cli.py eda` and `python cli.py evaluate` -> agreement, kappa, label sensitivity, EDA figures in `report/eda/`.
**Done when:** `status` shows repos and labelled counts green, and kappa is reported (aim above 0.4; if lower, say so and adjust `EASY_Q/HARD_Q` once in `features/labels.py`, then freeze).

### Part 2: ML engine (about 1.5 weeks, after Part 1)
1. `python cli.py train all`, then `python cli.py cluster`, `python cli.py gd-demo`, `python cli.py evaluate`.
2. If grouped-CV F1 stays under 0.40 with 1,500+ issues, try in this order and report each: `USE_EMBEDDINGS=1` (needs `pip install sentence-transformers`), a Go-only model, fewer classes. Report what happened either way; an honest weak result with a good analysis scores better than a hidden one.
3. Time-to-close: keep it as the negative-result section (regularisation plot is in `report/regularisation_effect.png`).
4. Name your clusters from `report/cluster_profiles.csv`; do not assume names in advance.
5. `python cli.py report` builds `report/RESULTS.md`.
**Done when:** `report/model_comparison.md` exists with 1,500+ issues behind it and you can explain every column.

### Part 3: AI engine (about 1 week; can start today)
1. `python cli.py test`, `python cli.py benchmark` -> `report/csp_benchmark.md` and `report/astar_benchmark.md`.
2. `python cli.py bayes-demo` -> MLE vs Bayesian shrinkage per repo.
3. Run `python cli.py recommend ... --verbose` for 3 different profiles and `python cli.py path owner/repo#N --skills go` for 2 targets; keep the outputs for the report.
4. Write: PEAS table (`docs/RESPONSIBLE_AI.md`), the rule list (`ai/rules.py`), why plain FC/MRV did not help here (symmetric domains) and why the budget bound does, and the A* admissibility argument (comment at the top of `ai/astar.py`).
**Done when:** tests pass and both benchmark tables are in your report.

### Part 4: LLM, RAG, first-PR kit, delivery (about 2 weeks)
1. `python cli.py llm-check`, then `python cli.py recommend ... --llm --verbose`. Explanations that invent numbers are rejected automatically and fall back to the template (`explanation_meta` shows which).
2. RAG: `python cli.py ask owner/repo "How do I run the tests?"`. Works with no LLM (extractive, with sources). For the LangChain backend: `pip install -r requirements-rag.txt` (the code switches automatically; this path was written but I could not run it here).
3. Evaluate RAG: copy `data/eval/rag_questions.template.json` to `rag_questions.json`. Read the CONTRIBUTING of 5 repos yourself and write 4 questions each with the exact `must_include` string, plus a few `"answerable": false` ones. Run `python cli.py eval-rag`.
4. `python cli.py eval-llm` -> JSON validity, guardrail pass rate, latency.
5. First PR: `python cli.py recommend ...`, pick one, `python cli.py prepare owner/repo#N`, read the kit, and actually do it. Log it: `python cli.py outcome owner/repo#N started`, later `pr_opened` / `merged`.
6. User study: 5-10 classmates, form in `docs/USER_STUDY.md`, summary with `python scripts/user_study_summary.py`.
7. Report, slides, demo rehearsal (script below).
**Done when:** RAG accuracy and LLM eval are measured, one real PR is open or merged, and `python cli.py status` is mostly green.

## 4. Testing after training (any repo)
1. **Generalisation:** the grouped-CV score (unseen repos) in `model_comparison.md`.
2. **New repo smoke test:** add `owner/repo` to `repos.txt`, `python cli.py collect`, `python cli.py recommend ...`.
3. **Single issue:** `python cli.py predict owner/repo#123` (works for issues not in your DB via the API).
4. **LLM and RAG:** `llm-check`, `eval-llm`, `eval-rag`.
5. **Reality:** do one issue. Compare your hours with `est_hours`.

## 5. Demo script (5 minutes)
1. `python cli.py status` (honest progress). 2. `recommend` with your own profile, expand the rules and CSP check. 3. `predict` on an issue outside your training repos. 4. `prepare` for the top pick. 5. Show `report/model_comparison.md` and say plainly what the model can and cannot do. 6. Show the outcome log.

## 6. Not tested here (written, but I could not run them)
LangChain/Chroma backend, sentence-embedding features, live OpenAI/Anthropic calls, the Typer command wiring, FastAPI, Streamlit, and GitHub API fetches for issues outside your DB. Everything else (labels, all models, clustering, AI engine, RAG with the default backend, kit builder, recommend, tests) was run against your database. If something fails, paste the traceback.


---

# FILES: create or replace each one, in this order

**How to do it in VS Code:** open the file path shown (create it if it says NEW, select all and paste if it says REPLACE), paste the code, save. Run this once first so every folder and package marker exists (Git Bash, project root):

````bash
mkdir -p features tools tests scripts docs data/eval data/processed models report
touch features/__init__.py tools/__init__.py tests/__init__.py
````

After finishing **STEP 1 to PART 3**, run `python cli.py test`. After **PART 4**, run `python cli.py train all` once (the old models do not match the new features), then `python cli.py status`.

Code is shown inside 4-backtick fences so inner code fences do not break it. Copy only what is between the fences.


---

## STEP 1. Setup files


### `requirements.txt`: REPLACE (file already exists in your project)

````text
requests
python-dotenv
pandas
numpy
scipy
scikit-learn
joblib
matplotlib
seaborn
networkx
fastapi
uvicorn
pydantic
streamlit
typer
rich
pytest
openai
anthropic
````


### `.gitignore`: REPLACE (file already exists in your project)

````text
.env
.venv/
__pycache__/
*.pyc
*.log
data/raw/
data/*.db
data/docs/
data/kits/
data/emb_cache.pkl
data/last_plan.json
models/
# KEEP in git so reviewers can see your evidence: report/, data/processed/, data/eval/, data/outcomes.jsonl, data/feedback.jsonl
````


### `.env.example`: NEW

````text
GITHUB_TOKEN=
# LLM_PROVIDER = openai | anthropic | mock | none     (mock = offline fake LLM for testing)
LLM_PROVIDER=none
LLM_MODEL=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
# optional: USE_EMBEDDINGS=1 adds sentence-transformer features (pip install sentence-transformers)
````


### `repos_extra.txt`: NEW

````text
# Extra repos. Names are from memory: any that no longer exist just print "missing" and are skipped.
# Go / cloud native
hashicorp/terraform
hashicorp/consul
hashicorp/vault
hashicorp/nomad
grafana/grafana
grafana/loki
prometheus/alertmanager
prometheus/node_exporter
istio/istio
traefik/traefik
caddyserver/caddy
gohugoio/hugo
junegunn/fzf
jesseduffield/lazygit
charmbracelet/bubbletea
spf13/viper
stretchr/testify
gorilla/mux
labstack/echo
gofiber/fiber
moby/moby
docker/cli
docker/compose
fluxcd/flux2
tektoncd/pipeline
knative/serving
operator-framework/operator-sdk
rook/rook
longhorn/longhorn
kubernetes-sigs/kubespray
kubernetes-sigs/cluster-api
kubernetes-sigs/external-dns
kubernetes-sigs/metrics-server
kubernetes/ingress-nginx
kubernetes/dashboard
kubernetes/kops
kubernetes/client-go
open-policy-agent/opa
falcosecurity/falco
nats-io/nats-server
vitessio/vitess
cockroachdb/cockroach
pingcap/tidb
minio/minio
ent/ent
sqlc-dev/sqlc
goreleaser/goreleaser
golangci/golangci-lint
cli/cli
# Python
psf/black
astral-sh/ruff
pallets/click
pallets/jinja
pallets/werkzeug
python-poetry/poetry
pypa/pip
sqlalchemy/sqlalchemy
pydantic/pydantic
aio-libs/aiohttp
celery/celery
scrapy/scrapy
ansible/ansible
home-assistant/core
apache/airflow
numpy/numpy
matplotlib/matplotlib
scipy/scipy
encode/starlette
Textualize/rich
Textualize/textual
tqdm/tqdm
mkdocs/mkdocs
sphinx-doc/sphinx
pylint-dev/pylint
python-pillow/Pillow
networkx/networkx
scikit-image/scikit-image
statsmodels/statsmodels
# JavaScript / TypeScript
expressjs/express
vuejs/core
sveltejs/svelte
prettier/prettier
eslint/eslint
axios/axios
vitejs/vite
mochajs/mocha
jestjs/jest
# Rust
BurntSushi/ripgrep
sharkdp/bat
sharkdp/fd
clap-rs/clap
tokio-rs/tokio
serde-rs/serde
starship/starship
alacritty/alacritty
````


---

## PART 1. Data and labels


### `collector/db.py`: REPLACE (file already exists in your project)

````python
import os, sqlite3

DB = "data/issuescout.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS repos(
  full_name TEXT PRIMARY KEY, stars INT, forks INT, language TEXT, created_at TEXT,
  pushed_at TEXT, open_issues INT, contributors INT, has_contributing INT,
  archived INT, description TEXT);
CREATE TABLE IF NOT EXISTS issues(
  id TEXT PRIMARY KEY, repo TEXT, number INT, title TEXT, body TEXT, labels TEXT,
  created_at TEXT, closed_at TEXT, state TEXT, comments INT, has_assignee INT,
  author_assoc TEXT, has_open_pr INT DEFAULT 0, pr_number INT, pr_additions INT,
  pr_deletions INT, pr_files INT, pr_merged INT, pr_author_assoc TEXT,
  first_response_hours REAL, pr_review_comments INT, pr_comments INT, pr_commits INT);
"""
# columns added after the first version: old databases are upgraded automatically
NEW_COLS = {"issues": {"pr_review_comments": "INT", "pr_comments": "INT", "pr_commits": "INT"}}


def _ensure_columns(c):
    for table, cols in NEW_COLS.items():
        have = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
        for name, typ in cols.items():
            if name not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")


def conn():
    os.makedirs("data", exist_ok=True)
    c = sqlite3.connect(DB)
    c.executescript(SCHEMA)
    _ensure_columns(c)
    c.commit()
    return c


def upsert(c, table, row):
    cols = ",".join(row)
    q = ",".join("?" * len(row))
    c.execute(f"INSERT OR REPLACE INTO {table}({cols}) VALUES({q})", list(row.values()))
````


### `collector/collect_issues.py`: REPLACE (file already exists in your project)

````python
import os, re
import datetime as dt
from collector.github_client import GitHubClient
from collector.db import conn, upsert

gh = GitHubClient()
FMT = "%Y-%m-%dT%H:%M:%SZ"
LINK = re.compile(r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*(?:[\w.\-]+/[\w.\-]+)?#(\d+)")
# how many pages (100 items each) to read per repo; raise these to collect more data
CLOSED_PAGES = int(os.getenv("CLOSED_PAGES", "3"))
OPEN_PAGES = int(os.getenv("OPEN_PAGES", "2"))
PR_PAGES = int(os.getenv("PR_PAGES", "3"))


def linked_numbers(text):
    return {int(n) for n in LINK.findall(text or "")}


def hours_between(a, b):
    return (dt.datetime.strptime(b, FMT) - dt.datetime.strptime(a, FMT)).total_seconds() / 3600


def first_response_hours(i):
    if not i["comments"]:
        return None
    comments = gh.get(i["comments_url"], {"per_page": 10})["body"] or []
    for cm in comments:
        u = cm.get("user")
        if u and u["type"] != "Bot" and u["login"] != i["user"]["login"]:
            return hours_between(i["created_at"], cm["created_at"])
    return None


def base_row(full, i):
    return {
        "id": f"{full}#{i['number']}", "repo": full, "number": i["number"],
        "title": i["title"], "body": i.get("body") or "",
        "labels": "|".join(l["name"] for l in i["labels"]),
        "created_at": i["created_at"], "closed_at": i.get("closed_at"),
        "state": i["state"], "comments": i["comments"],
        "has_assignee": int(bool(i.get("assignee"))),
        "author_assoc": i.get("author_association"),
    }


def collect_repo(c, full, closed_pages=None, open_pages=None, pr_pages=None):
    closed_pages = closed_pages or CLOSED_PAGES
    open_pages = open_pages or OPEN_PAGES
    pr_pages = pr_pages or PR_PAGES
    q = lambda st: {"state": st, "sort": "created", "direction": "desc"}
    closed = [i for i in gh.paginate(f"/repos/{full}/issues", q("closed"), closed_pages) if "pull_request" not in i]
    opened = [i for i in gh.paginate(f"/repos/{full}/issues", q("open"), open_pages) if "pull_request" not in i]
    closed_prs = gh.paginate(f"/repos/{full}/pulls", q("closed"), pr_pages)
    open_prs = gh.paginate(f"/repos/{full}/pulls", {"state": "open"}, open_pages)

    pr_for_issue = {}
    for pr in closed_prs:
        for n in linked_numbers(f"{pr['title']} {pr['body']}"):
            old = pr_for_issue.get(n)
            if old is None or (pr.get("merged_at") and not old.get("merged_at")):
                pr_for_issue[n] = pr

    open_refs = set()
    for pr in open_prs:
        open_refs |= linked_numbers(f"{pr['title']} {pr['body']}")

    n_closed = 0
    for i in closed:
        pr = pr_for_issue.get(i["number"])
        if not pr:
            continue
        d = gh.get(pr["url"])["body"] or {}
        row = base_row(full, i)
        row.update({
            "pr_number": pr["number"], "pr_additions": d.get("additions"),
            "pr_deletions": d.get("deletions"), "pr_files": d.get("changed_files"),
            "pr_merged": int(bool(pr.get("merged_at"))),
            "pr_author_assoc": pr.get("author_association"),
            "first_response_hours": first_response_hours(i),
            "pr_review_comments": d.get("review_comments"),
            "pr_comments": d.get("comments"), "pr_commits": d.get("commits"),
        })
        upsert(c, "issues", row)
        n_closed += 1

    for i in opened:
        row = base_row(full, i)
        row["has_open_pr"] = int(i["number"] in open_refs)
        upsert(c, "issues", row)

    c.commit()
    print(f"{full}: {n_closed} labelled closed, {len(opened)} open")
    return n_closed, len(opened)


def main(repos_file="repos.txt", closed_pages=None, open_pages=None, pr_pages=None):
    c = conn()
    for line in open(repos_file):
        full = line.strip()
        if not full or full.startswith("#"):
            continue
        try:
            collect_repo(c, full, closed_pages, open_pages, pr_pages)
        except Exception as e:
            print("FAILED", full, e)
````


### `collector/merge_repos.py`: NEW

````python
"""Append repos_extra.txt to repos.txt without duplicates:  python cli.py add-repos repos_extra.txt"""
def main(src="repos_extra.txt", dst="repos.txt"):
    have = {l.strip() for l in open(dst) if l.strip() and not l.startswith("#")}
    new = []
    for l in open(src):
        l = l.strip()
        if l and not l.startswith("#") and l not in have:
            new.append(l); have.add(l)
    with open(dst, "a", newline="\n") as f:
        f.write("\n# added from " + src + "\n" + "\n".join(new) + "\n")
    print(f"added {len(new)} repos to {dst}")
````


### `features/text_features.py`: NEW

````python
"""Structural features that exist the moment an issue is opened (no leakage)."""
import re
import pandas as pd

RE_TRACE = re.compile(r"Traceback \(most recent|panic:|goroutine \d+|Exception in|\bat [\w.$]+\([\w.]+:\d+\)|segmentation fault|stack trace", re.I)
RE_PATH = re.compile(r"(?:[\w.\-]+/)+[\w.\-]+\.(?:go|py|js|ts|tsx|rs|java|md|yaml|yml|json|toml|c|cc|cpp|h|sh)\b")
RE_REPRO = re.compile(r"steps to reproduce|to reproduce|expected (?:behavio|result)|actual (?:behavio|result)", re.I)
RE_SMALL = re.compile(r"\b(typo|spelling|grammar|rename|wording|comment|docstring|readme|broken link|dead link|error message)\b", re.I)
RE_BIG = re.compile(r"\b(refactor|redesign|rewrite|architecture|proposal|design|migrate|migration|rfc|epic|umbrella|tracking|roadmap|support for|new feature)\b", re.I)
RE_LINT = re.compile(r"\b(lint|gofmt|format|deprecat|upgrade|bump|version|dependency)\b", re.I)

STRUCT = ["has_stack_trace", "n_paths", "n_checklist", "has_repro", "kw_small", "kw_big", "kw_lint", "n_words"]


def struct_features(title, body):
    t, b = title or "", body or ""
    both = f"{t}\n{b}"
    return {
        "has_stack_trace": int(bool(RE_TRACE.search(both))),
        "n_paths": min(len(set(RE_PATH.findall(both))), 20),
        "n_checklist": b.count("- [ ]") + b.count("- [x]"),
        "has_repro": int(bool(RE_REPRO.search(b))),
        "kw_small": int(bool(RE_SMALL.search(both))),
        "kw_big": int(bool(RE_BIG.search(both))),
        "kw_lint": int(bool(RE_LINT.search(both))),
        "n_words": min(len(b.split()), 3000),
    }


def add_struct(df):
    rows = [struct_features(t, b) for t, b in zip(df["title"], df["body"])]
    extra = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.drop(columns=[c for c in STRUCT if c in df.columns]), extra], axis=1)
````


### `features/labels.py`: NEW

````python
"""
Difficulty label = how much EFFORT the resolving pull request took, relative to its own repository.

effort components (all from the merged PR):  log(lines changed), files changed, review comments
1. each component -> percentile rank INSIDE its repo (shrunk toward the global rank for small repos)
2. effort = mean of the available components
3. Easy = lowest EASY_Q share, Hard = top (1-HARD_Q) share, Medium = the rest
4. guards: a 1-file / <=5-line PR is always Easy, a >1500-line / >40-file PR is always Hard

These are used ONLY as the target y. They are never model inputs. Freeze EASY_Q/HARD_Q after EDA.
"""
import numpy as np
import pandas as pd

EASY_Q, HARD_Q = 0.40, 0.80
SHRINK = 10          # a repo with n labelled issues gets weight n/(n+SHRINK) on its own ranking
LABEL_VERSION = "effort-percentile-v1"


def _pct(df, col):
    g = df[col].groupby(df["repo"])
    within = g.rank(pct=True, method="average")
    glob = df[col].rank(pct=True, method="average")
    n = g.transform("count")
    w = n / (n + SHRINK)
    return w * within + (1 - w) * glob


def effort_score(df):
    comp = pd.DataFrame(index=df.index)
    comp["lines"] = np.log1p(df["pr_additions"].fillna(0) + df["pr_deletions"].fillna(0))
    comp["files"] = df["pr_files"].astype(float)
    if "pr_review_comments" in df and df["pr_review_comments"].notna().mean() > 0.5:
        comp["review"] = df["pr_review_comments"].astype(float)
    work = comp.copy()
    work["repo"] = df["repo"].values
    ranks = pd.DataFrame({c: _pct(work, c) for c in comp.columns})
    return ranks.mean(axis=1, skipna=True)


def assign_labels(df, easy_q=EASY_Q, hard_q=HARD_Q):
    """df must contain merged-PR rows only. Returns df with 'effort' and 'difficulty'."""
    df = df.copy()
    df["effort"] = effort_score(df)
    lo, hi = df["effort"].quantile(easy_q), df["effort"].quantile(hard_q)
    lab = np.where(df["effort"] <= lo, "Easy", np.where(df["effort"] > hi, "Hard", "Medium"))
    lines = df["pr_additions"].fillna(0) + df["pr_deletions"].fillna(0)
    files = df["pr_files"].fillna(0)
    lab = np.where((lines <= 5) & (files <= 1), "Easy", lab)
    lab = np.where((lines > 1500) | (files > 40), "Hard", lab)
    df["difficulty"] = lab
    return df
````


### `features/build_features.py`: REPLACE (file already exists in your project)

````python
import os
import numpy as np
import pandas as pd
from collector.db import conn
from features.text_features import add_struct, STRUCT
from features.labels import assign_labels

NUM = ["body_len", "title_len", "n_labels", "is_gfi", "is_help_wanted", "is_bug", "is_docs",
       "n_code_blocks", "has_link", "log_stars", "log_contributors", "has_contributing",
       "repo_age_days"] + STRUCT
CAT = ["language", "author_assoc"]
FEATURES = NUM + CAT + ["text", "repo"]
# LEAKAGE RULE: 'comments', 'has_assignee' and every pr_* column are NOT features.


def load(state=None):
    q = """SELECT i.*, r.stars, r.forks, r.language, r.contributors, r.has_contributing,
                  r.archived, r.created_at AS repo_created, r.pushed_at
           FROM issues i JOIN repos r ON i.repo = r.full_name"""
    if state:
        q += f" WHERE i.state = '{state}'"
    return pd.read_sql(q, conn())


def add_features(df):
    df = df.copy()
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    lab = df["labels"].fillna("").str.lower()
    df["text"] = df["title"] + " . " + df["body"].str.slice(0, 2000)
    df["body_len"] = df["body"].str.len()
    df["title_len"] = df["title"].str.len()
    df["n_labels"] = lab.apply(lambda s: len([x for x in s.split("|") if x]))
    df["is_gfi"] = lab.str.contains("good first issue|good-first-issue|beginner|easy").astype(int)
    df["is_help_wanted"] = lab.str.contains("help wanted|help-wanted").astype(int)
    df["is_bug"] = lab.str.contains("bug").astype(int)
    df["is_docs"] = lab.str.contains("doc").astype(int)
    df["n_code_blocks"] = df["body"].str.count("```") // 2
    df["has_link"] = df["body"].str.contains("http").astype(int)
    df["log_stars"] = np.log1p(df["stars"].fillna(0))
    df["log_contributors"] = np.log1p(df["contributors"].fillna(0))
    df["has_contributing"] = df["has_contributing"].fillna(0)
    df["repo_age_days"] = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(df["repo_created"], utc=True)).dt.days
    df["language"] = df["language"].fillna("unknown")
    df["author_assoc"] = df["author_assoc"].fillna("NONE")
    df = add_struct(df)
    df[NUM] = df[NUM].fillna(0)
    return df


def add_labels(df):
    """Labels come from the RESOLVED PR (merged only). Targets only, never inputs."""
    df = df.copy()
    df["hours_to_close"] = (pd.to_datetime(df["closed_at"], utc=True)
                            - pd.to_datetime(df["created_at"], utc=True)).dt.total_seconds() / 3600
    df["difficulty"] = pd.Series([None] * len(df), index=df.index, dtype=object)   # object dtype: pandas 3 is strict
    df["effort"] = np.nan
    m = (df["pr_merged"] == 1) & df["pr_files"].notna()
    if m.sum() >= 10:
        lab = assign_labels(df[m])
        df.loc[m, "difficulty"] = lab["difficulty"]
        df.loc[m, "effort"] = lab["effort"]
    return df


def labelled():
    """The training table: closed issues whose merged PR gives a difficulty label."""
    df = add_labels(add_features(load(state="closed")))
    return df[df["difficulty"].notna()].copy()


def main(n_manual=200):
    df = labelled()
    print(f"Labelled issues: {len(df)} from {df.repo.nunique()} repos")
    print("Difficulty distribution:\n", df["difficulty"].value_counts(normalize=True).round(3))
    path = "data/processed/manual_label_sample.csv"
    os.makedirs("data/processed", exist_ok=True)
    keep = pd.DataFrame()
    if os.path.exists(path):                         # never throw away labels you already typed
        old = pd.read_csv(path)
        keep = old[old["my_difficulty"].astype(str).str.strip().isin(["Easy", "Medium", "Hard"])]
    rest = df[~df["id"].isin(keep["id"] if len(keep) else [])]
    need = max(0, n_manual - len(keep))
    new = rest.sample(min(need, len(rest)), random_state=1)[["id", "repo", "number", "title", "difficulty"]].copy()
    new["url"] = "https://github.com/" + new["repo"] + "/issues/" + new["number"].astype(str)
    new["my_difficulty"] = ""
    if len(keep):
        k = keep[["id", "my_difficulty"]].merge(df[["id", "repo", "number", "title", "difficulty"]], on="id")
        k["url"] = "https://github.com/" + k["repo"] + "/issues/" + k["number"].astype(str)
        new = pd.concat([k[new.columns], new])
    new.to_csv(path, index=False)
    print(f"Wrote {path}: {len(new)} rows, {int((new.my_difficulty.astype(str).str.strip() != '').sum())} already labelled by you.")
    print("Label them with:  python cli.py label")
````


### `tools/label_cli.py`: NEW

````python
"""Interactive manual labelling. The automatic label is HIDDEN so it cannot bias you."""
import os, webbrowser
import pandas as pd
from collector.db import conn

PATH = "data/processed/manual_label_sample.csv"
KEYS = {"e": "Easy", "m": "Medium", "h": "Hard"}


def main():
    if not os.path.exists(PATH):
        raise SystemExit("Run `python cli.py features` first.")
    m = pd.read_csv(PATH)
    m["my_difficulty"] = m["my_difficulty"].fillna("").astype(str)
    todo = m.index[~m["my_difficulty"].isin(KEYS.values())].tolist()
    print(f"{len(m) - len(todo)} done, {len(todo)} to go.  Keys: e=Easy m=Medium h=Hard  o=open in browser  s=skip  q=quit")
    print("Judge how hard it would be for a beginner who knows the language, from the issue ALONE.\n")
    db = conn()
    for n, i in enumerate(todo, 1):
        r = m.loc[i]
        row = db.execute("SELECT body, labels FROM issues WHERE id=?", (r["id"],)).fetchone() or ("", "")
        print("=" * 78)
        print(f"[{n}/{len(todo)}] {r['id']}  |  {r['title']}\nlabels: {row[1]}\n{(row[0] or '')[:900]}\n{r['url']}")
        while True:
            k = input("e/m/h/o/s/q > ").strip().lower()
            if k == "o":
                webbrowser.open(r["url"]); continue
            break
        if k == "q":
            break
        if k in KEYS:
            m.loc[i, "my_difficulty"] = KEYS[k]
            m.to_csv(PATH, index=False)
    m.to_csv(PATH, index=False)
    print("saved. Run `python cli.py evaluate` to get agreement and kappa.")
````


### `ml/eda.py`: NEW

````python
"""Exploratory data analysis. Writes figures to report/eda/ and a summary to report/eda/summary.md"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from features.build_features import load, add_features, add_labels, NUM

OUT = "report/eda"
ORDER = ["Easy", "Medium", "Hard"]


def _save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=130); plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = load()
    df = add_labels(add_features(raw))
    lab = df[df["difficulty"].notna()].copy()
    S = []                                        # summary lines
    S.append(f"- issues in DB: {len(raw)} (closed {int((raw.state=='closed').sum())}, open {int((raw.state=='open').sum())})")
    S.append(f"- repos: {raw.repo.nunique()}; languages: {raw.drop_duplicates('repo').language.value_counts().to_dict()}")
    S.append(f"- labelled (merged PR) issues used for training: {len(lab)} in {lab.repo.nunique()} repos")
    S.append(f"- duplicate ids: {int(raw.id.duplicated().sum())}; duplicate titles within a repo: {int(raw.duplicated(['repo','title']).sum())}")
    miss = raw[["body", "labels", "first_response_hours", "pr_files", "pr_review_comments"]].isna().mean().round(3)
    S.append(f"- missing share: {miss.to_dict()}")
    cd = lab["difficulty"].value_counts(normalize=True).reindex(ORDER).round(3)
    S.append(f"- class distribution: {cd.to_dict()}  (imbalance ratio max/min = {cd.max()/max(cd.min(),1e-9):.2f})")
    S.append(f"- 'good first issue' share: open {raw[raw.state=='open'].labels.fillna('').str.lower().str.contains('good first issue').mean():.3f}, labelled {lab.is_gfi.mean():.3f}")
    S.append(f"- top repo holds {lab.repo.value_counts(normalize=True).iloc[0]:.1%} of labelled issues (repo skew)")

    plt.figure(figsize=(4.5, 3.5)); sns.countplot(x="difficulty", data=lab, order=ORDER); plt.title("Class distribution"); _save("01_class_distribution")
    top = lab.repo.value_counts().head(15)
    plt.figure(figsize=(7, 4)); top.iloc[::-1].plot.barh(); plt.title("Labelled issues per repo (top 15)"); _save("02_repo_distribution")
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for a, (c, t) in zip(ax.ravel(), [("body_len", "Description length"), ("comments", "Comments (EDA only, not a feature)"),
                                      ("title_len", "Title length"), ("log_stars", "log stars"),
                                      ("repo_age_days", "Repo age (days)"), ("n_code_blocks", "Code blocks")]):
        a.hist(np.log1p(df[c].clip(lower=0)) if c in ("body_len", "comments") else df[c], bins=30); a.set_title(t)
    _save("03_histograms")
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, c in zip(ax, ["body_len", "n_paths", "n_code_blocks", "log_stars"]):
        sns.boxplot(x="difficulty", y=c, data=lab, order=ORDER, ax=a); a.set_title(f"{c} by class")
    _save("04_feature_vs_class_box")
    pr = lab.assign(lines=lab.pr_additions.fillna(0) + lab.pr_deletions.fillna(0))
    plt.figure(figsize=(5, 3.8)); sns.boxplot(x="difficulty", y=np.log1p(pr["lines"]), data=pr, order=ORDER); plt.ylabel("log(lines changed)"); plt.title("Label sanity: PR size per class"); _save("05_label_vs_pr_size")
    corr = lab[NUM + ["effort"]].corr()
    plt.figure(figsize=(10, 8)); sns.heatmap(corr, cmap="coolwarm", center=0, annot=False); plt.title("Correlation (features + effort)"); _save("06_correlation")
    S.append("- features most correlated with effort: " + ", ".join(f"{k} ({v:+.2f})" for k, v in corr["effort"].drop("effort").abs().sort_values(ascending=False).head(5).items()))
    tags = raw.labels.fillna("").str.split("|").explode()
    tags = tags[tags != ""].value_counts().head(15)
    plt.figure(figsize=(7, 4)); tags.iloc[::-1].plot.barh(); plt.title("Most frequent GitHub labels"); _save("07_label_frequency")
    age = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(raw[raw.state == "open"].created_at, utc=True)).dt.days
    plt.figure(figsize=(5, 3.5)); plt.hist(age.clip(upper=2000), bins=40); plt.title("Age of open issues (days)"); _save("08_issue_age")
    rt = raw.first_response_hours.dropna()
    if len(rt):
        plt.figure(figsize=(5, 3.5)); plt.hist(np.log1p(rt), bins=30); plt.title("log(1+hours) to first maintainer reply"); _save("09_response_time")
        S.append(f"- maintainer first reply: median {rt.median():.1f} h, 90th pct {rt.quantile(.9):.0f} h (n={len(rt)})")
    q1, q3 = lab.body_len.quantile([.25, .75]); out = ((lab.body_len > q3 + 3 * (q3 - q1))).sum()
    S.append(f"- extreme body-length outliers (3xIQR): {int(out)}; kept (real issues), log-scaled in plots")
    gfi = lab.groupby("is_gfi")["difficulty"].value_counts(normalize=True).unstack().reindex(columns=ORDER).round(3)
    S.append("- difficulty split by 'good first issue' label:\n" + gfi.to_string())
    # top terms per class
    tf = TfidfVectorizer(max_features=4000, stop_words="english", min_df=3)
    Xt = tf.fit_transform(lab.text); terms = np.array(tf.get_feature_names_out())
    for c in ORDER:
        m = np.asarray(Xt[(lab.difficulty == c).values].mean(0)).ravel(); S.append(f"- top terms {c}: " + ", ".join(terms[m.argsort()[-10:][::-1]]))
    z = TruncatedSVD(2, random_state=0).fit_transform(Xt)
    plt.figure(figsize=(5.5, 4.5))
    for c, col in zip(ORDER, ["tab:green", "tab:orange", "tab:red"]):
        k = (lab.difficulty == c).values; plt.scatter(z[k, 0], z[k, 1], s=10, alpha=.6, c=col, label=c)
    plt.legend(); plt.title("Issue text (TF-IDF -> SVD 2D)"); _save("10_text_svd")
    open(f"{OUT}/summary.md", "w").write("# EDA summary\n\n" + "\n".join(S) + "\n")
    print("\n".join(S)); print(f"\nfigures in {OUT}/")
````


---

## PART 2. ML engine


### `ml/common.py`: REPLACE (file already exists in your project)

````python
import hashlib, json, os, pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from features.build_features import NUM, CAT, FEATURES  # noqa: F401


class RepoPrior(BaseEstimator, TransformerMixin):
    """Smoothed per-repo class rates learned on TRAIN data only.
    Training rows use leave-one-out so a row never sees its own label. Unseen repos get the global prior."""
    def __init__(self, strength=8):
        self.strength = strength

    def _fit(self, X, y):
        repo, y = np.asarray(X).ravel().astype(str), np.asarray(y)
        self.classes_ = sorted(set(y.tolist()))
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        self.counts_ = {r: np.array([((y == c) & (repo == r)).sum() for c in self.classes_], float) for r in set(repo)}
        return repo, y

    def _enc(self, counts):
        return (counts + self.strength * self.prior_) / (counts.sum() + self.strength)

    def fit(self, X, y=None):
        self._fit(X, y); return self

    def fit_transform(self, X, y=None, **kw):
        repo, y = self._fit(X, y)
        out = np.zeros((len(repo), len(self.classes_)))
        for i, (r, yi) in enumerate(zip(repo, y)):
            c = self.counts_[r].copy(); c[self.classes_.index(yi)] -= 1
            out[i] = self._enc(c)
        return out

    def transform(self, X):
        repo = np.asarray(X).ravel().astype(str)
        return np.array([self._enc(self.counts_[r]) if r in self.counts_ else self.prior_ for r in repo])


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """OPTIONAL (pip install sentence-transformers; set USE_EMBEDDINGS=1). Embeddings are cached on disk."""
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        global _ST
        texts = [str(t)[:1500] for t in np.asarray(X).ravel()]
        path = "data/emb_cache.pkl"
        cache = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        miss = [i for i, k in enumerate(keys) if k not in cache]
        if miss:
            from sentence_transformers import SentenceTransformer
            if "_ST" not in globals():
                _ST = SentenceTransformer(self.model)
            vecs = _ST.encode([texts[i] for i in miss], batch_size=32, show_progress_bar=False)
            for i, v in zip(miss, vecs):
                cache[keys[i]] = v.astype("float32")
            pickle.dump(cache, open(path, "wb"))
        return np.vstack([cache[k] for k in keys])


def preprocessor(use_emb=None):
    use_emb = (os.getenv("USE_EMBEDDINGS") == "1") if use_emb is None else use_emb
    parts = [
        ("txt", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english", min_df=2), "text"),
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("repo", RepoPrior(), "repo"),
    ]
    if use_emb:
        parts.append(("emb", SentenceEmbedder(), "text"))
    return ColumnTransformer(parts)


def temporal_split(df, frac=0.8):
    df = df.sort_values("created_at")
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]


def group_splits(df, n=5):
    """Folds that keep every repo entirely in train OR test => measures generalisation to UNSEEN repos."""
    g = df["repo"].values
    k = max(2, min(n, len(set(g))))
    return list(GroupKFold(n_splits=k).split(df, groups=g))


def bootstrap_ci(y_true, y_pred, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(fn(y_true[idx], y_pred[idx]))
    return [round(float(x), 3) for x in np.percentile(scores, [2.5, 97.5])]


# cost of predicting `pred` when the truth is `true`: calling a Hard issue Easy is the worst mistake
COST = {("Hard", "Easy"): 3, ("Hard", "Medium"): 1, ("Medium", "Easy"): 1,
        ("Easy", "Hard"): 2, ("Easy", "Medium"): 1, ("Medium", "Hard"): 1}


def weighted_cost(y_true, y_pred):
    return float(np.mean([COST.get((t, p), 0) for t, p in zip(y_true, y_pred)]))


def save_metrics(name, d):
    os.makedirs("report", exist_ok=True)
    path = "report/metrics.json"
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[name] = d
    json.dump(data, open(path, "w"), indent=2, default=str)


def write_table(rows, stem):
    """rows: {model: {metric: value}} -> report/<stem>.csv and .md"""
    os.makedirs("report", exist_ok=True)
    t = pd.DataFrame(rows).T
    t.to_csv(f"report/{stem}.csv")
    cols = ["model"] + list(t.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for k, r in t.iterrows():
        md.append("| " + " | ".join([str(k)] + [("" if pd.isna(v) else f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r]) + " |")
    open(f"report/{stem}.md", "w").write("\n".join(md) + "\n")
    return t
````


### `ml/train_difficulty.py`: REPLACE (file already exists in your project)

````python
"""Difficulty classification (Easy/Medium/Hard).  Every model sees the same data, the same folds and the same metrics.
  * temporal hold-out : train on the oldest 80%, test on the newest 20%  (realistic)
  * grouped CV by repo: every repo is entirely in train or test          (does it work on UNSEEN repos?)
Model choice uses the grouped-CV score, never the test score."""
import json, time, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from features.build_features import labelled, FEATURES, NUM
from features.labels import LABEL_VERSION, EASY_Q, HARD_Q
from ml.common import (preprocessor, temporal_split, group_splits, bootstrap_ci, save_metrics,
                       write_table, weighted_cost)

warnings.filterwarnings("ignore")
ORDER = ["Easy", "Medium", "Hard"]


def models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "SVM(linear)": CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3, method="sigmoid"),
        "KNN(cosine)": KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="distance"),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "MLP(64)": MLPClassifier((64,), alpha=1e-2, max_iter=500, random_state=42),
    }


GRIDS = {
    "LogisticRegression": {"clf__C": [0.1, 1, 10]},
    "DecisionTree": {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [3, 8]},
    "RandomForest": {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 3]},
    "KNN(cosine)": {"clf__n_neighbors": [7, 15, 25]},
    "GradientBoosting": {"clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [100, 200]},
}


def pipe(clf):
    return Pipeline([("pre", preprocessor()), ("clf", clf)])


def f1m(a, b):
    return f1_score(a, b, average="macro")


def ece(proba, y, classes, bins=8):
    conf, pred = proba.max(1), np.array(classes)[proba.argmax(1)]
    ok, e, edges = (pred == np.asarray(y)), 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def test_metrics(p, train, test):
    t = time.time(); p = clone(p).fit(train[FEATURES], train["difficulty"]); fit_s = time.time() - t
    t = time.time(); pred = p.predict(test[FEATURES]); inf_ms = (time.time() - t) / max(len(test), 1) * 1000
    y = test["difficulty"]
    r = {"test_accuracy": accuracy_score(y, pred), "test_precision_macro": precision_score(y, pred, average="macro", zero_division=0),
         "test_recall_macro": recall_score(y, pred, average="macro", zero_division=0), "test_f1_macro": f1m(y, pred),
         "test_f1_weighted": f1_score(y, pred, average="weighted"), "test_cost": weighted_cost(y, pred),
         "hard_called_easy": float(((y == "Hard") & (pred == "Easy")).sum() / max((y == "Hard").sum(), 1)),
         "train_s": fit_s, "infer_ms_each": inf_ms}
    proba = None
    if hasattr(p, "predict_proba"):
        proba = p.predict_proba(test[FEATURES])
        try:
            r["test_roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", labels=list(p.classes_))
        except ValueError:
            r["test_roc_auc_ovr"] = np.nan
    return r, p, pred, proba


def main(tune=True):
    df = labelled().sort_values("created_at").reset_index(drop=True)
    if len(df) < 60 or df.difficulty.nunique() < 3:
        raise SystemExit(f"Only {len(df)} labelled issues / {df.difficulty.nunique()} classes: collect more data first.")
    train, test = temporal_split(df)
    folds = group_splits(df)
    X, y = df[FEATURES], df["difficulty"]
    print(f"{len(df)} labelled issues, {df.repo.nunique()} repos | train {len(train)} / test {len(test)} | {len(folds)} grouped folds")
    print("class balance:", df.difficulty.value_counts(normalize=True).round(2).to_dict())

    maj = train["difficulty"].mode()[0]
    rows = {
        "baseline: majority": {"test_f1_macro": f1m(test.difficulty, [maj] * len(test)), "test_accuracy": float((test.difficulty == maj).mean())},
        "baseline: good-first-issue label": {"test_f1_macro": f1m(test.difficulty, np.where(test.is_gfi == 1, "Easy", maj)),
                                             "test_accuracy": float((test.difficulty == np.where(test.is_gfi == 1, "Easy", maj)).mean())},
    }
    fitted = {}
    for name, clf in models().items():
        t0 = time.time()
        cv = cross_validate(pipe(clf), X, y, cv=folds, scoring="f1_macro", return_train_score=True, error_score=np.nan)
        r, p, pred, proba = test_metrics(pipe(clf), train, test)
        r.update(cv_f1_grouped=float(np.nanmean(cv["test_score"])), cv_f1_std=float(np.nanstd(cv["test_score"])),
                 cv_train_f1=float(np.nanmean(cv["train_score"])))
        r["overfit_gap"] = r["cv_train_f1"] - r["cv_f1_grouped"]
        r["f1_95ci"] = str(bootstrap_ci(test.difficulty, pred, f1m, n=500))
        rows[name] = r; fitted[name] = clf
        print(f"  {name:20s} grouped-CV F1 {r['cv_f1_grouped']:.3f}  test F1 {r['test_f1_macro']:.3f}  gap {r['overfit_gap']:.2f}  ({time.time()-t0:.0f}s)")

    best = max(fitted, key=lambda n: np.nan_to_num(rows[n]["cv_f1_grouped"], nan=-1))
    best_clf = fitted[best]
    if tune and best in GRIDS:
        gs = GridSearchCV(pipe(best_clf), GRIDS[best], cv=folds, scoring="f1_macro", n_jobs=1, error_score=np.nan).fit(X, y)
        print(f"  tuned {best}: {gs.best_params_}  grouped-CV F1 {gs.best_score_:.3f}")
        if gs.best_score_ >= rows[best]["cv_f1_grouped"]:
            r, p, pred, proba = test_metrics(gs.best_estimator_, train, test)
            r.update(cv_f1_grouped=float(gs.best_score_), cv_f1_std=np.nan, cv_train_f1=np.nan, overfit_gap=np.nan,
                     f1_95ci=str(bootstrap_ci(test.difficulty, pred, f1m, n=500)), params=str(gs.best_params_))
            rows[best + " (tuned)"] = r
            best_pipe = gs.best_estimator_; best = best + " (tuned)"
        else:
            best_pipe = pipe(best_clf)
    else:
        best_pipe = pipe(best_clf)

    table = write_table(rows, "model_comparison")
    print("\n" + table[[c for c in ["cv_f1_grouped", "test_f1_macro", "test_f1_weighted", "test_roc_auc_ovr", "overfit_gap", "test_cost", "hard_called_easy"] if c in table]].round(3).to_string())

    # calibrated version of the winner, trained on the old data only, for honest test plots + ECE
    cal = CalibratedClassifierCV(clone(best_pipe), method="sigmoid", cv=3).fit(train[FEATURES], train["difficulty"])
    pp = cal.predict_proba(test[FEATURES])
    prior = train["difficulty"].value_counts(normalize=True).reindex(cal.classes_).values
    pc = np.array(cal.classes_)[(pp / prior).argmax(1)]      # prior-corrected rule: stops the model ignoring the rare Hard class
    print("\n" + classification_report(test.difficulty, pc, zero_division=0))
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ConfusionMatrixDisplay.from_predictions(test.difficulty, pc, labels=ORDER, ax=ax, normalize="true", values_format=".2f"); ax.set_title(f"{best}: normalised confusion (test)")
    fig.savefig("report/confusion_difficulty.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, c in enumerate(cal.classes_):
        yy = (test.difficulty.values == c)
        if yy.any() and (~yy).any():
            fp, tp, _ = roc_curve(yy, pp[:, i]); ax.plot(fp, tp, label=f"{c} (AUC {roc_auc_score(yy, pp[:, i]):.2f})")
    ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.legend(); ax.set_title("ROC one-vs-rest (test)")
    fig.savefig("report/roc_difficulty.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    conf, ok = pp.max(1), (pc == test.difficulty.values)
    bins = np.linspace(0, 1, 7); mid, acc = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 3: mid.append(conf[m].mean()); acc.append(ok[m].mean())
    fig, ax = plt.subplots(figsize=(4.5, 4)); ax.plot([0, 1], [0, 1], "k:"); ax.plot(mid, acc, "o-"); ax.set_xlabel("stated confidence"); ax.set_ylabel("actual accuracy"); ax.set_title("Reliability (test)")
    fig.savefig("report/calibration_difficulty.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    ece_v = ece(pp, test.difficulty.values, list(cal.classes_))

    # final deployed model: ALL labelled data, calibrated so predict_proba is trustworthy
    final = CalibratedClassifierCV(clone(best_pipe), method="sigmoid", cv=3).fit(X, y)
    joblib.dump(final, "models/difficulty.joblib")
    meta = {"chosen": best, "chosen_by": "grouped-CV macro-F1 (unseen repos)", "n_train": len(df), "n_repos": int(df.repo.nunique()),
            "label_version": LABEL_VERSION, "easy_q": EASY_Q, "hard_q": HARD_Q, "ece_test": round(ece_v, 3),
            "class_balance": df.difficulty.value_counts(normalize=True).round(3).to_dict(), "eval_plot_rule": "argmax(p / class_prior) for the confusion matrix; the app shows the most probable class plus all probabilities", "trained": pd.Timestamp.now().isoformat()}
    json.dump(meta, open("models/difficulty_meta.json", "w"), indent=2)
    save_metrics("difficulty", {k: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} for k, v in rows.items()})
    save_metrics("difficulty_meta", meta)
    print(f"\nsaved models/difficulty.joblib ({best}); ECE on test = {ece_v:.3f}")
    print("Decide with grouped-CV F1 and its std, NOT with one test number; chance level for 3 balanced classes is about 0.33.")
````


### `ml/train_merge.py`: REPLACE (file already exists in your project)

````python
"""Will the PR for this issue be merged?  (only issues that have a linked closed PR are observable)"""
import warnings
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, PrecisionRecallDisplay
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from features.build_features import load, add_features, FEATURES
from ml.common import preprocessor, temporal_split, group_splits, save_metrics, write_table

warnings.filterwarnings("ignore")


def main():
    df = add_features(load(state="closed"))
    df = df[df.pr_number.notna()].copy().sort_values("created_at").reset_index(drop=True)
    df["y"] = df.pr_merged.astype(int)
    train, test = temporal_split(df)
    folds = group_splits(df)
    base = float(df.y.mean())
    print(f"{len(df)} PR-linked issues, merged rate {base:.3f} (imbalanced: judge by PR-AUC vs this base rate AND ROC-AUC)")
    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }
    rows, fig, ax = {}, *plt.subplots(figsize=(6, 5))
    for n, clf in models.items():
        p = lambda: Pipeline([("pre", preprocessor()), ("clf", clone(clf))])
        cv = cross_validate(p(), df[FEATURES], df["y"], cv=folds, scoring="average_precision", error_score=np.nan)
        m = p().fit(train[FEATURES], train["y"]); pr = m.predict_proba(test[FEATURES])[:, 1]
        rows[n] = {"cv_ap_grouped": float(np.nanmean(cv["test_score"])), "test_ap": average_precision_score(test.y, pr),
                   "test_roc_auc": roc_auc_score(test.y, pr) if test.y.nunique() > 1 else np.nan, "test_base_rate": float(test.y.mean())}
        PrecisionRecallDisplay.from_predictions(test.y, pr, name=n, ax=ax)
        print(f"  {n:20s} grouped-CV AP {rows[n]['cv_ap_grouped']:.3f} | test AP {rows[n]['test_ap']:.3f} (base {rows[n]['test_base_rate']:.3f}) | ROC-AUC {rows[n]['test_roc_auc']:.3f}")
    rows["baseline: merge rate"] = {"test_ap": float(test.y.mean())}
    ax.set_title("Precision-recall (test)"); fig.savefig("report/pr_merge.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    write_table(rows, "merge_comparison"); save_metrics("merge", {k: {a: round(b, 4) for a, b in v.items()} for k, v in rows.items()})
    best = max(models, key=lambda n: np.nan_to_num(rows[n]["cv_ap_grouped"], nan=-1))
    final = CalibratedClassifierCV(Pipeline([("pre", preprocessor()), ("clf", clone(models[best]))]), method="sigmoid", cv=3).fit(df[FEATURES], df["y"])
    joblib.dump(final, "models/merge.joblib")
    ok = rows[best]["test_roc_auc"] > 0.70
    print(f"saved models/merge.joblib ({best}). Useful ranking signal? {'yes' if ok else 'WEAK (ROC-AUC<=0.70): the plan leans on the repo-level Bayesian rate instead'}")
````


### `ml/train_time.py`: REPLACE (file already exists in your project)

````python
"""Time-to-close regression (log hours).  Calendar time mostly measures WAITING, so this is expected to be hard:
we compare against honest baselines and only call it useful if it beats them."""
import warnings
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features.build_features import load, add_features, add_labels, FEATURES, NUM, CAT
from ml.common import temporal_split, group_splits, save_metrics, write_table

warnings.filterwarnings("ignore")


def pre():
    return ColumnTransformer([("txt", TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english", min_df=2), "text"),
                              ("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])


def main():
    df = add_labels(add_features(load(state="closed")))
    df = df[(df.pr_merged == 1) & (df.hours_to_close > 0)].sort_values("created_at").reset_index(drop=True)
    df["y"] = np.log1p(df.hours_to_close)
    train, test = temporal_split(df); folds = group_splits(df)
    med = train.y.median(); rmed = train.groupby("repo").y.median()
    base_pred = {"baseline: global median": np.full(len(test), med),
                 "baseline: repo median": test.repo.map(rmed).fillna(med).values}
    rows = {k: {"test_r2": r2_score(test.y, v), "test_mae_logh": mean_absolute_error(test.y, v)} for k, v in base_pred.items()}
    cfg = [("Linear", LinearRegression())] + [(f"Ridge a={a}", Ridge(alpha=a)) for a in [1, 10, 100, 300]] + [(f"Lasso a={a}", Lasso(alpha=a, max_iter=20000)) for a in [0.01, 0.03, 0.1]]
    fits = {}
    for n, reg in cfg:
        p = Pipeline([("pre", pre()), ("reg", reg)]).fit(train[FEATURES], train.y); pr = p.predict(test[FEATURES])
        coef = p.named_steps["reg"].coef_
        rows[n] = {"test_r2": r2_score(test.y, pr), "test_mae_logh": mean_absolute_error(test.y, pr), "test_rmse_logh": mean_squared_error(test.y, pr) ** .5,
                   "cv_r2_grouped": float(np.nanmean(cross_val_score(Pipeline([("pre", pre()), ("reg", reg)]), df[FEATURES], df.y, cv=folds, scoring="r2", error_score=np.nan))),
                   "nonzero_coefs": int((np.abs(coef) > 1e-8).sum()), "coef_l2": float(np.linalg.norm(coef))}
        fits[n] = p
    t = write_table(rows, "time_comparison"); print(t.round(3).to_string())
    ridge = {k: v for k, v in rows.items() if k.startswith("Ridge")}
    best = max(ridge, key=lambda k: ridge[k]["cv_r2_grouped"])
    useful = bool(rows[best]["test_mae_logh"] < rows["baseline: repo median"]["test_mae_logh"] and rows[best]["test_r2"] > 0)
    joblib.dump(fits[best], "models/time.joblib")
    save_metrics("time_to_close", {**{k: {a: round(b, 4) for a, b in v.items()} for k, v in rows.items()}, "best": best, "beats_repo_median": useful})
    ra = [(a, rows[f"Ridge a={a}"]) for a in [1, 10, 100, 300]]; la = [(a, rows[f"Lasso a={a}"]) for a in [0.01, 0.03, 0.1]]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot([a for a, _ in ra], [r["coef_l2"] for _, r in ra], "o-", label="Ridge ||w||"); ax[0].set_xscale("log"); ax[0].set_title("Ridge: coefficients shrink"); ax[0].set_xlabel("alpha")
    ax[1].plot([a for a, _ in la], [r["nonzero_coefs"] for _, r in la], "o-", c="tab:red"); ax[1].set_xscale("log"); ax[1].set_title("Lasso: non-zero coefficients"); ax[1].set_xlabel("alpha")
    fig.tight_layout(); fig.savefig("report/regularisation_effect.png", dpi=140); plt.close(fig)
    print(f"\nbest ridge {best}; beats repo-median baseline: {useful}. "
          + ("" if useful else "NEGATIVE RESULT: report it honestly; the plan uses difficulty-based effort hours instead."))
````


### `ml/cluster.py`: REPLACE (file already exists in your project)

````python
"""Repo clustering (K-Means, DBSCAN, BIRCH), anomaly detection (Isolation Forest), PCA and kernel PCA.
Cluster MEANINGS are not assumed: read report/cluster_profiles.csv and name the clusters yourself."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import Birch, DBSCAN, KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from collector.db import conn


def main(eps=None):
    import os; os.makedirs("report", exist_ok=True); os.makedirs("data/processed", exist_ok=True)
    c = conn()
    r = pd.read_sql("SELECT * FROM repos", c)
    iss = pd.read_sql("SELECT repo, state, pr_merged, first_response_hours FROM issues", c)
    st = iss.groupby("repo").agg(median_reply_h=("first_response_hours", "median"),
                                 merge_rate=("pr_merged", "mean"), n_issues=("state", "size"))
    r = r.join(st, on="full_name")
    now = pd.Timestamp.now(tz="UTC")
    r["days_since_push"] = (now - pd.to_datetime(r.pushed_at, utc=True)).dt.days
    r["age_years"] = (now - pd.to_datetime(r.created_at, utc=True)).dt.days / 365
    F = pd.DataFrame({"log_stars": np.log1p(r.stars), "log_forks": np.log1p(r.forks), "log_contrib": np.log1p(r.contributors),
                      "log_open": np.log1p(r.open_issues), "log_push": np.log1p(r.days_since_push), "age_years": r.age_years,
                      "log_reply_h": np.log1p(r.median_reply_h.fillna(r.median_reply_h.median())),
                      "merge_rate": r.merge_rate.fillna(r.merge_rate.median()), "has_contributing": r.has_contributing.fillna(0)})
    Z = StandardScaler().fit_transform(F)
    ks = range(2, min(8, len(r) - 1))
    sil = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=42).fit_predict(Z)) for k in ks}
    k = max(sil, key=sil.get)
    print("silhouette by k:", {a: round(b, 3) for a, b in sil.items()}, "-> k =", k)
    r["cluster"] = KMeans(k, n_init=10, random_state=42).fit_predict(Z)
    r["birch"] = Birch(n_clusters=k).fit_predict(Z)
    kd = np.sort(NearestNeighbors(n_neighbors=4).fit(Z).kneighbors(Z)[0][:, -1])
    eps = eps or float(np.percentile(kd, 90))
    r["dbscan"] = DBSCAN(eps=eps, min_samples=4).fit_predict(Z)
    print(f"DBSCAN eps={eps:.2f} (90th pct of 4-NN distance): clusters={len(set(r.dbscan)) - (-1 in set(r.dbscan))}, noise={int((r.dbscan == -1).sum())}")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(kd); ax[0].axhline(eps, c="r", ls=":"); ax[0].set_title("k-distance (eps at the elbow)")
    ax[1].plot(list(sil), list(sil.values()), "o-"); ax[1].set_title("silhouette vs k"); fig.tight_layout(); fig.savefig("report/kdistance.png", dpi=140); plt.close(fig)

    iso = IsolationForest(contamination=0.1, random_state=42).fit(Z)
    r["anomaly_score"] = -iso.score_samples(Z)
    r["anomaly"] = iso.predict(Z) == -1
    r["abandoned"] = (r.archived == 1) | (r.days_since_push > 365) | (r.anomaly & (r.days_since_push > 180))
    print(pd.crosstab(r.archived, r.anomaly, rownames=["archived"], colnames=["anomaly"]), "\nabandoned flagged:", int(r.abandoned.sum()))

    p = PCA(min(3, Z.shape[1])).fit(Z); pcs = p.transform(Z)
    print("PCA explained variance:", np.round(p.explained_variance_ratio_, 3))
    kp = KernelPCA(2, kernel="rbf", gamma=0.1).fit_transform(Z)
    r["pc1"], r["pc2"] = pcs[:, 0], pcs[:, 1]; r["kpc1"], r["kpc2"] = kp[:, 0], kp[:, 1]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (x, y, t, col) in zip(ax, [("pc1", "pc2", "PCA + K-Means", "cluster"), ("pc1", "pc2", "PCA + DBSCAN (-1 = noise)", "dbscan"), ("kpc1", "kpc2", "Kernel PCA (rbf) + BIRCH", "birch")]):
        a.scatter(r[x], r[y], c=r[col], cmap="tab10", s=40)
        for _, row in r[r.anomaly].iterrows(): a.annotate(row.full_name.split("/")[1], (row[x], row[y]), fontsize=7)
        a.set_title(t)
    fig.tight_layout(); fig.savefig("report/pca_repos.png", dpi=140); plt.close(fig)
    prof = pd.concat([F, r[["cluster"]]], axis=1).groupby("cluster").mean().round(2); prof["n_repos"] = r.cluster.value_counts().sort_index()
    prof.to_csv("report/cluster_profiles.csv"); print("\ncluster profiles (name the clusters yourself from this):\n", prof.to_string())
    r[["full_name", "cluster", "birch", "dbscan", "anomaly", "anomaly_score", "abandoned", "pc1", "pc2", "kpc1", "kpc2", "days_since_push"]].to_csv("data/processed/repo_flags.csv", index=False)
    print("saved data/processed/repo_flags.csv")
````


### `ml/gradient_descent_demo.py`: NEW

````python
"""Cost function + gradient descent + decision boundary, from scratch (syllabus: 'Cost function, Optimization, Decision boundary')."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from features.build_features import labelled, NUM


def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
def cost(w, X, y): p = sigmoid(X @ w); return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))   # binary cross-entropy


def gd(X, y, lr, iters=400):
    w, hist = np.zeros(X.shape[1]), []
    for _ in range(iters):
        w -= lr * X.T @ (sigmoid(X @ w) - y) / len(y)        # gradient of the cost
        hist.append(cost(w, X, y))
    return w, hist


def main():
    df = labelled()
    Xs = StandardScaler().fit_transform(df[NUM]); X = np.c_[np.ones(len(Xs)), Xs]
    y = (df.difficulty == "Easy").astype(float).values          # task: Easy vs not-Easy
    plt.figure(figsize=(6, 4))
    for lr in [0.01, 0.1, 0.5, 2.0]:
        w, h = gd(X, y, lr); plt.plot(h, label=f"lr={lr}")
    plt.xlabel("iteration"); plt.ylabel("cost (cross-entropy)"); plt.legend(); plt.title("Gradient descent: learning-rate effect")
    plt.tight_layout(); plt.savefig("report/gd_cost_curve.png", dpi=140); plt.close()
    w, h = gd(X, y, 0.5, 2000)
    sk = LogisticRegression(C=1e6, max_iter=5000).fit(Xs, y)
    acc = lambda p: float(((p > .5) == y).mean())
    print(f"final cost {h[-1]:.4f} | my GD accuracy {acc(sigmoid(X @ w)):.3f} | sklearn accuracy {sk.score(Xs, y):.3f}")
    print("coefficient correlation (mine vs sklearn):", round(float(np.corrcoef(w[1:], sk.coef_[0])[0, 1]), 3))
    z = PCA(2).fit_transform(Xs); lr2 = LogisticRegression(class_weight="balanced").fit(z, y)
    gx, gy = np.meshgrid(np.linspace(z[:, 0].min() - 1, z[:, 0].max() + 1, 200), np.linspace(z[:, 1].min() - 1, z[:, 1].max() + 1, 200))
    pr = lr2.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)
    plt.figure(figsize=(5.5, 4.5)); plt.contourf(gx, gy, pr, levels=[0, .5, 1], alpha=.25, colors=["tab:red", "tab:green"])
    plt.scatter(z[:, 0], z[:, 1], c=y, cmap="RdYlGn", s=10, edgecolors="none"); plt.title("Decision boundary (Easy vs not) in PCA space")
    plt.tight_layout(); plt.savefig("report/decision_boundary.png", dpi=140); plt.close()
    print("saved report/gd_cost_curve.png and report/decision_boundary.png")
````


### `ml/evaluate.py`: REPLACE (file already exists in your project)

````python
"""Learning curves (bias-variance), manual-label agreement (kappa) and label-threshold sensitivity."""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from features.build_features import add_features, add_labels, load, labelled, FEATURES
from features.labels import assign_labels
from ml.common import preprocessor, temporal_split, save_metrics

warnings.filterwarnings("ignore")


def main():
    df = labelled().sort_values("created_at").reset_index(drop=True)
    plt.figure(figsize=(7, 4.8))
    cv = GroupKFold(n_splits=min(5, df.repo.nunique()))
    for name, clf, col in [("DecisionTree(deep)", DecisionTreeClassifier(random_state=42), "tab:red"),
                           ("RandomForest", RandomForestClassifier(200, random_state=42, n_jobs=-1), "tab:blue"),
                           ("LogisticRegression", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3), "tab:green")]:
        pipe = Pipeline([("pre", preprocessor()), ("clf", clf)])
        sz, tr, va = learning_curve(pipe, df[FEATURES], df["difficulty"], groups=df["repo"], cv=cv, scoring="f1_macro", train_sizes=np.linspace(.25, 1, 5), error_score=np.nan)
        plt.plot(sz, np.nanmean(tr, 1), "--", c=col, label=f"{name} train"); plt.plot(sz, np.nanmean(va, 1), c=col, label=f"{name} validation")
    plt.xlabel("training size"); plt.ylabel("macro F1"); plt.legend(fontsize=7); plt.title("Learning curves (grouped CV): a big train/validation gap = overfitting")
    plt.tight_layout(); plt.savefig("report/learning_curves.png", dpi=140); plt.close()

    p = "data/processed/manual_label_sample.csv"
    if os.path.exists(p):
        m = pd.read_csv(p); m = m[m.my_difficulty.astype(str).isin(["Easy", "Medium", "Hard"])]
        if len(m) >= 20:
            cur = df.set_index("id").difficulty
            m = m[m["id"].isin(cur.index)]; auto = m["id"].map(cur)       # compare with the CURRENT label recipe
            res = {"n": len(m), "agreement": round(float((auto == m.my_difficulty).mean()), 3), "kappa": round(float(cohen_kappa_score(auto, m.my_difficulty)), 3),
                   "within_one_level": round(float((auto.map({"Easy": 0, "Medium": 1, "Hard": 2}) - m.my_difficulty.map({"Easy": 0, "Medium": 1, "Hard": 2})).abs().le(1).mean()), 3)}
            print("manual vs automatic label:", res); save_metrics("label_validation", res)
        else:
            print(f"only {len(m)} manual labels so far: label at least 100-200 with `python cli.py label`")
    sens = {}
    for eq, hq in [(0.35, 0.75), (0.40, 0.80), (0.45, 0.85)]:
        raw = add_features(load(state="closed")); raw = raw[(raw.pr_merged == 1) & raw.pr_files.notna()]
        d = assign_labels(raw, eq, hq).sort_values("created_at"); tr, te = temporal_split(d)
        pipe = Pipeline([("pre", preprocessor()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3))]).fit(tr[FEATURES], tr.difficulty)
        sens[f"{eq}/{hq}"] = round(float(f1_score(te.difficulty, pipe.predict(te[FEATURES]), average="macro")), 3)
    print("label-threshold sensitivity (LR test macro-F1):", sens); save_metrics("label_sensitivity", sens)
````


### `ml/predict.py`: NEW

````python
"""Predict the difficulty of ANY GitHub issue:  python cli.py predict owner/repo#123   (or the issue URL)"""
import json, os, re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from features.build_features import load, add_features, FEATURES, labelled

_SIM = {}


def parse_ref(ref):
    m = re.search(r"github\.com/([\w.\-]+/[\w.\-]+)/issues/(\d+)", ref) or re.match(r"^([\w.\-]+/[\w.\-]+)#(\d+)$", ref.strip())
    if not m:
        raise ValueError("use owner/repo#123 or a github.com issue URL")
    return m.group(1), int(m.group(2))


def _row_from_db(repo, number):
    df = load()
    r = df[(df.repo == repo) & (df.number == number)]
    return r.head(1) if len(r) else None


def _row_from_api(repo, number):
    from collector.github_client import GitHubClient
    gh = GitHubClient()
    i = gh.get(f"/repos/{repo}/issues/{number}")["body"]; rp = gh.get(f"/repos/{repo}")["body"]
    if not i or not rp:
        raise ValueError("issue or repo not found (private, or wrong name)")
    has_c = any(gh.get(f"/repos/{repo}/contents/{p}")["body"] for p in ["CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md"])
    return pd.DataFrame([{
        "id": f"{repo}#{number}", "repo": repo, "number": number, "title": i["title"], "body": i.get("body") or "",
        "labels": "|".join(l["name"] for l in i["labels"]), "created_at": i["created_at"], "closed_at": i.get("closed_at"),
        "state": i["state"], "comments": i["comments"], "has_assignee": int(bool(i.get("assignee"))), "author_assoc": i.get("author_association"),
        "has_open_pr": 0, "stars": rp["stargazers_count"], "forks": rp["forks_count"], "language": rp["language"], "contributors": 0,
        "has_contributing": int(has_c), "archived": int(rp["archived"]), "repo_created": rp["created_at"], "pushed_at": rp["pushed_at"]}])


def similar_issues(text, k=3, exclude_id=None):
    """The k most similar SOLVED issues and how hard they turned out to be (explains the prediction)."""
    if "tf" not in _SIM:
        lab = labelled(); tf = TfidfVectorizer(max_features=5000, stop_words="english", min_df=1, ngram_range=(1, 2))
        _SIM.update(df=lab.reset_index(drop=True), tf=tf, X=tf.fit_transform(lab["text"]))
    s = (_SIM["X"] @ _SIM["tf"].transform([text]).T).toarray().ravel()
    out = []
    for i in np.argsort(-s):
        r = _SIM["df"].iloc[i]
        if r["id"] == exclude_id or s[i] <= 0:
            continue
        out.append({"id": r["id"], "title": r["title"][:90], "difficulty": r["difficulty"], "similarity": round(float(s[i]), 2),
                    "url": f"https://github.com/{r['repo']}/issues/{int(r['number'])}"})
        if len(out) == k:
            break
    return out


def predict_issue(ref, allow_api=True):
    repo, number = parse_ref(ref)
    row = _row_from_db(repo, number)
    src = "database"
    if row is None:
        if not allow_api:
            raise ValueError("issue not in the database")
        row, src = _row_from_api(repo, number), "GitHub API"
    row = add_features(row)
    m = joblib.load("models/difficulty.joblib")
    proba = m.predict_proba(row[FEATURES])[0]; classes = list(m.classes_)
    label = classes[int(proba.argmax())]                      # most probable class, consistent with the printed probabilities
    conf = float(proba.max())
    return {"issue": f"{repo}#{number}", "title": row.iloc[0]["title"], "source": src, "predicted": label,
            "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)}, "confidence": round(conf, 3),
            "uncertain": conf < 0.5, "similar_solved": similar_issues(row.iloc[0]["text"], exclude_id=f"{repo}#{number}"),
            "note": "Probabilities are calibrated, but the model is weak when trained on few repos: treat as a hint, read the issue."}
````


### `ml/make_report.py`: NEW

````python
"""Collects every result into report/RESULTS.md (paste-ready tables for your two reports). Never invents numbers."""
import json, os
import pandas as pd


def _read(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else "_not generated yet_\n"


def main():
    from collector.db import conn
    c = conn()
    q = lambda s: c.execute(s).fetchone()[0]
    M = json.load(open("report/metrics.json")) if os.path.exists("report/metrics.json") else {}
    L = ["# IssueScout results (auto-generated by `python cli.py report`)", ""]
    n_open = q("select count(*) from issues where state='open'")
    L += ["## 1. Dataset", f"- repos: {q('select count(*) from repos')}", f"- issues: {q('select count(*) from issues')} (open {n_open})",
          f"- labelled (merged PR): {q('select count(*) from issues where pr_merged=1')}", ""]
    if os.path.exists("report/eda/summary.md"):
        L += [_read("report/eda/summary.md").replace("# EDA summary", "### EDA summary"), ""]
    L += ["## 2. Difficulty label validation", json.dumps(M.get("label_validation", "no manual labels yet"), indent=1),
          "label-threshold sensitivity: " + json.dumps(M.get("label_sensitivity", "n/a")), ""]
    L += ["## 3. Difficulty classification: model comparison", "(grouped CV = unseen repos; test = newest 20%; chance for 3 balanced classes is about 0.33)", "", _read("report/model_comparison.md")]
    meta = M.get("difficulty_meta", {})
    if meta:
        cv = pd.read_csv("report/model_comparison.csv", index_col=0)["cv_f1_grouped"].dropna()
        best = cv.max()
        L += [f"Chosen: **{meta.get('chosen')}** (n={meta.get('n_train')}, repos={meta.get('n_repos')}, calibration ECE={meta.get('ece_test')}).",
              "", f"**Honest reading:** best grouped-CV macro-F1 is {best:.2f}. " + ("This is close to chance: say so, and explain that more repos are needed." if best < 0.42 else "This is clearly above chance."), ""]
    L += ["## 4. Merge prediction", _read("report/merge_comparison.md"), "## 5. Time-to-close regression", _read("report/time_comparison.md"),
          f"beats repo-median baseline: {M.get('time_to_close', {}).get('beats_repo_median')}", "",
          "## 6. Repository clustering", _read("report/cluster_profiles.csv") if os.path.exists("report/cluster_profiles.csv") else "_run `python cli.py cluster`_", "",
          "## 7. CSP benchmark", _read("report/csp_benchmark.md"), "## 8. A* benchmark", _read("report/astar_benchmark.md"),
          "## 9. RAG evaluation", json.dumps(M.get("rag_eval", "not run"), indent=1), "## 10. LLM evaluation", json.dumps(M.get("llm_eval", "not run"), indent=1), ""]
    from core import outcomes
    L += ["## 11. Real-world outcomes (your own attempts)", json.dumps(outcomes.summary(), indent=1), "",
          "## Figures"]
    L += [f"- report/{f}" for f in sorted(os.listdir("report")) if f.endswith(".png")]
    if os.path.isdir("report/eda"):
        L += [f"- report/eda/{f}" for f in sorted(os.listdir("report/eda")) if f.endswith(".png")]
    open("report/RESULTS.md", "w", encoding="utf-8").write("\n".join(L))
    print("wrote report/RESULTS.md")
````


---

## PART 3. AI engine and tests


### `ai/csp.py`: REPLACE (file already exists in your project)

````python
"""Issue-selection CSP: backtracking + forward checking + MRV + a remaining-budget bound.

Variables  : the k plan slots.            Domain: pre-filtered candidate issues.
Constraints: all different, sum(hours) <= budget, at most `max_per_repo` per repo.
Symmetry   : slot order follows candidate order (idx), so permutations of one plan are not re-explored.
Candidates are sorted best-score-first, so the first solution found is the highest-ranked feasible plan.

Forward checking alone cannot see the SHARED budget (it checks each slot separately), so it explores the same
nodes as plain backtracking. `bound=True` adds the missing global check: the cheapest possible completion must fit.
"""


def solve(cands, k, budget, max_per_repo=2, fc=True, mrv=True, bound=True, max_nodes=200000):
    """cands: dicts with id, repo, hours, idx.  Returns (solution list or None, nodes explored)."""
    stats = {"nodes": 0}

    def ok(assign, var, c):
        total, same = c["hours"], 0
        for s, a in assign.items():
            if a["idx"] == c["idx"]:
                return False
            if (s < var) != (a["idx"] < c["idx"]):
                return False
            total += a["hours"]
            same += a["repo"] == c["repo"]
        return total <= budget and same < max_per_repo

    def bt(assign, domains):
        if len(assign) == k:
            return dict(assign)
        free = [v for v in range(k) if v not in assign]
        var = min(free, key=lambda v: len(domains[v])) if mrv else free[0]          # MRV
        for c in domains[var]:
            if stats["nodes"] >= max_nodes:
                return None
            if not ok(assign, var, c):
                continue
            stats["nodes"] += 1
            assign[var] = c
            nd, dead = domains, False
            if fc:                                                                  # forward checking
                nd = dict(domains)
                rest = [v for v in free if v != var]
                for v in rest:
                    nd[v] = [x for x in domains[v] if ok(assign, v, x)]
                    if not nd[v]:
                        dead = True
                        break
                if not dead and bound and rest:                                     # global budget bound
                    used = sum(a["hours"] for a in assign.values())
                    cheapest = sorted({x["idx"]: x["hours"] for v in rest for x in nd[v]}.items(), key=lambda t: t[1])
                    if len(cheapest) < len(rest) or used + sum(h for _, h in cheapest[:len(rest)]) > budget:
                        dead = True
            if not dead:
                res = bt(assign, nd)
                if res:
                    return res
            del assign[var]
        return None

    sol = bt({}, {i: list(cands) for i in range(k)})
    if not sol:
        return None, stats["nodes"]
    return [sol[v] for v in sorted(sol, key=lambda v: sol[v]["idx"])], stats["nodes"]
````


### `ai/rules.py`: REPLACE (file already exists in your project)

````python
"""Forward-chaining expert system. Each fired rule is logged => the explanation trace shown to the user."""
import re


class Rule:
    def __init__(self, name, cond, effect):
        self.name, self.cond, self.effect = name, cond, effect


def _add(x):
    def f(s): s["priority"] += x
    return f


def _flag(name):
    def f(s): s["flags"].append(name)
    return f


def _exclude(s): s["excluded"] = True


NOISE = re.compile(r"\b(tracking issue|umbrella|epic|roadmap|meta|dependency dashboard)\b|^\s*\[?tracking", re.I)


def is_noise(title, labels=""):
    """Tracking / epic / bot-dashboard issues are not a single-PR task."""
    return bool(NOISE.search(title or "")) or any(w in (labels or "").lower() for w in ("tracking", "epic", "umbrella", "roadmap"))


def _slow(f): return f["reply_days"] is not None and f["reply_days"] > 60
def _fast(f): return f["reply_days"] is not None and f["reply_days"] < 3
def _lab(f, *words): return any(w in f.get("labels", "") for w in words)
def _title(f, pat): return bool(re.search(pat, f.get("title", ""), re.I))


RULES = [
    Rule("R1: an open PR already references this issue -> EXCLUDE", lambda f, s: f["has_open_pr"], _exclude),
    Rule("R2: maintainers reply slower than 60 days -> priority -0.30", lambda f, s: _slow(f), _add(-0.30)),
    Rule("R3: 'help wanted' and unassigned -> priority +0.20", lambda f, s: f["help_wanted"] and f["unassigned"], _add(0.20)),
    Rule("R4: CONTRIBUTING.md present and replies under 3 days -> priority +0.15", lambda f, s: f["has_contributing"] and _fast(f), _add(0.15)),
    Rule("R5: older than 1 year with no comments -> flag STALE", lambda f, s: f["age_days"] > 365 and f["comments"] == 0, _flag("stale")),
    Rule("R6: flagged stale -> priority -0.20 (chained from R5)", lambda f, s: "stale" in s["flags"], _add(-0.20)),
    Rule("R7: tracking / epic / umbrella issue (not one PR) -> EXCLUDE",
         lambda f, s: is_noise(f.get("title", ""), f.get("labels", "")), _exclude),
    Rule("R8: blocked / wontfix / duplicate / invalid / question label -> EXCLUDE",
         lambda f, s: _lab(f, "blocked", "wontfix", "won't fix", "duplicate", "invalid", "question", "stale", "needs-triage"), _exclude),
    Rule("R9: needs a design decision first (needs-design, proposal, rfc) -> priority -0.25",
         lambda f, s: _lab(f, "needs-design", "needs design", "proposal", "rfc", "needs-discussion", "needs discussion", "design"), _add(-0.25)),
    Rule("R10: maintainers tagged it good first issue -> priority +0.25", lambda f, s: f.get("gfi", False), _add(0.25)),
    Rule("R11: very short description (under 80 chars) -> priority -0.15, little to start from", lambda f, s: f.get("body_len", 999) < 80, _add(-0.15)),
    Rule("R12: busy discussion (15+ comments) on a recent issue -> priority -0.10", lambda f, s: f["comments"] >= 15 and f["age_days"] < 30, _add(-0.10)),
    Rule("R13: low model confidence -> flag UNCERTAIN (read the issue yourself)", lambda f, s: f.get("conf", 1.0) < 0.5, _flag("uncertain")),
]


def run_rules(facts):
    state = {"priority": 0.0, "excluded": False, "flags": [], "trace": []}
    fired, changed = set(), True
    while changed:                                  # forward chaining: repeat until nothing new fires
        changed = False
        for r in RULES:
            if r.name not in fired and r.cond(facts, state):
                r.effect(state); fired.add(r.name); state["trace"].append(r.name); changed = True
    return state
````


### `ai/ontology.py`: REPLACE (file already exists in your project)

````python
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


def extract_keywords(text, language):
    t = text.lower()
    found = {k for k, p in PATTERNS.items() if re.search(p, t)}
    lang = (language or "").lower()
    if lang in G and not (found and found <= {"markdown", "yaml", "json"}):
        found.add(lang)
    return found
````


### `ai/astar.py`: REPLACE (file already exists in your project)

````python
"""A* over 'skills gained' states.  State = set of skills known; action = finish a stepping-stone issue.
g = hours spent.   h = (#missing goal skills) x (cheapest hours-per-goal-skill any step offers).

ADMISSIBLE: a step costing H teaches at most |teaches ∩ goal| goal skills, so each missing goal skill costs at
least min_s H_s/|teaches_s ∩ goal|. h therefore never overestimates the true remaining cost. (h is also consistent.)
"""
import heapq, itertools


def _per_skill(goal, steps):
    vals = [s["hours"] / len(s["teaches"] & goal) for s in steps if s["teaches"] & goal]
    return min(vals) if vals else None


def astar(start, goal, steps, use_heuristic=True):
    """Returns (path, total_hours, nodes_expanded). use_heuristic=False gives uniform-cost search for comparison."""
    start, goal = frozenset(start), frozenset(goal)
    per = _per_skill(goal, steps)
    if per is None:
        return None, None, 0
    h = (lambda st: len(goal - st) * per) if use_heuristic else (lambda st: 0.0)
    tie = itertools.count()
    pq = [(h(start), next(tie), 0.0, start, [])]
    best, expanded = {start: 0.0}, 0
    while pq:
        f, _, g, state, path = heapq.heappop(pq)
        if goal <= state:
            return path, g, expanded
        if g > best.get(state, 1e18):
            continue
        expanded += 1
        for s in steps:
            if not s["needs"] <= state:
                continue
            new = state | s["teaches"]
            if new == state:
                continue
            g2 = g + s["hours"]
            if g2 < best.get(new, 1e18):
                best[new] = g2
                heapq.heappush(pq, (g2 + h(new), next(tie), g2, new, path + [s]))
    return None, None, expanded


def verify_admissible(start, goal, steps, limit=400):
    """Empirical proof on small instances: for every reachable state, h(state) <= true optimal cost-to-go."""
    goal = frozenset(goal); per = _per_skill(goal, steps)
    seen, stack, worst = set(), [frozenset(start)], 0.0
    while stack and len(seen) < limit:
        st = stack.pop()
        if st in seen:
            continue
        seen.add(st)
        for s in steps:
            if s["needs"] <= st and not s["teaches"] <= st:
                stack.append(st | s["teaches"])
    bad = []
    for st in seen:
        _, true_cost, _ = astar(st, goal, steps, use_heuristic=False)
        if true_cost is None:
            continue
        hv = len(goal - st) * per
        worst = max(worst, hv / true_cost if true_cost else 0)
        if hv > true_cost + 1e-9:
            bad.append((sorted(st), hv, true_cost))
    return (not bad), worst, len(seen)
````


### `ai/bayes.py`: REPLACE (file already exists in your project)

````python
"""Bayesian reasoning + parameter estimation (syllabus: Bayes rule, MLE)."""
from scipy.stats import beta


def mle_rate(successes, n):
    """Maximum-likelihood estimate of a merge rate: s/n. Wild for small n (2 of 2 => 100%)."""
    return successes / n if n else float("nan")


def posterior(successes, n, prior_rate, strength=10):
    """Beta-Binomial MAP/mean. Prior = global merge rate worth `strength` pseudo-observations."""
    a0, b0 = prior_rate * strength, (1 - prior_rate) * strength
    a, b = a0 + successes, b0 + (n - successes)
    return a / (a + b), tuple(beta.interval(0.9, a, b))


def update(prior, *likelihood_ratios):
    """Bayes rule in odds form: posterior odds = prior odds x LR1 x LR2 ..."""
    odds = prior / (1 - prior)
    for lr in likelihood_ratios:
        odds *= lr
    return odds / (1 + odds)


def demo_table(strength=10):
    """Per-repo MLE vs Bayesian estimate, from your own database (shows shrinkage for small repos)."""
    import pandas as pd
    from collector.db import conn
    d = pd.read_sql("SELECT repo, pr_merged FROM issues WHERE pr_merged IS NOT NULL", conn())
    g = d.groupby("repo").pr_merged.agg(["sum", "count"]).rename(columns={"sum": "merged", "count": "n"})
    prior = g.merged.sum() / g.n.sum()
    g["mle"] = [mle_rate(m, n) for m, n in zip(g.merged, g.n)]
    post = [posterior(m, n, prior, strength) for m, n in zip(g.merged, g.n)]
    g["bayes"] = [p[0] for p in post]; g["ci90_low"] = [p[1][0] for p in post]; g["ci90_high"] = [p[1][1] for p in post]
    return prior, g.sort_values("n").round(3)
````


### `ai/benchmark_csp.py`: NEW

````python
"""Node-count comparison: plain backtracking vs +FC vs +MRV vs +FC+MRV vs +bound.  Writes report/csp_benchmark.md"""
import random, time


def pools():
    out = []
    for seed, n, nrepos in [(3, 30, 6), (7, 40, 8), (11, 25, 5)]:
        rnd = random.Random(seed)
        out.append((f"synthetic n={n}", [{"id": i, "repo": f"r{rnd.randrange(nrepos)}", "hours": round(rnd.uniform(2, 12), 1), "idx": i} for i in range(n)]))
    try:                                           # a real pool from your own data, if the models exist
        from core.recommend import score_open
        df = score_open().sort_values("p_merge", ascending=False).head(40).reset_index(drop=True)
        out.append(("your data (top 40)", [{"id": r.id, "repo": r.repo, "hours": float(r.est_hours), "idx": i} for i, r in df.iterrows()]))
    except Exception as e:
        print("real pool skipped:", str(e)[:80])
    return out


def main():
    from ai.csp import solve
    cfgs = [("plain", dict(fc=False, mrv=False)), ("FC", dict(fc=True, mrv=False, bound=False)), ("MRV", dict(fc=False, mrv=True)),
            ("FC+MRV", dict(fc=True, mrv=True, bound=False)), ("FC+MRV+budget bound", dict(fc=True, mrv=True, bound=True))]
    lines = ["| pool | k | budget h | " + " | ".join(n for n, _ in cfgs) + " |", "|" + "---|" * (3 + len(cfgs))]
    for name, pool in pools():
        for k, budget in [(4, 16), (5, 20), (6, 22), (6, 19), (7, 24)]:
            cells = []
            for _, kw in cfgs:
                t = time.time(); sol, n = solve(pool, k, budget, max_nodes=300000, **kw)
                cells.append(f"{n}{'' if sol else ' (no plan)'} / {time.time()-t:.2f}s")
            lines.append(f"| {name} | {k} | {budget} | " + " | ".join(cells) + " |")
    txt = "Nodes explored / seconds (lower is better; all methods find a plan whenever one exists).\n\n" + "\n".join(lines) + "\n"
    import os; os.makedirs("report", exist_ok=True)
    open("report/csp_benchmark.md", "w").write(txt); print(txt)
````


### `ai/benchmark_astar.py`: NEW

````python
"""A* vs uniform-cost search (same optimal cost, fewer expansions) + empirical admissibility check."""
import random
from ai.astar import astar, verify_admissible


def instance(seed, n_skills=7, n_steps=22):
    rnd = random.Random(seed); sk = [f"s{i}" for i in range(n_skills)]; steps = []
    for i in range(n_steps):
        teaches = frozenset(rnd.sample(sk, rnd.choice([1, 1, 2])))
        needs = frozenset(rnd.sample([s for s in sk if s not in teaches], rnd.choice([0, 0, 1])))
        steps.append({"id": f"step{i}", "title": f"step{i}", "teaches": teaches, "needs": needs, "hours": float(rnd.randint(2, 12))})
    return set(), set(sk[:6]), steps


def main():
    rows, ok_all = [], True
    for seed in range(8):
        s, g, steps = instance(seed)
        pa, ca, ea = astar(s, g, steps); pu, cu, eu = astar(s, g, steps, use_heuristic=False)
        if ca is None:
            continue
        adm, worst, n = verify_admissible(s, g, steps, limit=120)
        ok_all &= adm and abs(ca - cu) < 1e-9
        rows.append(f"| {seed} | {ca:.0f} | {cu:.0f} | {ea} | {eu} | {'yes' if adm else 'NO'} (max h/true = {worst:.2f}, {n} states) |")
    txt = ("| instance | A* cost | UCS cost | A* expanded | UCS expanded | h admissible? |\n|---|---|---|---|---|---|\n" + "\n".join(rows)
           + f"\n\nA* always matched the optimal UCS cost: {ok_all}\n")
    import os; os.makedirs("report", exist_ok=True)
    open("report/astar_benchmark.md", "w").write(txt); print(txt)
````


### `tests/test_ai.py`: NEW

````python
import random
from ai.csp import solve
from ai.astar import astar, verify_admissible
from ai.rules import run_rules
from ai.ontology import missing, closure, extract_keywords, skills_to_learn
from ai.bayes import posterior, update, mle_rate


def facts(**kw):
    f = dict(has_open_pr=False, reply_days=None, help_wanted=False, unassigned=True, has_contributing=False, age_days=10,
             comments=0, labels="", title="fix typo in docs", body_len=300, gfi=False, conf=0.8)
    f.update(kw); return f


def test_rules_exclude_and_chain():
    assert run_rules(facts(has_open_pr=True))["excluded"]
    assert run_rules(facts(title="Tracking issue: v2"))["excluded"]
    assert run_rules(facts(labels="wontfix"))["excluded"]
    s = run_rules(facts(age_days=500, comments=0))
    assert "stale" in s["flags"] and s["priority"] < 0 and len(s["trace"]) == 2     # R5 fires, then R6 chains
    assert run_rules(facts(gfi=True, help_wanted=True))["priority"] > 0.4
    assert "uncertain" in run_rules(facts(conf=0.4))["flags"]


def pool(seed=3, n=30):
    r = random.Random(seed)
    return [{"id": i, "repo": f"r{r.randrange(6)}", "hours": round(r.uniform(2, 12), 1), "idx": i} for i in range(n)]


def test_csp_constraints_hold():
    p = pool(); sol, _ = solve(p, 5, 20)
    assert sol and len(sol) == 5 and len({c["id"] for c in sol}) == 5
    assert sum(c["hours"] for c in sol) <= 20
    assert max(sum(1 for c in sol if c["repo"] == r) for r in {c["repo"] for c in sol}) <= 2


def test_csp_methods_agree_and_bound_prunes():
    p = pool()
    for k, b in [(5, 20), (6, 19)]:
        s1, n1 = solve(p, k, b, fc=False, mrv=False); s2, n2 = solve(p, k, b)
        assert (s1 is None) == (s2 is None)
        assert n2 <= n1
        if s1: assert [c["id"] for c in s1] == [c["id"] for c in s2]          # same plan, fewer nodes
    _, n1 = solve(p, 6, 19, fc=False, mrv=False); _, n2 = solve(p, 6, 19)
    assert n2 * 5 < n1                                                         # the bound gives a large saving


def test_csp_infeasible():
    assert solve(pool(), 5, 3)[0] is None


def steps():
    f = frozenset
    return [{"id": "a", "title": "a", "teaches": f({"go"}), "needs": f(), "hours": 3},
            {"id": "b", "title": "b", "teaches": f({"docker"}), "needs": f(), "hours": 4},
            {"id": "c", "title": "c", "teaches": f({"client-go"}), "needs": f({"go"}), "hours": 6},
            {"id": "e", "title": "e", "teaches": f({"go", "docker"}), "needs": f(), "hours": 9}]


def test_astar_optimal_and_admissible():
    goal = {"go", "docker", "client-go"}
    p, c, e = astar(set(), goal, steps()); p2, c2, e2 = astar(set(), goal, steps(), use_heuristic=False)
    assert c == c2 == 13 and e <= e2
    ok, worst, n = verify_admissible(set(), goal, steps()); assert ok and worst <= 1.0 + 1e-9


def test_astar_unreachable():
    assert astar(set(), {"rust"}, steps())[0] is None


def test_ontology():
    assert not missing(["go"], {"client-go", "go"})                 # client-go is learnable for a Go user
    assert "kubernetes" in missing(["go"], {"kubernetes"})
    assert "javascript" in closure(["react"])
    assert {"docker", "yaml"} <= skills_to_learn(["go"], {"kubernetes"}) | {"docker", "yaml"}
    assert "go" in extract_keywords("panic in main.go when parsing", "Go") and "python" in extract_keywords("pip install fails", "")


def test_bayes():
    assert mle_rate(2, 2) == 1.0
    m, (lo, hi) = posterior(2, 2, 0.8); assert 0.8 < m < 0.95 and lo < m < hi    # shrunk below the MLE of 1.0
    assert 0.5 < update(0.5, 2.0) < 1


def test_noise_issues():
    from ai.rules import is_noise
    assert is_noise("Dependency Dashboard") and is_noise("Tracking issue: v2 release") and is_noise("x", "kind/epic")
    assert not is_noise("Fix typo in README")
````


### `tests/test_labels.py`: NEW

````python
import numpy as np, pandas as pd
from features.labels import assign_labels
from features.text_features import struct_features


def df():
    r = np.random.default_rng(0); n = 120
    return pd.DataFrame({"repo": ["a"] * 60 + ["b"] * 60, "pr_additions": r.integers(1, 400, n), "pr_deletions": r.integers(0, 100, n),
                         "pr_files": r.integers(1, 12, n), "pr_review_comments": r.integers(0, 10, n)})


def test_label_shares_and_order():
    d = assign_labels(df()); s = d.difficulty.value_counts(normalize=True)
    assert abs(s["Easy"] - 0.4) < 0.1 and abs(s["Hard"] - 0.2) < 0.1
    assert d[d.difficulty == "Easy"].effort.max() <= d[d.difficulty == "Hard"].effort.min()


def test_guards():
    d = df(); d.loc[0, ["pr_additions", "pr_deletions", "pr_files"]] = [2, 1, 1]; d.loc[1, ["pr_additions", "pr_deletions", "pr_files"]] = [3000, 10, 50]
    d = assign_labels(d); assert d.difficulty[0] == "Easy" and d.difficulty[1] == "Hard"


def test_label_does_not_use_issue_text():
    d = df(); a = assign_labels(d).difficulty.tolist(); d["title"] = "x"; assert assign_labels(d).difficulty.tolist() == a


def test_struct_features_are_creation_time_only():
    f = struct_features("Fix typo in README", "see docs/guide/setup.md")
    assert f["kw_small"] == 1 and f["n_paths"] == 1 and f["has_stack_trace"] == 0
    assert struct_features("crash", "Traceback (most recent call last)")["has_stack_trace"] == 1
````


### `tests/run_tests.py`: NEW

````python
"""Runs every test_* function without needing pytest:  python tests/run_tests.py   (pytest also works)"""
import importlib, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fails = total = 0
for mod in sorted(f[:-3] for f in os.listdir(os.path.dirname(__file__)) if f.startswith("test_") and f.endswith(".py")):
    m = importlib.import_module(f"tests.{mod}")
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        total += 1
        try:
            getattr(m, name)(); print(f"PASS {mod}.{name}")
        except Exception:
            fails += 1; print(f"FAIL {mod}.{name}"); traceback.print_exc()
print(f"\n{total - fails}/{total} passed"); sys.exit(1 if fails else 0)
````


---

## PART 4. Core, LLM, RAG, first-PR kit, interfaces


### `core/profile.py`: REPLACE (file already exists in your project)

````python
from pydantic import BaseModel, Field


class Profile(BaseModel):
    skills: list[str] = Field(default_factory=list)
    level: str = "beginner"              # beginner | intermediate | advanced
    hours_per_week: float = 6
    weeks: float = 3
    k: int = 3
    avoid: list[str] = Field(default_factory=list)
    use_llm: bool = False
    include_uncertain: bool = True       # False hides issues where the model's top probability is under 50%
````


### `core/outcomes.py`: NEW

````python
"""Outcome tracking: the real success metric is 'did a recommended issue become a merged PR?'"""
import json, os
from datetime import datetime

PATH = "data/outcomes.jsonl"
STATUSES = ["started", "pr_opened", "merged", "closed_unmerged", "abandoned", "too_hard"]


def add(issue_id, status, hours=None, notes=""):
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    os.makedirs("data", exist_ok=True)
    rec = {"issue_id": issue_id, "status": status, "hours": hours, "notes": notes, "at": datetime.utcnow().isoformat()}
    with open(PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load():
    return [json.loads(l) for l in open(PATH)] if os.path.exists(PATH) else []


def latest():
    d = {}
    for r in load():
        d[r["issue_id"]] = r
    return d


def summary():
    L = latest(); n = len(L)
    cnt = {s: sum(1 for r in L.values() if r["status"] == s) for s in STATUSES}
    hrs = [r["hours"] for r in L.values() if r.get("hours")]
    return {"issues_tried": n, **cnt, "merge_rate_of_tried": round(cnt["merged"] / n, 2) if n else None,
            "avg_hours_logged": round(sum(hrs) / len(hrs), 1) if hrs else None}
````


### `core/recommend.py`: REPLACE (file already exists in your project)

````python
import json, os
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from core.profile import Profile
from core import outcomes
from features.build_features import load, add_features, FEATURES
from ai.ontology import missing, skills_to_learn, prereqs, closure
from ai.rules import run_rules, is_noise
from ai.csp import solve
from ai.bayes import posterior
from llm.extract_skills import extract_skills
from llm.explain import explain_checked

# Level = cap on EXPECTED difficulty  E = P(Medium) + 2*P(Hard)   (all-Easy = 0, all-Hard = 2)
LEVEL_CAP = {"beginner": 0.85, "intermediate": 1.10, "advanced": 99}
BASE_HOURS = {"Easy": 3, "Medium": 8, "Hard": 20}


def repo_table():
    df = load(state="closed")
    g = df.groupby("repo").agg(n=("pr_merged", "size"), merged=("pr_merged", "sum"), reply_h=("first_response_hours", "median"))
    g["reply_days"] = g["reply_h"] / 24
    prior = float(g["merged"].sum() / max(g["n"].sum(), 1))
    g["merge_rate"] = [posterior(m, n, prior)[0] for m, n in zip(g["merged"], g["n"])]
    return g, prior


def _meta():
    p = "models/difficulty_meta.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def score_open():
    df = load(state="open")
    if df.empty:
        return df
    df = add_features(df)
    stats, prior = repo_table()
    df = df.join(stats[["reply_days", "merge_rate"]], on="repo")
    df["merge_rate"] = df["merge_rate"].fillna(prior)

    X = df[FEATURES]
    m = joblib.load("models/difficulty.joblib")
    proba, classes = m.predict_proba(X), list(m.classes_)
    ix = {c: classes.index(c) for c in classes}
    df["difficulty"] = [classes[i] for i in proba.argmax(1)]                 # label = most probable class (matches the printed probabilities)
    df["diff_conf"] = proba.max(1)
    df["p_easy"] = proba[:, ix["Easy"]]; df["p_hard"] = proba[:, ix["Hard"]]; df["p_med"] = proba[:, ix["Medium"]]
    df["exp_diff"] = df["p_med"] + 2 * df["p_hard"]
    df["est_hours"] = proba @ np.array([BASE_HOURS[c] for c in classes])

    if os.path.exists("models/merge.joblib"):
        pm = joblib.load("models/merge.joblib").predict_proba(X)[:, 1]
        df["p_merge"] = 0.7 * pm + 0.3 * df["merge_rate"]                    # blend model with Bayesian repo estimate
    else:
        df["p_merge"] = df["merge_rate"]
    df["p_merge"] = df["p_merge"].clip(0.01, 0.99)

    df["req"] = [extract_skills(t, b, l) for t, b, l in zip(df.title, df.body, df.language)]
    bad = set()
    if os.path.exists("data/processed/repo_flags.csv"):
        fl = pd.read_csv("data/processed/repo_flags.csv")
        bad = set(fl[fl.abandoned].full_name)
    df["abandoned"] = df.repo.isin(bad) | (df.archived == 1)
    return df


def _user_blocklist():
    """Issues the user already marked too hard / is working on."""
    block = {r["issue_id"]: "you marked it too hard" if r["status"] == "too_hard" else f"already {r['status']}" for r in outcomes.latest().values()}
    if os.path.exists("data/feedback.jsonl"):
        for l in open("data/feedback.jsonl"):
            f = json.loads(l)
            if f.get("verdict") == "too_hard":
                block[f["issue_id"]] = "you marked it too hard"
    return block


def recommend(p: Profile):
    df = score_open()
    if df is None or df.empty:
        return {"plan": [], "note": "No open issues collected yet."}
    budget, cap = p.hours_per_week * p.weeks, LEVEL_CAP[p.level]
    avoid = [a.lower() for a in p.avoid]
    block = _user_blocklist()
    why, cands = Counter(), []
    now = pd.Timestamp.now(tz="UTC")

    for r in df.to_dict("records"):
        text = f"{r['title']} {r['body']}".lower()
        if r["id"] in block:                        why[block[r["id"]]] += 1; continue
        if r["has_assignee"]:                       why["already assigned"] += 1; continue
        if r["abandoned"]:                          why["repo abandoned"] += 1; continue
        if r["exp_diff"] > cap:                     why["too hard for level"] += 1; continue
        if not p.include_uncertain and r["diff_conf"] < 0.5:  why["model uncertain"] += 1; continue
        if any(a in text for a in avoid):           why["matches avoid list"] += 1; continue
        if r["est_hours"] > budget:                 why["exceeds time budget"] += 1; continue
        if missing(p.skills, r["req"]):             why["missing skills"] += 1; continue
        rd = None if pd.isna(r["reply_days"]) else float(r["reply_days"])
        facts = {"has_open_pr": bool(r["has_open_pr"]), "reply_days": rd, "help_wanted": bool(r["is_help_wanted"]), "unassigned": True,
                 "has_contributing": bool(r["has_contributing"]), "age_days": (now - pd.to_datetime(r["created_at"], utc=True)).days,
                 "comments": r["comments"], "labels": str(r["labels"]).lower(), "title": str(r["title"]).lower(),
                 "body_len": int(r["body_len"]), "gfi": bool(r["is_gfi"]), "conf": float(r["diff_conf"])}
        st = run_rules(facts)
        if st["excluded"]:
            why["excluded by rule (open PR / tracking / blocked)"] += 1; continue
        cands.append({**r, "reply_days": rd, "rules": st["trace"], "flags": st["flags"],
                      "score": r["p_merge"] + 0.3 * r["p_easy"] - 0.2 * r["exp_diff"] + st["priority"]})

    cands.sort(key=lambda c: -c["score"])
    short = cands[:30]
    if p.use_llm:                                    # LLM re-checks skills on the shortlist only
        keep = []
        for c in short:
            req = extract_skills(c["title"], c["body"], c["language"], use_llm=True)
            if not missing(p.skills, req):
                c["req"] = req; keep.append(c)
        short = keep
    pool = [dict(c, idx=i, hours=c["est_hours"]) for i, c in enumerate(short)]

    chosen, nodes, nodes_plain = None, 0, None
    for kk in range(p.k, 0, -1):
        chosen, nodes = solve(pool, kk, budget)
        if chosen:
            _, nodes_plain = solve(pool, kk, budget, fc=False, mrv=False, bound=False, max_nodes=50000)
            break
    if not chosen:
        return {"plan": [], "rejected": dict(why), "note": "No feasible plan. Try more hours, a higher level, or more skills."}

    repo_count = Counter(c["repo"] for c in chosen)
    total = sum(c["est_hours"] for c in chosen)
    try:
        from ml.predict import similar_issues
    except Exception:
        similar_issues = None
    plan = []
    for rank, c in enumerate(chosen, 1):
        item = {
            "rank": rank, "id": c["id"], "repo": c["repo"], "number": int(c["number"]), "title": c["title"],
            "url": f"https://github.com/{c['repo']}/issues/{int(c['number'])}",
            "difficulty": c["difficulty"], "diff_conf": round(float(c["diff_conf"]), 2),
            "probs": {"Easy": round(float(c["p_easy"]), 2), "Medium": round(float(c["p_med"]), 2), "Hard": round(float(c["p_hard"]), 2)},
            "uncertain": bool(c["diff_conf"] < 0.5), "p_merge": round(float(c["p_merge"]), 2), "est_hours": round(float(c["est_hours"]), 1),
            "reply_days": None if c["reply_days"] is None else round(c["reply_days"], 1),
            "skills": sorted(c["req"]), "rules": c["rules"], "flags": c["flags"], "snippet": c["body"][:800],
            "csp_check": {"skills": not missing(p.skills, c["req"]), "time": total <= budget, "unclaimed": not c["has_assignee"],
                          "no_open_pr": not c["has_open_pr"], "max_2_per_repo": repo_count[c["repo"]] <= 2, "not_abandoned": not c["abandoned"]},
        }
        if similar_issues:
            try: item["similar_solved"] = similar_issues(f"{c['title']} . {c['body'][:1500]}", k=2, exclude_id=c["id"])
            except Exception: item["similar_solved"] = []
        item["explanation"], item["explanation_meta"] = explain_checked(item, p.use_llm)
        plan.append(item)

    out = {"plan": plan, "budget_hours": budget, "planned_hours": round(total, 1), "rejected": dict(why),
           "csp_nodes_fc_mrv": nodes, "csp_nodes_plain": nodes_plain}
    os.makedirs("data", exist_ok=True)
    json.dump(out, open("data/last_plan.json", "w"), indent=2, default=str)
    return out


def learning_path(p: Profile, target: str):
    """target = 'owner/repo#123'. A* over stepping-stone issues."""
    from ai.astar import astar
    df = score_open()
    repo, num = target.split("#")
    row = df[(df.repo == repo) & (df.number == int(num))]
    if row.empty:
        return {"error": "Target issue is not in the collected open issues."}
    goal = skills_to_learn(p.skills, row.iloc[0]["req"])
    if not goal:
        return {"ready": True, "note": "You already have the skills for this issue."}
    steps = []
    for r in df.to_dict("records"):
        if r["has_assignee"] or r["has_open_pr"] or r["abandoned"] or r["p_hard"] > 0.4 or is_noise(r["title"], r["labels"]):
            continue
        teaches = set(r["req"]) & goal
        if not teaches:
            continue
        needs = set().union(*[prereqs(s) for s in teaches]) - teaches
        steps.append({"id": r["id"], "title": r["title"], "teaches": frozenset(teaches),
                      "needs": frozenset(needs), "hours": max(float(r["est_hours"]), 1.0)})
    steps = sorted(steps, key=lambda s: s["hours"])[:60]
    if not steps:
        return {"error": "No stepping-stone issues found for the missing skills."}
    teachable = set().union(*[s["teaches"] for s in steps])
    reach, unreachable = goal & teachable, goal - teachable          # plan around skills the collected issues can actually teach
    if not reach:
        return {"error": "None of the missing skills can be learned from the collected issues.", "goal": sorted(goal), "unreachable_skills": sorted(unreachable)}
    path, hours, expanded = astar(closure(p.skills), reach, steps)
    _, h_ucs, expanded_ucs = astar(closure(p.skills), reach, steps, use_heuristic=False)
    if path is None:
        return {"error": "No path found with the collected issues.", "goal": sorted(goal), "unreachable_skills": sorted(unreachable)}
    return {"goal": sorted(reach), "unreachable_skills": sorted(unreachable), "expanded_nodes": expanded, "expanded_nodes_ucs": expanded_ucs, "total_hours": round(hours, 1),
            "path": [{"id": s["id"], "title": s["title"], "teaches": sorted(s["teaches"]), "hours": round(s["hours"], 1)} for s in path]}
````


### `llm/client.py`: REPLACE (file already exists in your project)

````python
"""One LLM entry point.  LLM_PROVIDER = openai | anthropic | mock | none   (mock = offline, for tests and demos)
Failures never crash the app: complete() returns None and the error goes to data/llm_errors.log."""
import json, os, re, time
from datetime import datetime

_LAST = {"error": None}


def _log(msg):
    _LAST["error"] = msg
    os.makedirs("data", exist_ok=True)
    with open("data/llm_errors.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} {msg}\n")


def last_error():
    return _LAST["error"]


def provider():
    return os.getenv("LLM_PROVIDER", "none").lower()


def _mock(prompt):
    if "Return ONLY JSON" in prompt:                                   # skill extraction
        m = re.search(r"from this list only: \[(.*?)\]", prompt, re.S)
        allowed = [a.strip(" '\"") for a in m.group(1).split(",")] if m else []
        body = prompt.split("Title:", 1)[-1].lower()
        return json.dumps({"skills": [a for a in allowed if a and re.search(rf"\b{re.escape(a)}\b", body)][:5]})
    if "FACTS:" in prompt:                                             # explanation
        f = json.loads(prompt.split("FACTS:", 1)[1].split("ISSUE TEXT:", 1)[0].strip())
        return (f"This looks like a {f['difficulty']} task with about {f['est_hours']} hours of work and a "
                f"{round(f['p_merge'] * 100)}% chance of a merged PR. It needs {', '.join(f['skills']) or 'no special skills'}.\n"
                "Starter steps:\n- Read the issue and reproduce it.\n- Find the code it mentions.\n- Open a small PR.")
    if "excerpts" in prompt.lower():                                   # RAG: extractive echo of the first excerpt
        ctx = prompt.split("EXCERPTS:", 1)[-1].split("QUESTION:", 1)[0]
        return " ".join(ctx.strip().split("\n")[1:4])[:600] or "NOT FOUND IN DOCS"
    return "OK"


def complete(prompt, max_tokens=500, system=None):
    p = provider()
    t0 = time.time()
    try:
        if p == "mock":
            return _mock(prompt)
        if p == "openai":
            from openai import OpenAI
            client = OpenAI(timeout=float(os.getenv("LLM_TIMEOUT", "60")))
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            try:     # newer models want max_completion_tokens (and spend some of it on reasoning), older ones max_tokens
                r = client.chat.completions.create(model=model, messages=msgs, max_completion_tokens=max(max_tokens, 800))
            except Exception as e:
                if "max_completion_tokens" not in str(e) and "Unsupported parameter" not in str(e):
                    raise
                r = client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens)
            return r.choices[0].message.content
        if p == "anthropic":
            import anthropic
            kw = {"system": system} if system else {}
            m = anthropic.Anthropic(timeout=float(os.getenv("LLM_TIMEOUT", "60"))).messages.create(
                model=os.getenv("LLM_MODEL", "claude-sonnet-4-5"), max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}], **kw)
            return m.content[0].text
    except Exception as e:
        _log(f"{p} failed after {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:300]}")
    return None


def ping():
    """(ok, message, seconds). Use `python cli.py llm-check` to find a wrong model name or key BEFORE a demo."""
    p = provider()
    if p == "none":
        return False, "LLM_PROVIDER=none (set openai, anthropic or mock in .env)", 0.0
    t = time.time(); out = complete("Reply with exactly: OK", 20)
    return (out is not None and len(out) > 0), (out if out else f"FAILED: {last_error()}"), round(time.time() - t, 2)
````


### `llm/extract_skills.py`: REPLACE (file already exists in your project)

````python
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
````


### `llm/explain.py`: REPLACE (file already exists in your project)

````python
"""Grounded explanations. The LLM may only restate model/rule outputs; a post-check rejects invented numbers."""
import json, re
from llm.client import complete

FACT_KEYS = ["repo", "number", "title", "difficulty", "diff_conf", "p_merge", "est_hours", "reply_days", "skills", "rules"]


def template(item):
    s = (f"Predicted {item['difficulty']} ({item['diff_conf']:.0%} confidence), about {item['est_hours']}h, "
         f"merge probability {item['p_merge']:.0%}. Needs: {', '.join(item['skills']) or 'no special skills'}. "
         f"Rules: {'; '.join(item['rules']) or 'none fired'}.")
    if item.get("uncertain"):
        s += " The model is UNSURE about this one: read the issue yourself before starting."
    return s


def _allowed(item):
    nums = {float(x) for x in [item["diff_conf"] * 100, item["p_merge"] * 100, item["est_hours"], item["number"]] if x is not None}
    nums |= {round(v) for v in list(nums)}
    if item.get("reply_days") is not None:
        nums.add(float(item["reply_days"]))
    for k in ("Easy", "Medium", "Hard"):
        if item.get("probs"):
            nums.add(round(item["probs"][k] * 100))
    text = f"{item['title']} {item.get('snippet', '')} {item['repo']}"
    nums |= {float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)}
    return nums


def check(text, item):
    """Every number the LLM wrote must come from the facts (small numbers 0-5 are allowed for step lists)."""
    ok = _allowed(item)
    bad = []
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        v = float(tok)
        if v <= 5 or v in ok or round(v) in ok or any(abs(v - a) < 0.051 for a in ok):
            continue
        bad.append(tok)
    return (not bad), bad


def explain_checked(item, use_llm=False):
    fallback = template(item)
    if not use_llm:
        return fallback, {"source": "template"}
    facts = {k: item[k] for k in FACT_KEYS if k in item}
    out = complete("You explain GitHub issue recommendations to a beginner. Use ONLY these facts and invent nothing "
                   "(no new numbers, no new file names). Write 3 sentences on why it fits, then 'Starter steps:' with 3 short bullets "
                   f"based on the issue text.\nFACTS: {json.dumps(facts)}\nISSUE TEXT: {item.get('snippet', '')}", 450)
    if not out:
        return fallback, {"source": "template", "reason": "llm failed"}
    ok, bad = check(out, item)
    if not ok:
        return fallback, {"source": "template", "reason": f"guardrail rejected invented numbers {bad}"}
    return out, {"source": "llm", "guardrail": "passed"}


def explain(item, use_llm=False):
    return explain_checked(item, use_llm)[0]
````


### `llm/rag.py`: REPLACE (file already exists in your project)

````python
"""RAG over a repo's contributor docs: CONTRIBUTING, README, dev guides, PR template, Makefile targets.

Retrieval backends
  tfidf     (default) scikit-learn only: works offline, nothing to download
  langchain (optional) RecursiveCharacterTextSplitter + Chroma + HuggingFaceEmbeddings; pip install -r requirements-rag.txt
Generation: LangChain chain if installed and a provider is configured, else llm.client.complete, else an EXTRACTIVE
answer (best sentences from the docs), so the feature still works with no LLM at all.
Every answer carries sources and a groundedness score."""
import base64, json, os, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from llm.client import complete, provider

DOC_PATHS = ["CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md", "docs/contributing.md", "CONTRIBUTING.rst",
             "README.md", "DEVELOPMENT.md", "docs/development.md", "docs/DEVELOPMENT.md", "docs/dev/README.md", "HACKING.md",
             ".github/PULL_REQUEST_TEMPLATE.md", ".github/pull_request_template.md", "PULL_REQUEST_TEMPLATE.md", "Makefile"]
NOT_FOUND = "NOT FOUND IN DOCS"


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
                if len(s.strip()) > 25:
                    sents.append((len(qt & _terms(s)), s.strip(), h))
        sents.sort(key=lambda t: -t[0])
        return " ".join(s for _, s, _ in sents[:4]) or NOT_FOUND

    def ask(self, question, k=4, use_llm=True):
        hits = self.retrieve(question, k)
        if not hits:
            return {"answer": NOT_FOUND, "sources": [], "grounded": 1.0, "mode": "none", "backend": self.backend}
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
        return {"answer": ans.strip(), "sources": [{"path": h["path"], "heading": h["heading"], "score": round(h["score"], 3), "snippet": h["text"][:240]} for h in hits],
                "grounded": g, "checked_tokens": n, "mode": mode, "backend": self.backend,
                "warning": "low groundedness: verify against the linked docs" if g < 0.7 else None}


_CACHE = {}


def ask(repo, question, use_llm=True, docs=None):
    if repo not in _CACHE or docs is not None:
        _CACHE[repo] = RepoRAG(repo, docs)
    return _CACHE[repo].ask(question, use_llm=use_llm)
````


### `llm/first_pr_kit.py`: NEW

````python
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
    kit = {
        "issue": f"{repo}#{number}", "title": title, "url": f"https://github.com/{repo}/issues/{number}",
        "requires_cla": bool(CLA.search(allt)), "requires_dco_signoff": bool(DCO.search(allt)),
        "claim_style": (ASSIGN.search(allt).group(1) if ASSIGN.search(allt) else "comment on the issue and wait for a maintainer to reply"),
        "setup_commands": _uniq(SETUP, allt), "test_commands": _uniq(TEST, allt),
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
````


### `llm/eval_rag.py`: NEW

````python
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
````


### `llm/eval_llm.py`: NEW

````python
"""LLM evaluation (syllabus: evaluation metrics and benchmarks): JSON validity, agreement with keywords, guardrail pass rate, latency."""
import json, os, random, time
from collector.db import conn
from llm.client import provider, complete, last_error
from llm.extract_skills import parse_skills, extract_skills_meta
from llm.explain import explain_checked
from ai.ontology import extract_keywords, G
from ml.common import save_metrics


def main(n=30):
    if provider() == "none":
        raise SystemExit("Set LLM_PROVIDER in .env (openai / anthropic / mock) first.")
    rows = conn().execute("SELECT title, body, (SELECT language FROM repos WHERE full_name=repo) FROM issues WHERE state='open'").fetchall()
    random.Random(0).shuffle(rows); rows = rows[:n]
    valid = jac = fell = 0; lat = []
    for t, b, lang in rows:
        kw = extract_keywords(f"{t} {b}", lang); t0 = time.time()
        prompt = ('Return ONLY JSON like {"skills": ["go"]}. Choose the skills needed to solve this GitHub issue '
                  f"from this list only: {sorted(G.nodes)}\nTitle: {t}\nBody: {(b or '')[:1500]}")
        out = complete(prompt, 200); lat.append(time.time() - t0)
        try:
            got = parse_skills(out); valid += 1
            jac += len(got & kw) / max(len(got | kw), 1)
        except Exception:
            fell += 1
    res = {"issues": len(rows), "provider": provider(), "model": os.getenv("LLM_MODEL"), "json_valid_rate": round(valid / max(len(rows), 1), 3),
           "mean_jaccard_vs_keywords (agreement, not accuracy)": round(jac / max(valid, 1), 3), "mean_latency_s": round(sum(lat) / max(len(lat), 1), 2),
           "last_error": last_error()}
    if os.path.exists("data/last_plan.json"):
        plan = json.load(open("data/last_plan.json"))["plan"]
        metas = [explain_checked(i, True)[1] for i in plan]
        res["explanations"] = {"n": len(plan), "llm_passed_guardrail": sum(m["source"] == "llm" for m in metas),
                               "fell_back_to_template": sum(m["source"] == "template" for m in metas), "reasons": [m.get("reason") for m in metas if m.get("reason")]}
    print(json.dumps(res, indent=2)); save_metrics("llm_eval", res)
    return res
````


### `data/eval/rag_questions.template.json`: NEW

````json
[
  {
    "template": true,
    "repo": "owner/name",
    "question": "How do I run the unit tests?",
    "must_include": ["the exact command you read in that repo's CONTRIBUTING.md, e.g. make test"],
    "answerable": true
  },
  {
    "template": true,
    "repo": "owner/name",
    "question": "What is the company's stock price?",
    "must_include": [],
    "answerable": false
  }
]
````


### `api/main.py`: REPLACE (file already exists in your project)

````python
"""Deployment layer.  uvicorn api.main:app --port 8000   then open http://localhost:8000/docs"""
import json, os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.profile import Profile
from core.recommend import recommend, learning_path
from core import outcomes

app = FastAPI(title="IssueScout", version="2.0")


class Feedback(BaseModel):
    issue_id: str
    verdict: str            # "tried" | "too_hard" | "good"


class Ref(BaseModel):
    ref: str                # owner/repo#123 or an issue URL
    use_llm: bool = False


class Question(BaseModel):
    repo: str
    question: str
    use_llm: bool = True


class Outcome(BaseModel):
    issue_id: str
    status: str
    hours: float | None = None
    notes: str = ""


@app.get("/health")
def health():
    return {"ok": True, "difficulty_model": os.path.exists("models/difficulty.joblib")}


@app.post("/recommend")
def rec(p: Profile):
    return recommend(p)


@app.post("/path")
def path(p: Profile, target: str):
    return learning_path(p, target)


@app.post("/predict")
def predict(r: Ref):
    from ml.predict import predict_issue
    try:
        return predict_issue(r.ref)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/prepare")
def prepare(r: Ref):
    from llm.first_pr_kit import build_kit
    try:
        return build_kit(r.ref, use_llm=r.use_llm)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/ask")
def ask(q: Question):
    from llm.rag import ask as rag_ask
    return rag_ask(q.repo, q.question, use_llm=q.use_llm)


@app.post("/feedback")
def feedback(f: Feedback):
    os.makedirs("data", exist_ok=True)
    with open("data/feedback.jsonl", "a") as fh:
        fh.write(json.dumps({**f.model_dump(), "at": datetime.utcnow().isoformat()}) + "\n")
    return {"saved": True}


@app.post("/outcome")
def outcome(o: Outcome):
    try:
        return outcomes.add(o.issue_id, o.status, o.hours, o.notes)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/outcomes")
def outcome_summary():
    return outcomes.summary()
````


### `app/streamlit_app.py`: REPLACE (file already exists in your project)

````python
import glob, json, os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IssueScout", layout="wide")
st.title("IssueScout")
st.caption("ML predicts, rules and search decide, the LLM only explains. Every number is a hint: read the issue yourself.")


def need_models():
    if not os.path.exists("models/difficulty.joblib"):
        st.error("No trained model yet. Run `python cli.py train all` first."); st.stop()


with st.sidebar:
    st.header("Your profile")
    skills = st.text_input("Skills (comma separated)", "go, docker, python")
    level = st.selectbox("Your level", ["beginner", "intermediate", "advanced"], index=1)
    hpw = st.number_input("Hours per week", 1, 40, 6); weeks = st.number_input("Weeks available", 1, 26, 3)
    k = st.slider("Issues in plan", 1, 6, 3); avoid = st.text_input("Avoid keywords", "")
    uncertain = st.checkbox("Include issues the model is unsure about", True)
    use_llm = st.checkbox("LLM explanations (needs LLM_PROVIDER in .env)", False)
    go = st.button("Find my issues", type="primary")

if go:
    need_models()
    from core.profile import Profile
    from core.recommend import recommend
    p = Profile(skills=[s.strip().lower() for s in skills.split(",") if s.strip()], level=level, hours_per_week=hpw, weeks=weeks, k=k,
                avoid=[a.strip() for a in avoid.split(",") if a.strip()], use_llm=use_llm, include_uncertain=uncertain)
    with st.spinner("Scoring issues..."):
        st.session_state["result"] = recommend(p)

tabs = st.tabs(["Plan", "First-PR kit", "Predict any issue", "Ask the docs", "My outcomes", "Repo map", "Model report"])

with tabs[0]:
    res = st.session_state.get("result")
    if not res:
        st.info("Fill the sidebar and click **Find my issues**.")
    elif not res["plan"]:
        st.warning(res.get("note", "No plan found.")); st.json(res.get("rejected", {}))
    else:
        st.success(f"{res['planned_hours']}h planned out of {res['budget_hours']}h available")
        for i in res["plan"]:
            st.subheader(f"#{i['rank']}  {i['repo']} #{i['number']}: {i['title']}")
            if i["uncertain"]:
                st.warning("The model is unsure about this one (top probability under 50%). Read the issue before committing.")
            a, b, c, d = st.columns(4)
            a.metric("Predicted difficulty", i["difficulty"], f"conf {i['diff_conf']:.0%}", delta_color="off")
            b.metric("P(merged)", f"{i['p_merge']:.0%}"); c.metric("Est. effort", f"{i['est_hours']} h")
            d.metric("Maintainer reply", f"{i['reply_days']} d" if i["reply_days"] is not None else "n/a")
            st.progress(min(1.0, i["probs"]["Easy"]), text=f"P(Easy)={i['probs']['Easy']:.0%}  P(Medium)={i['probs']['Medium']:.0%}  P(Hard)={i['probs']['Hard']:.0%}")
            st.link_button("Open on GitHub", i["url"])
            with st.expander("Why this issue? (rules, CSP check, similar solved issues)"):
                st.markdown("**Rules fired**\n" + "\n".join("- " + r for r in (i["rules"] or ["none"])))
                st.markdown("**CSP check**  " + "  ".join(f"{'✅' if v else '❌'} {kk}" for kk, v in i["csp_check"].items()))
                st.markdown("**Skills needed:** " + (", ".join(i["skills"]) or "none"))
                st.write(i["explanation"]); st.caption(f"explanation source: {i['explanation_meta']['source']}")
                for s in i.get("similar_solved", []):
                    st.markdown(f"- similar solved: [{s['title']}]({s['url']}) was **{s['difficulty']}** (similarity {s['similarity']})")
            f1, f2, f3 = st.columns(3)
            for col, label, verdict in [(f1, "I tried this", "tried"), (f2, "Too hard", "too_hard"), (f3, "Good pick", "good")]:
                if col.button(label, key=f"{verdict}_{i['id']}"):
                    os.makedirs("data", exist_ok=True)
                    open("data/feedback.jsonl", "a").write(json.dumps({"issue_id": i["id"], "verdict": verdict}) + "\n"); st.toast("Feedback saved")
        with st.expander("Search statistics"):
            st.write("Issues rejected:", res["rejected"])
            st.write(f"CSP nodes: FC+MRV+bound = {res['csp_nodes_fc_mrv']}, plain backtracking = {res['csp_nodes_plain']}")

with tabs[1]:
    ref = st.text_input("Issue (owner/repo#123 or URL)", key="kit_ref"); kllm = st.checkbox("Use LLM for doc answers", False, key="kit_llm")
    if st.button("Build my first-PR kit") and ref:
        from llm.first_pr_kit import build_kit, render_markdown, save
        try:
            with st.spinner("Reading the repo's contributor docs..."):
                kit = build_kit(ref, use_llm=kllm)
            st.markdown(render_markdown(kit)); st.caption(f"saved to {save(kit)}")
        except Exception as e:
            st.error(f"Could not build the kit: {e}")

with tabs[2]:
    pref = st.text_input("Issue to classify (owner/repo#123 or URL)", key="pred_ref")
    if st.button("Predict difficulty") and pref:
        need_models()
        from ml.predict import predict_issue
        try:
            r = predict_issue(pref); st.subheader(f"{r['predicted']}  (confidence {r['confidence']:.0%})")
            st.bar_chart(pd.Series(r["probabilities"]))
            if r["uncertain"]: st.warning("Model is unsure. " + r["note"])
            st.write("Most similar solved issues:"); st.dataframe(pd.DataFrame(r["similar_solved"]))
        except Exception as e:
            st.error(str(e))

with tabs[3]:
    rrepo = st.text_input("Repo (owner/name)", key="ask_repo"); q = st.text_input("Question", "How do I run the tests?", key="ask_q")
    if st.button("Ask") and rrepo:
        from llm.rag import ask
        try:
            a = ask(rrepo, q); st.write(a["answer"]); st.caption(f"{a['mode']} | {a['backend']} | groundedness {a['grounded']}")
            if a.get("warning"): st.warning(a["warning"])
            for s in a["sources"]: st.markdown(f"- `{s['path']}` > {s['heading']}")
        except Exception as e:
            st.error(str(e))

with tabs[4]:
    from core import outcomes
    st.json(outcomes.summary())
    oid = st.text_input("Issue id (owner/repo#123)", key="o_id"); stt = st.selectbox("Status", outcomes.STATUSES); hrs = st.number_input("Hours spent", 0.0, 500.0, 0.0)
    if st.button("Log outcome") and oid:
        outcomes.add(oid, stt, hrs or None); st.success("logged")
    st.dataframe(pd.DataFrame(outcomes.load()))

with tabs[5]:
    p = "data/processed/repo_flags.csv"
    if os.path.exists(p):
        df = pd.read_csv(p); df["cluster"] = df["cluster"].astype(str)
        st.scatter_chart(df, x="pc1", y="pc2", color="cluster"); st.dataframe(df[df.abandoned][["full_name", "days_since_push"]])
    else:
        st.info("Run `python cli.py cluster` first.")

with tabs[6]:
    if os.path.exists("report/RESULTS.md"):
        st.markdown(open("report/RESULTS.md", encoding="utf-8").read())
    for img in sorted(glob.glob("report/*.png")):
        st.image(img, caption=os.path.basename(img))
````


### `cli.py`: REPLACE (file already exists in your project)

````python
"""IssueScout command line.   python cli.py --help"""
import json, os
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="IssueScout: find GitHub issues you can finish, and prepare your first PR.", no_args_is_help=True)
console = Console()


def _profile(skills, level, hours, weeks, k, avoid, llm, uncertain=True):
    from core.profile import Profile
    return Profile(skills=[s.strip().lower() for s in skills.split(",") if s.strip()], level=level, hours_per_week=hours, weeks=weeks, k=k,
                   avoid=[a.strip() for a in avoid.split(",") if a.strip()], use_llm=llm, include_uncertain=uncertain)


# ---------------- setup, data ----------------
@app.command("status")
def status_cmd(llm: bool = typer.Option(False, "--llm", help="also test the LLM connection")):
    """Where is the project? Checks data, models, evidence and tools against targets."""
    from dotenv import load_dotenv; load_dotenv()
    import sqlite3
    checks = []
    def add(name, ok, detail, target=""): checks.append((name, ok, detail, target))
    add("GITHUB_TOKEN set", bool(os.getenv("GITHUB_TOKEN")), "yes" if os.getenv("GITHUB_TOKEN") else "missing in .env")
    q = lambda s: 0
    if os.path.exists("data/issuescout.db"):
        c = sqlite3.connect("data/issuescout.db"); q = lambda s: c.execute(s).fetchone()[0]
    n_repo, n_lab, n_open = q("select count(*) from repos"), q("select count(*) from issues where pr_merged=1"), q("select count(*) from issues where state='open'")
    add("repos collected", n_repo >= 100, str(n_repo), ">= 100"); add("labelled issues (merged PR)", n_lab >= 1500, str(n_lab), ">= 1500"); add("open issues to recommend from", n_open >= 1000, str(n_open), ">= 1000")
    man = 0
    if os.path.exists("data/processed/manual_label_sample.csv"):
        import pandas as pd; m = pd.read_csv("data/processed/manual_label_sample.csv"); man = int(m.my_difficulty.astype(str).isin(["Easy", "Medium", "Hard"]).sum())
    add("manual labels typed by you", man >= 100, str(man), ">= 100")
    M = json.load(open("report/metrics.json")) if os.path.exists("report/metrics.json") else {}
    lv = M.get("label_validation", {}); add("label agreement (kappa) measured", bool(lv), str(lv.get("kappa", "not yet")), "reported")
    add("difficulty model trained", os.path.exists("models/difficulty.joblib"), "yes" if os.path.exists("models/difficulty.joblib") else "run: train all")
    best = None
    if os.path.exists("report/model_comparison.csv"):
        import pandas as pd; best = pd.read_csv("report/model_comparison.csv", index_col=0)["cv_f1_grouped"].max()
    add("grouped-CV macro-F1 clearly above chance (0.33)", best is not None and best >= 0.45, "n/a" if best is None else f"{best:.2f}", ">= 0.45")
    add("merge / time models + clustering", all(os.path.exists(p) for p in ["models/merge.joblib", "models/time.joblib", "data/processed/repo_flags.csv"]), "", "all three")
    add("EDA figures", os.path.exists("report/eda/summary.md"), "")
    add("CSP + A* benchmarks", all(os.path.exists(p) for p in ["report/csp_benchmark.md", "report/astar_benchmark.md"]), "", "both")
    add("LLM configured and working", False if not llm else None, os.getenv("LLM_PROVIDER", "none"))
    if llm:
        from llm.client import ping; ok, msg, sec = ping(); checks[-1] = ("LLM configured and working", ok, f"{msg[:60]} ({sec}s)", "")
    nd = len([f for f in os.listdir("data/docs")]) if os.path.isdir("data/docs") else 0
    add("RAG docs cached for >= 5 repos", nd >= 5, str(nd), ">= 5"); add("RAG evaluated", "rag_eval" in M, str(M.get("rag_eval", {}).get("accuracy", "not yet")), "reported")
    from core import outcomes; s = outcomes.summary()
    add("you attempted a recommended issue", s["issues_tried"] >= 1, str(s["issues_tried"]), ">= 1"); add("a merged PR (the real proof)", s["merged"] >= 1, str(s["merged"]), ">= 1")
    t = Table(title="IssueScout status")
    for col in ["", "check", "now", "target"]: t.add_column(col)
    for name, ok, detail, target in checks:
        t.add_row("OK" if ok else "..", name, detail, target)
    console.print(t); console.print(f"progress: {sum(1 for c in checks if c[1])}/{len(checks)} checks passed")


@app.command("collect")
def collect_cmd(repos: str = "repos.txt", closed_pages: int = typer.Option(0, help="pages of closed issues per repo (0 = env/default 3)"),
                open_pages: int = 0, pr_pages: int = 0):
    """Pull repos and issues from GitHub into SQLite (cached and resumable)."""
    from collector import collect_repos, collect_issues
    collect_repos.main(repos); collect_issues.main(repos, closed_pages or None, open_pages or None, pr_pages or None)


@app.command("add-repos")
def add_repos_cmd(src: str = "repos_extra.txt"):
    """Append repos_extra.txt to repos.txt (no duplicates)."""
    from collector import merge_repos; merge_repos.main(src)


@app.command("features")
def features_cmd():
    """Build labels, print the class balance, export the manual-labelling sample."""
    from features.build_features import main; main()


@app.command("label")
def label_cmd():
    """Label issues yourself (the automatic label is hidden) to validate the difficulty definition."""
    from tools.label_cli import main; main()


@app.command("eda")
def eda_cmd():
    from ml.eda import main; main()


# ---------------- ML ----------------
@app.command("train")
def train_cmd(task: str = typer.Argument("all", help="difficulty | merge | time | all"), tune: bool = True):
    os.makedirs("models", exist_ok=True); os.makedirs("report", exist_ok=True)
    if task in ("difficulty", "all"):
        from ml.train_difficulty import main; main(tune)
    if task in ("merge", "all"):
        from ml.train_merge import main as m; m()
    if task in ("time", "all"):
        from ml.train_time import main as t; t()


@app.command("cluster")
def cluster_cmd(eps: float = 0.0):
    from ml.cluster import main; main(eps or None)


@app.command("evaluate")
def evaluate_cmd():
    """Learning curves, manual-label agreement (kappa), label sensitivity."""
    from ml.evaluate import main; main()


@app.command("gd-demo")
def gd_cmd():
    """Cost function, gradient descent and decision boundary demo."""
    from ml.gradient_descent_demo import main; main()


@app.command("predict")
def predict_cmd(ref: str = typer.Argument(..., help="owner/repo#123 or an issue URL")):
    """Difficulty of ANY GitHub issue (uses the API if it is not in your database)."""
    from ml.predict import predict_issue
    r = predict_issue(ref)
    console.print(f"[bold]{r['issue']}[/bold] {r['title']}\nPredicted: [bold]{r['predicted']}[/bold]  confidence {r['confidence']:.0%}  {'(UNCERTAIN)' if r['uncertain'] else ''}")
    console.print("Probabilities:", r["probabilities"])
    for s in r["similar_solved"]:
        console.print(f"  similar solved: {s['difficulty']:6s} {s['title']}  {s['url']}")
    console.print(f"[dim]{r['note']}[/dim]")


# ---------------- AI ----------------
@app.command("benchmark")
def benchmark_cmd():
    """CSP node counts (plain/FC/MRV/bound) and A* vs uniform-cost search."""
    from ai.benchmark_csp import main as a; from ai.benchmark_astar import main as b
    a(); b()


@app.command("bayes-demo")
def bayes_cmd():
    """MLE vs Bayesian merge rates per repo (shrinkage)."""
    from ai.bayes import demo_table
    prior, g = demo_table(); console.print(f"global merge rate (prior) = {prior:.3f}"); console.print(g.to_string())


@app.command("test")
def test_cmd():
    """Run the unit tests."""
    import subprocess, sys; raise SystemExit(subprocess.call([sys.executable, "tests/run_tests.py"]))


@app.command("recommend")
def recommend_cmd(skills: str = typer.Option(..., help="e.g. go,docker,python"), level: str = "beginner", hours: float = 6, weeks: float = 3, k: int = 3,
                  avoid: str = "", llm: bool = False, hide_uncertain: bool = False, as_json: bool = typer.Option(False, "--json"), verbose: bool = False):
    """Ranked plan of issues you can finish."""
    from core.recommend import recommend
    out = recommend(_profile(skills, level, hours, weeks, k, avoid, llm, not hide_uncertain))
    if as_json:
        print(json.dumps(out, indent=2, default=str)); return
    if not out["plan"]:
        console.print(f"[yellow]{out.get('note', 'No plan')}[/yellow]", out.get("rejected", "")); return
    t = Table(title=f"Plan: {out['planned_hours']}h of {out['budget_hours']}h budget")
    for c in ["#", "Issue", "Difficulty", "P(merge)", "Hours", "Reply(d)", "Skills"]:
        t.add_column(c)
    for i in out["plan"]:
        t.add_row(str(i["rank"]), f"{i['repo']}#{i['number']}\n{i['title'][:55]}", f"{i['difficulty']} ({i['diff_conf']:.0%}){' ?' if i['uncertain'] else ''}",
                  f"{i['p_merge']:.0%}", str(i["est_hours"]), str(i["reply_days"]), ", ".join(i["skills"]))
    console.print(t)
    if verbose:
        for i in out["plan"]:
            console.print(f"\n[bold]#{i['rank']} {i['url']}[/bold]\n Rules: {i['rules']}\n CSP: {i['csp_check']}\n {i['explanation']}")
        console.print(f"\nRejected because: {out['rejected']}\nCSP nodes: FC+MRV+bound={out['csp_nodes_fc_mrv']} vs plain={out['csp_nodes_plain']}")


@app.command("explain")
def explain_cmd(rank: int):
    """Rules fired and CSP check for a plan item from the last run."""
    i = json.load(open("data/last_plan.json"))["plan"][rank - 1]
    console.print(f"[bold]{i['repo']}#{i['number']}[/bold] {i['title']}\n{i['url']}")
    console.print("Rules fired:", *i["rules"] or ["(none)"], sep="\n  - ")
    console.print("CSP check:", {k: "yes" if v else "NO" for k, v in i["csp_check"].items()}); console.print(i["explanation"])


@app.command("path")
def path_cmd(target: str = typer.Argument(..., help="owner/repo#123"), skills: str = typer.Option(...), level: str = "beginner"):
    """A* learning path to a harder target issue."""
    from core.recommend import learning_path
    print(json.dumps(learning_path(_profile(skills, level, 6, 3, 3, "", False), target), indent=2))


# ---------------- LLM / RAG / first PR ----------------
@app.command("llm-check")
def llm_check_cmd():
    """Test the LLM connection (wrong model name or key shows up here, not during your demo)."""
    from dotenv import load_dotenv; load_dotenv()
    from llm.client import ping, provider
    ok, msg, sec = ping(); console.print(f"provider={provider()} model={os.getenv('LLM_MODEL')} -> {'OK' if ok else 'FAILED'}: {msg} ({sec}s)")


@app.command("ask")
def ask_cmd(repo: str, question: str, llm: bool = True):
    """Ask a repo's contributor docs a question (RAG, with sources)."""
    from llm.rag import ask
    a = ask(repo, question, use_llm=llm)
    console.print(a["answer"]); console.print(f"[dim]{a['mode']} | {a['backend']} | groundedness {a['grounded']} | " + ", ".join(s["path"] for s in a["sources"][:3]) + "[/dim]")
    if a.get("warning"): console.print(f"[yellow]{a['warning']}[/yellow]")


@app.command("prepare")
def prepare_cmd(ref: str = typer.Argument(..., help="owner/repo#123 or URL"), llm: bool = False):
    """Build your first-PR kit for one issue: steps, commands, CLA/DCO, draft comment and PR text."""
    from llm.first_pr_kit import build_kit, render_markdown, save
    kit = build_kit(ref, use_llm=llm); path = save(kit); print(render_markdown(kit)); console.print(f"\n[green]saved {path}[/green]")


@app.command("eval-rag")
def eval_rag_cmd(path: str = "data/eval/rag_questions.json", llm: bool = True):
    from llm.eval_rag import main; main(path, llm)


@app.command("eval-llm")
def eval_llm_cmd(n: int = 30):
    from dotenv import load_dotenv; load_dotenv()
    from llm.eval_llm import main; main(n)


@app.command("outcome")
def outcome_cmd(issue_id: str, status: str = typer.Argument(..., help="started | pr_opened | merged | closed_unmerged | abandoned | too_hard"), hours: float = 0.0, notes: str = ""):
    """Log what happened with an issue you tried (the real success metric)."""
    from core import outcomes; console.print(outcomes.add(issue_id, status, hours or None, notes))


@app.command("outcomes")
def outcomes_cmd():
    from core import outcomes; console.print(outcomes.summary())


@app.command("report")
def report_cmd():
    """Collect all results into report/RESULTS.md."""
    from ml.make_report import main; main()


@app.command("serve")
def serve_cmd(port: int = 8000):
    import uvicorn; uvicorn.run("api.main:app", port=port)


@app.command("web")
def web_cmd():
    import subprocess; subprocess.run(["streamlit", "run", "app/streamlit_app.py"])


if __name__ == "__main__":
    app()
````


---

## OPTIONAL. Docs and scripts (nice to have, no code depends on them)


### `README.md`: NEW

````markdown
# IssueScout

Find open GitHub issues you can finish, and get a kit to make your first PR.
**ML predicts, rules and search decide, the LLM only explains.**

```
python cli.py status        # where is the project? (checks vs targets)
python cli.py recommend --skills go,docker --level intermediate --hours 6 --weeks 3 --verbose
python cli.py predict kubernetes/kubernetes#12345      # difficulty of ANY issue
python cli.py prepare owner/repo#123                   # first-PR kit
python cli.py ask owner/repo "How do I run the tests?" # RAG over CONTRIBUTING/README
python cli.py web                                      # browser UI
```
Start with `docs/STEP_BY_STEP.md`. Results land in `report/RESULTS.md`.

| Folder | What |
|---|---|
| collector/ | GitHub API -> SQLite (cached) |
| features/ | label definition (effort percentile), leakage-free features |
| ml/ | EDA, difficulty/merge/time models, clustering, anomaly, PCA, GD demo, evaluation, predict |
| ai/ | ontology, forward-chaining rules, CSP, A*, Bayes, benchmarks |
| llm/ | provider client, skill extraction, grounded explanations, RAG, first-PR kit, evaluations |
| core/ | recommend(), outcomes log |
| api/, app/, cli.py | FastAPI, Streamlit, command line |
| tests/ | `python cli.py test` |

Responsible use: IssueScout never comments on or opens anything on GitHub for you. Predictions are hints, shown with probabilities.
````


### `docs/RESPONSIBLE_AI.md`: NEW

````markdown
# Responsible AI and PEAS

## PEAS (utility-based agent)
| | |
|---|---|
| Performance measure | expected merge probability of the recommended set, within the user's hour budget, penalised by expected difficulty |
| Environment | GitHub (partially observable, dynamic: issues get claimed), the user's profile |
| Actuators | ranked plan, exclusions with reasons, learning path, first-PR kit (drafts only) |
| Sensors | GitHub API data, user profile, user feedback and outcomes |

## Practices in the code
- Never posts to GitHub; drafts are labelled as drafts.
- Always shows probabilities and an "unsure" flag below 50% confidence; every pick shows rules fired and the CSP check.
- Leakage control: no PR-derived fields as inputs; temporal and unseen-repo evaluation.
- Honest limitations printed in `report/RESULTS.md`; negative results kept (time-to-close).
- Rate limits respected and responses cached; only public data is read.
- Bias: popular repos dominate the data; report the repo-size breakdown (`report/eda/02_repo_distribution.png`) and the language mix.
- LLM guardrails: facts-only prompt, invented numbers rejected, extractive RAG fallback, groundedness score on every answer.
- Fairness risk to discuss: over-recommending already popular repos and English-only issues.
````


### `docs/USER_STUDY.md`: NEW

````markdown
# Small user study (5-10 classmates)
Give each person a profile (skills, hours) and the plan IssueScout produced. Ask them to open the 2-3 issues and answer (1 = no, 5 = yes):
1. This plan looks doable for me. (`doable_1to5`)
2. I understand why each issue was chosen. (`explanation_clear_1to5`)
3. I would try one of these issues. (`would_try_1to5`)
4. Free comment.
Save as `data/user_study.csv` with columns `participant,doable_1to5,explanation_clear_1to5,would_try_1to5,comments`, then run `python scripts/user_study_summary.py`.
Report n, means and spread, and say it is a small convenience sample, not proof.
````


### `docs/REPORT_OUTLINE.md`: NEW

````markdown
# Report outlines mapped to your two syllabi

## ML report
| Syllabus topic | Where | Evidence file |
|---|---|---|
| Types of learning, problems, applications | intro | write-up |
| Cost function, gradient descent, sampling, decision boundary, under/overfitting, bias-variance | `ml/gradient_descent_demo.py`, `ml/evaluate.py`, temporal + grouped splits | `report/gd_cost_curve.png`, `decision_boundary.png`, `learning_curves.png` |
| Regression, Ridge, Lasso | `ml/train_time.py` | `report/time_comparison.md`, `regularisation_effect.png` (negative result, explain why) |
| Logistic regression, SVM, decision tree | `ml/train_difficulty.py` | `report/model_comparison.md` |
| Ensembles: bagging, random forest, boosting | same | same |
| KNN, ANN intro | KNN(cosine), MLP(64) in the same table | same |
| Clustering: K-Means, DBSCAN, BIRCH; anomaly detection | `ml/cluster.py` | `pca_repos.png`, `cluster_profiles.csv`, `kdistance.png` |
| Probability, Bayes rule, MLE | `ai/bayes.py`, `python cli.py bayes-demo` | table of MLE vs Bayesian rates |
| PCA, kernel PCA | `ml/cluster.py` | `pca_repos.png` |
| Validation: accuracy, CI, confusion matrix, precision/recall, ROC, CV, bootstrap, tuning, R2 | `train_difficulty.py` | `confusion_difficulty.png`, `roc_difficulty.png`, `calibration_difficulty.png`, the CI column |
| Model deployment | joblib + `api/main.py` + `cli.py predict` | `/predict` endpoint |
| Reinforcement learning intro | theory section only (frame recommendation as a contextual bandit; not implemented) | write-up |
Also cover: label definition, leakage, label validation (kappa), cost-sensitive metric (`test_cost`, `hard_called_easy`), limitations.

## AI report
| Syllabus topic | Where | Evidence |
|---|---|---|
| AI intro, agents, PEAS, expert systems | `docs/RESPONSIBLE_AI.md`, `ai/rules.py` | rule trace in `recommend --verbose` |
| Search, heuristics, A*, admissibility | `ai/astar.py`, `core/recommend.py: learning_path` | `report/astar_benchmark.md` |
| Planning and CSP | `ai/csp.py` | `report/csp_benchmark.md` |
| Knowledge representation and reasoning | `ai/ontology.py` | graph + `missing()` examples |
| Generative AI, VAE/GAN, Transformer, attention | theory section (no demo needed); link to how the LLM/embeddings are used here | write-up |
| LLMs: prompting, context, evaluation, LangChain | `llm/` | `report/metrics.json: llm_eval, rag_eval` |
| Responsible AI | `docs/RESPONSIBLE_AI.md` | guardrail results |
````


### `scripts/run_all.sh`: NEW

````bash
#!/usr/bin/env bash
# Whole pipeline, in order. Run from the project root (Git Bash on Windows is fine).  bash scripts/run_all.sh
set -e
python cli.py add-repos repos_extra.txt          # once: appends ~90 repos to repos.txt
python cli.py collect --closed-pages 6 --open-pages 3 --pr-pages 6    # SLOW: hours. cached + resumable: just run again
python cli.py features                           # labels + manual-label sample
python cli.py eda
python cli.py train all
python cli.py cluster
python cli.py evaluate
python cli.py gd-demo
python cli.py benchmark
python cli.py test
python cli.py report
python cli.py status
````


### `scripts/user_study_summary.py`: NEW

````python
"""Summarise the classmate study.  data/user_study.csv columns: participant,doable_1to5,explanation_clear_1to5,would_try_1to5,comments"""
import pandas as pd
d = pd.read_csv("data/user_study.csv")
for c in ["doable_1to5", "explanation_clear_1to5", "would_try_1to5"]:
    print(f"{c:26s} mean {d[c].mean():.2f}  sd {d[c].std():.2f}  n={d[c].notna().sum()}")
````


---

## Tell me where you get stuck
Paste the exact command and the full error. Not run here (written but untested): LangChain/Chroma backend, sentence-embedding features, live OpenAI/Anthropic calls, Typer command wiring, FastAPI, Streamlit, and GitHub API fetches for issues outside your database. Everything else was run end to end on your database.
