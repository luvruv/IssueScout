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
def status_cmd(llm: bool = typer.Option(True, "--llm/--no-llm", help="test the LLM connection (default: True)")):
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
def add_repos_cmd(src: str = typer.Argument("repo_lists", help="Source file (.txt) or directory of .txt files to append")):
    """Append repos from a file or entire repo_lists/ directory to repos.txt (no duplicates)."""
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
def eval_rag_cmd(path: str = typer.Argument("data/eval/rag_questions.json"), llm: bool = typer.Option(True, "--llm/--no-llm")):
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


@app.command("solve-agent")
def solve_agent_cmd(ref: str = typer.Argument(..., help="owner/repo#123 or issue URL")):
    """Autonomous in-process AI agent: analyzes an issue, plans the fix, and writes code (no API key/Ollama required)."""
    from ml.predict import parse_ref
    from llm.local_agent import LocalIssueAgent
    import sqlite3
    
    full_repo, num = parse_ref(ref)
    
    title, body = f"Issue #{num}", ""
    if os.path.exists("data/issuescout.db"):
        c = sqlite3.connect("data/issuescout.db")
        row = c.execute("select title, body from issues where repo=? and number=?", (full_repo, num)).fetchone()
        if row:
            title, body = row[0], row[1] or ""
            
    console.print(f"[bold cyan]Local AI Agent starting for {full_repo}#{num}...[/bold cyan]")
    console.print(f"[bold]Title:[/bold] {title}\n")
    
    agent = LocalIssueAgent()
    solution = agent.analyze(full_repo, title, body)
    console.print("\n[bold green]=== AI AGENT PLAN & CODE FIX ===[/bold green]\n")
    console.print(solution)


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