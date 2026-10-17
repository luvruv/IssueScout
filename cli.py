import json
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="IssueScout CLI", no_args_is_help=True)
console = Console()


def _profile(skills, level, hours, weeks, k, avoid, llm):
    from core.profile import Profile
    return Profile(skills=[s.strip().lower() for s in skills.split(",") if s.strip()], level=level,
                   hours_per_week=hours, weeks=weeks, k=k,
                   avoid=[a.strip() for a in avoid.split(",") if a.strip()], use_llm=llm)


# ---------- admin commands ----------
@app.command("collect")
def collect_cmd(repos: str = "repos.txt"):
    """Pull repos and issues from GitHub into SQLite."""
    from collector import collect_repos, collect_issues
    collect_repos.main(repos); collect_issues.main(repos)


@app.command("features")
def features_cmd():
    """Show label distribution and export the manual-labelling sample."""
    from features.build_features import main; main()


@app.command("train")
def train_cmd(task: str = typer.Argument("all", help="difficulty | merge | time | all")):
    if task in ("difficulty", "all"):
        from ml.train_difficulty import main; main()
    if task in ("merge", "all"):
        from ml.train_merge import main; main()
    if task in ("time", "all"):
        from ml.train_time import main; main()


@app.command("cluster")
def cluster_cmd(eps: float = 1.5):
    from ml.cluster import main; main(eps)


@app.command("evaluate")
def evaluate_cmd():
    from ml.evaluate import main; main()


# ---------- user commands ----------
@app.command("recommend")
def recommend_cmd(skills: str = typer.Option(..., help="e.g. go,docker,python"),
                  level: str = "beginner", hours: float = 6, weeks: float = 3, k: int = 3,
                  avoid: str = "", llm: bool = False,
                  as_json: bool = typer.Option(False, "--json"), verbose: bool = False):
    """Ranked plan of issues you can finish."""
    from core.recommend import recommend
    out = recommend(_profile(skills, level, hours, weeks, k, avoid, llm))
    if as_json:
        print(json.dumps(out, indent=2)); return
    if not out["plan"]:
        console.print(f"[yellow]{out.get('note', 'No plan')}[/yellow]", out.get("rejected", "")); return
    t = Table(title=f"Plan: {out['planned_hours']}h of {out['budget_hours']}h budget")
    for c in ["#", "Issue", "Diff", "P(merge)", "Hours", "Reply(d)", "Skills"]:
        t.add_column(c)
    for i in out["plan"]:
        t.add_row(str(i["rank"]), f"{i['repo']}#{i['number']}\n{i['title'][:55]}",
                  f"{i['difficulty']} ({i['diff_conf']:.0%})", f"{i['p_merge']:.0%}", str(i["est_hours"]),
                  str(i["reply_days"]), ", ".join(i["skills"]))
    console.print(t)
    if verbose:
        for i in out["plan"]:
            console.print(f"\n[bold]#{i['rank']} {i['url']}[/bold]\n Rules: {i['rules']}\n CSP: {i['csp_check']}\n {i['explanation']}")
        console.print(f"\nRejected because: {out['rejected']}")
        console.print(f"CSP nodes: FC+MRV={out['csp_nodes_fc_mrv']} vs plain={out['csp_nodes_plain']}")


@app.command("explain")
def explain_cmd(rank: int):
    """Show rules fired and CSP check for a plan item from the last run."""
    plan = json.load(open("data/last_plan.json"))["plan"]
    i = plan[rank - 1]
    console.print(f"[bold]{i['repo']}#{i['number']}[/bold] {i['title']}\n{i['url']}")
    console.print("Rules fired:", *i["rules"] or ["(none)"], sep="\n  - ")
    console.print("CSP check:", {k: "✓" if v else "✗" for k, v in i["csp_check"].items()})
    console.print(i["explanation"])


@app.command("path")
def path_cmd(target: str = typer.Argument(..., help="owner/repo#123"), skills: str = typer.Option(...),
             level: str = "beginner"):
    """A* learning path to a hard target issue."""
    from core.recommend import learning_path
    print(json.dumps(learning_path(_profile(skills, level, 6, 3, 3, "", False), target), indent=2))


@app.command("serve")
def serve_cmd(port: int = 8000):
    import uvicorn; uvicorn.run("api.main:app", port=port)


@app.command("web")
def web_cmd():
    import subprocess; subprocess.run(["streamlit", "run", "app/streamlit_app.py"])


if __name__ == "__main__":
    app()