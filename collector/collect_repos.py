import re
from collector.github_client import GitHubClient
from collector.db import conn, upsert
from collector.merge_repos import load_repos

gh = GitHubClient()


def contributors_count(full):
    res = gh.get(f"/repos/{full}/contributors", {"per_page": 1, "anon": "false"})
    if res["last"]:
        m = re.search(r"[?&]page=(\d+)", res["last"])
        return int(m.group(1)) if m else 0
    return len(res["body"] or [])


def main(repos_file="repos.txt"):
    c = conn()
    repos = load_repos(repos_file)
    for full in repos:
        r = gh.get(f"/repos/{full}")["body"]
        if not r:
            print("missing:", full)
            continue
        has_contrib = any(
            gh.get(f"/repos/{full}/contents/{p}")["body"]
            for p in ["CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md"]
        )
        upsert(c, "repos", {
            "full_name": full, "stars": r["stargazers_count"], "forks": r["forks_count"],
            "language": r["language"], "created_at": r["created_at"], "pushed_at": r["pushed_at"],
            "open_issues": r["open_issues_count"], "contributors": contributors_count(full),
            "has_contributing": int(has_contrib), "archived": int(r["archived"]),
            "description": r["description"],
        })
        c.commit()
        print("repo ok:", full)
