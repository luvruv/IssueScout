import re
import datetime as dt
from collector.github_client import GitHubClient
from collector.db import conn, upsert

gh = GitHubClient()
FMT = "%Y-%m-%dT%H:%M:%SZ"
LINK = re.compile(r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*(?:[\w.\-]+/[\w.\-]+)?#(\d+)")


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


def collect_repo(c, full):
    q = lambda st: {"state": st, "sort": "created", "direction": "desc"}
    closed = [i for i in gh.paginate(f"/repos/{full}/issues", q("closed"), 3) if "pull_request" not in i]
    opened = [i for i in gh.paginate(f"/repos/{full}/issues", q("open"), 2) if "pull_request" not in i]
    closed_prs = gh.paginate(f"/repos/{full}/pulls", q("closed"), 3)
    open_prs = gh.paginate(f"/repos/{full}/pulls", {"state": "open"}, 2)

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
        })
        upsert(c, "issues", row)
        n_closed += 1

    for i in opened:
        row = base_row(full, i)
        row["has_open_pr"] = int(i["number"] in open_refs)
        upsert(c, "issues", row)

    c.commit()
    print(f"{full}: {n_closed} labelled closed, {len(opened)} open")


def main(repos_file="repos.txt"):
    c = conn()
    for line in open(repos_file):
        full = line.strip()
        if not full or full.startswith("#"):
            continue
        try:
            collect_repo(c, full)
        except Exception as e:
            print("FAILED", full, e)