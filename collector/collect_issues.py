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
    repos = [l.strip() for l in open(repos_file) if l.strip() and not l.startswith("#")]
    failed = []
    for n, full in enumerate(repos, 1):
        try:
            print(f"[{n}/{len(repos)}] (API requests left this hour: {gh.remaining})", end=" ", flush=True)
            collect_repo(c, full, closed_pages, open_pages, pr_pages)
        except Exception as e:
            print("FAILED", full, type(e).__name__, str(e)[:150], flush=True)
            failed.append(full)
    print(f"\nDone. {len(repos) - len(failed)} repos ok, {len(failed)} failed: {failed}")