"""Why is `collect` instant / why does a repo have no issues?    python -m tools.diagnose_collection [--clear-missing]"""
import hashlib, json, os, sqlite3, sys, time
import requests
from dotenv import load_dotenv

load_dotenv()
CACHE = "data/raw"


def cpath(url, params):
    return os.path.join(CACHE, hashlib.md5((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest() + ".json")


def cache_state(url, params):
    f = cpath(url, params)
    if not os.path.exists(f):
        return "NOT CACHED (collect never reached it, or it crashed: look for FAILED in the output)"
    b = json.load(open(f)).get("body")
    if b is None:
        return "CACHED AS MISSING (an earlier 403/404 was stored; run with --clear-missing)"
    return f"downloaded ({len(b)} items on page 1)" if isinstance(b, list) else "downloaded"


def main():
    clear = "--clear-missing" in sys.argv
    tok = os.getenv("GITHUB_TOKEN", "")
    try:
        r = requests.get("https://api.github.com/rate_limit", headers={"Authorization": f"Bearer {tok}"}, timeout=15)
        if r.status_code == 401:
            print("TOKEN: INVALID OR EXPIRED -> create a new token and put it in .env")
        else:
            core = r.json()["resources"]["core"]
            print(f"TOKEN: ok, {core['remaining']}/{core['limit']} requests left this hour (resets {time.strftime('%H:%M', time.localtime(core['reset']))})")
    except Exception as e:
        print("TOKEN: could not reach GitHub:", type(e).__name__)
    repos = [l.strip() for l in open("repos.txt") if l.strip() and not l.startswith("#")]
    c = sqlite3.connect("data/issuescout.db")
    stat = {r[0]: (r[1] or 0, r[2] or 0) for r in c.execute("select repo, sum(pr_merged=1), sum(state='open') from issues group by repo")}
    known = {r[0] for r in c.execute("select full_name from repos")}
    q = {"state": "closed", "sort": "created", "direction": "desc", "per_page": 100}
    empty, no_repo = [], []
    for rp in repos:
        if rp not in known:
            no_repo.append(rp); continue
        lab, op = stat.get(rp, (0, 0))
        if lab == 0 and op == 0:
            empty.append((rp, cache_state(f"https://api.github.com/repos/{rp}/issues", q)))
    tot_lab = sum(v[0] for v in stat.values())
    print(f"\nrepos in repos.txt: {len(repos)} | with issues in DB: {len(stat)} | labelled issues: {tot_lab} | open: {sum(v[1] for v in stat.values())}")
    print(f"repos whose repo-info call failed (name wrong or blocked): {no_repo}")
    print(f"repos with ZERO issues: {len(empty)}")
    for rp, why in empty:
        print(f"  {rp:34s} {why}")
    few = sorted([(v[0], k) for k, v in stat.items() if 0 < v[0] < 5])
    print(f"repos with under 5 labelled issues (little training value): {len(few)}")
    miss = [f for f in os.listdir(CACHE) if json.load(open(os.path.join(CACHE, f))).get("body") is None]
    print(f"\ncache files stored as 'missing': {len(miss)}")
    if clear:
        for f in miss:
            os.remove(os.path.join(CACHE, f))
        print("removed them: run `python cli.py collect ...` again to refetch.")
    elif miss:
        print("(404s for CONTRIBUTING.md paths are normal; but a stored 403 on issues blocks a repo forever. Use --clear-missing to retry them.)")


if __name__ == "__main__":
    main()