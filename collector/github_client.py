import os, json, time, hashlib
import requests
from dotenv import load_dotenv

load_dotenv()
BASE = "https://api.github.com"
CACHE = "data/raw"


class GitHubClient:
    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise SystemExit("Set GITHUB_TOKEN in .env")
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        os.makedirs(CACHE, exist_ok=True)

    def _path(self, url, params):
        raw = url + json.dumps(params or {}, sort_keys=True)
        return os.path.join(CACHE, hashlib.md5(raw.encode()).hexdigest() + ".json")

    def get(self, url, params=None):
        """Returns {"body":..., "next":..., "last":...}. body is None if missing."""
        if not url.startswith("http"):
            url = BASE + url
        path = self._path(url, params)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

        missing = {"body": None, "next": None, "last": None}
        while True:
            r = self.s.get(url, params=params, timeout=30)
            if r.status_code in (403, 429):
                msg = r.text.lower()
                if r.headers.get("X-RateLimit-Remaining") == "0":
                    wait = max(int(r.headers.get("X-RateLimit-Reset", 0)) - time.time(), 1) + 2
                elif "retry-after" in r.headers or "rate limit" in msg or "abuse" in msg:
                    wait = int(r.headers.get("Retry-After", 60))
                else:
                    out = missing          # forbidden for another reason, skip it
                    break
                print(f"Rate limited. Sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
            if r.status_code in (404, 410, 451):
                out = missing
                break
            r.raise_for_status()
            out = {"body": r.json(),
                   "next": r.links.get("next", {}).get("url"),
                   "last": r.links.get("last", {}).get("url")}
            break

        with open(path, "w") as f:
            json.dump(out, f)
        return out

    def paginate(self, url, params=None, max_pages=3):
        params = dict(params or {}, per_page=100)
        items, pages = [], 0
        while url and pages < max_pages:
            res = self.get(url, params if pages == 0 else None)
            if not isinstance(res["body"], list):
                break
            items += res["body"]
            url = res["next"]
            pages += 1
        return items