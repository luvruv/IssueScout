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