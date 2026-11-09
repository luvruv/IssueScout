"""Issue Claim & Activity Detector.
Analyzes issue metadata, assignees, and comment discussion to determine if someone
is already actively working on the issue, preventing wasted contributor effort.
"""
import re
from typing import Dict, Any, List, Optional
import sqlite3
import os


CLAIM_INTENT_PATTERNS = [
    re.compile(r"\b(?:can i|may i|i would like to|i'd like to|i will|let me|can you assign)\s+(?:take|work|fix|handle|do)\b", re.IGNORECASE),
    re.compile(r"\b(?:working on (?:a|this|the) (?:fix|pr|patch)|taking this up|started working)\b", re.IGNORECASE),
    re.compile(r"\b(?:please assign (?:to me|this to me)|assign it to me)\b", re.IGNORECASE),
    re.compile(r"\b(?:opened (?:a|the) (?:pr|pull request)|submitted (?:a|the) (?:pr|patch))\b", re.IGNORECASE),
    re.compile(r"\b(?:fixes|resolves|closes)\s+#\d+\b", re.IGNORECASE)
]

MAINTAINER_ASSIGNED_PATTERNS = [
    re.compile(r"\b(?:assigned to @?\w+|all yours|go ahead|feel free to (?:open|submit|work))\b", re.IGNORECASE),
    re.compile(r"\b(?:assigned!|assigned\.)\b", re.IGNORECASE)
]


def detect_issue_claim_status(repo: str, issue_num: int, issue_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyzes an issue's claim status.
    Returns:
    - status: 'AVAILABLE' | 'IN_DISCUSSION' | 'LIKELY_CLAIMED' | 'ASSIGNED' | 'CLOSED'
    - badge: Visual emoji badge for UI
    - confidence: float 0.0 - 1.0
    - reason: Human-readable explanation
    - assignees: List of usernames
    """
    if not issue_data:
        try:
            from collector.github_client import GitHubClient
            res = GitHubClient().get(f"/repos/{repo}/issues/{issue_num}")
            if isinstance(res, dict) and isinstance(res.get("body"), dict):
                issue_data = res.get("body")
        except Exception:
            issue_data = None

    # Offline / Local fallback: Query issuescout.db
    if not issue_data and os.path.exists("data/issuescout.db"):
        try:
            conn = sqlite3.connect("data/issuescout.db")
            row = conn.execute("select title, body, state, comments, pr_merged from issues where repo=? and number=?", (repo, issue_num)).fetchone()
            if row:
                issue_data = {
                    "number": issue_num,
                    "title": row[0],
                    "body": row[1],
                    "state": row[2],
                    "comments": row[3] or 0,
                    "assignees": [],
                    "pull_request": None if row[4] != 1 else {}
                }
        except Exception:
            pass

    if not issue_data or not isinstance(issue_data, dict):
        return {
            "status": "AVAILABLE",
            "badge": "✨ Available",
            "confidence": 0.70,
            "reason": "No active claims recorded for this issue.",
            "is_available": True,
            "assignees": [],
            "signals": ["Local database shows issue unassigned"]
        }

    # 1. Check if closed
    if issue_data.get("state") == "closed":
        return {
            "status": "CLOSED",
            "badge": "🟣 Closed",
            "confidence": 1.0,
            "reason": "Issue is closed/resolved.",
            "is_available": False,
            "assignees": [a.get("login") for a in issue_data.get("assignees", []) if isinstance(a, dict)],
            "signals": ["Issue state is closed"]
        }

    # 2. Check official assignees
    assignees = [a.get("login") for a in issue_data.get("assignees", []) if isinstance(a, dict)]
    if assignees:
        return {
            "status": "ASSIGNED",
            "badge": f"🔒 Assigned (@{assignees[0]})",
            "confidence": 0.95,
            "reason": f"Officially assigned to {', '.join('@' + u for u in assignees)}.",
            "is_available": False,
            "assignees": assignees,
            "signals": [f"Assigned to {assignees}"]
        }

    # 3. Check pull_request key (if it's a PR, not an issue)
    if issue_data.get("pull_request"):
        return {
            "status": "IS_PR",
            "badge": "🔀 Pull Request",
            "confidence": 1.0,
            "reason": "This reference is already a Pull Request.",
            "is_available": False,
            "assignees": assignees,
            "signals": ["Object is a Pull Request"]
        }

    # 4. Check comments for claim discussions
    num_comments = issue_data.get("comments", 0)
    if num_comments == 0:
        return {
            "status": "AVAILABLE",
            "badge": "✨ Available (0 comments)",
            "confidence": 0.92,
            "reason": "No comments or assignees. Completely open for newcomers.",
            "is_available": True,
            "assignees": [],
            "signals": ["Zero comments", "Zero assignees"]
        }

    comments = []
    try:
        from collector.github_client import GitHubClient
        c_res = GitHubClient().get(f"/repos/{repo}/issues/{issue_num}/comments", params={"per_page": 20})
        if isinstance(c_res, dict) and isinstance(c_res.get("body"), list):
            comments = c_res.get("body")
    except Exception:
        comments = []

    signals = []
    claim_intents = []
    maintainer_assigns = []

    if isinstance(comments, list):
        for c in comments:
            body = c.get("body", "")
            user = (c.get("user") or {}).get("login", "unknown")
            
            for pat in CLAIM_INTENT_PATTERNS:
                if pat.search(body):
                    claim_intents.append((user, pat.pattern))
                    signals.append(f"@{user} expressed intent to work on fix")
                    break
                    
            for pat in MAINTAINER_ASSIGNED_PATTERNS:
                if pat.search(body):
                    maintainer_assigns.append((user, pat.pattern))
                    signals.append(f"Maintainer approval signal found in thread: '{pat.pattern}'")
                    break

    if maintainer_assigns and claim_intents:
        latest_claimer = claim_intents[-1][0]
        return {
            "status": "LIKELY_CLAIMED",
            "badge": f"⚠️ Likely Claimed (@{latest_claimer})",
            "confidence": 0.85,
            "reason": f"A contributor (@{latest_claimer}) offered to solve this and maintainers acknowledged.",
            "is_available": False,
            "assignees": [latest_claimer],
            "signals": signals
        }

    if claim_intents:
        latest_claimer = claim_intents[-1][0]
        return {
            "status": "IN_DISCUSSION",
            "badge": "💬 In Discussion",
            "confidence": 0.65,
            "reason": f"Recent contributor interest from @{latest_claimer}. Check comments before starting.",
            "is_available": True,
            "assignees": [],
            "signals": signals
        }

    return {
        "status": "AVAILABLE",
        "badge": "✨ Open & Available",
        "confidence": 0.85,
        "reason": f"Discussion exists ({num_comments} comments), but no active claim or assignment detected.",
        "is_available": True,
        "assignees": [],
        "signals": [f"{num_comments} discussion comments, no active claims"]
    }
