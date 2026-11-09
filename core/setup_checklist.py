"""Automated Developer Setup Checklist.
Extracts concrete terminal setup commands (fork/clone, install dependencies, run test suite, linters)
from a repository's contributor guides (CONTRIBUTING.md, README.md, Makefile).
"""
import re
from typing import Dict, Any, List
from llm.rag import fetch_docs


COMMAND_PATTERNS = {
    "clone": [
        re.compile(r"(git clone\s+https://github\.com/[^\s`\"']+)", re.IGNORECASE),
        re.compile(r"(git clone\s+git@github\.com:[^\s`\"']+)", re.IGNORECASE)
    ],
    "install": [
        re.compile(r"((?:python3? -m venv|virtualenv)\s+\S+)", re.IGNORECASE),
        re.compile(r"((?:pip|pip3)\s+install\s+-(?:e|r)\s+[^\n`\"#]+)", re.IGNORECASE),
        re.compile(r"((?:poetry|pdm|flit)\s+install[^\n`\"#]*)", re.IGNORECASE),
        re.compile(r"((?:npm|yarn|pnpm|bun)\s+install[^\n`\"#]*)", re.IGNORECASE),
        re.compile(r"((?:cargo\s+build|go\s+mod\s+download)[^\n`\"#]*)", re.IGNORECASE)
    ],
    "test": [
        re.compile(r"((?:pytest|tox|nox|python3?\s+-m\s+unittest)[^\n`\"#]*)", re.IGNORECASE),
        re.compile(r"((?:npm|yarn|pnpm|bun)\s+test[^\n`\"#]*)", re.IGNORECASE),
        re.compile(r"((?:cargo\s+test|go\s+test\s+[^\n`\"#]*))", re.IGNORECASE),
        re.compile(r"(make\s+(?:test|check)[^\n`\"#]*)", re.IGNORECASE)
    ],
    "lint": [
        re.compile(r"(pre-commit\s+install[^\n`\"#]*)", re.IGNORECASE),
        re.compile(r"((?:ruff|flake8|black|mypy|pylint|eslint)\s+[^\n`\"#]*)", re.IGNORECASE),
        re.compile(r"((?:npm|yarn|pnpm)\s+run\s+lint[^\n`\"#]*)", re.IGNORECASE),
        re.compile(r"(make\s+lint[^\n`\"#]*)", re.IGNORECASE)
    ]
}


def extract_setup_checklist(repo: str) -> Dict[str, Any]:
    """Inspects cached or fetched contributor docs for `repo` and extracts a clean,
    executable checklist of setup commands for incoming contributors.
    """
    docs = fetch_docs(repo)
    if not docs:
        return {
            "repo": repo,
            "has_docs": False,
            "clone_cmd": f"git clone https://github.com/{repo}.git\ncd {repo.split('/')[-1]}",
            "install_cmds": ["# Look for setup instructions in repository README"],
            "test_cmds": ["# Run standard test runner (e.g. pytest or npm test)"],
            "lint_cmds": ["pre-commit install"],
            "sources": []
        }

    all_text = "\n".join(d.get("text", "") for d in docs)
    sources = [d.get("path") for d in docs if d.get("path")]

    extracted = {
        "clone": [],
        "install": [],
        "test": [],
        "lint": []
    }

    for category, patterns in COMMAND_PATTERNS.items():
        for pat in patterns:
            for match in pat.finditer(all_text):
                cmd = match.group(1).strip()
                # Clean stray backticks or trailing punctuation
                cmd = cmd.strip("`'\";$ ").replace("\\\n", " ")
                if len(cmd) < 120 and cmd not in extracted[category]:
                    extracted[category].append(cmd)

    # Defaults if not explicitly matched in text
    repo_name = repo.split('/')[-1]
    clone_cmd = extracted["clone"][0] if extracted["clone"] else f"git clone https://github.com/{repo}.git\ncd {repo_name}"
    
    install_cmds = extracted["install"][:3] or [
        "python -m venv .venv",
        "# Windows: .venv\\Scripts\\activate  |  Linux/Mac: source .venv/bin/activate",
        "pip install -e .[dev] || pip install -r requirements.txt"
    ]
    
    test_cmds = extracted["test"][:3] or ["pytest"]
    lint_cmds = extracted["lint"][:2] or ["pre-commit install"]

    return {
        "repo": repo,
        "has_docs": True,
        "clone_cmd": clone_cmd,
        "install_cmds": install_cmds,
        "test_cmds": test_cmds,
        "lint_cmds": lint_cmds,
        "sources": sources
    }

