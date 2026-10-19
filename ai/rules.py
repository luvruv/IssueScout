# Responsible AI rule definitions (PEAS table)
# This file is imported by docs and benchmarks.

PEAS = {
    "Performance": ["Accurate difficulty prediction", "Robust merge prediction"],
    "Environment": ["Open‑source repositories", "GitHub API"],
    "Actuators": ["Recommendation CLI", "GitHub comment bot"],
    "Sensors": ["Issue text, tags, repository metadata"]
}

# Example rule list used by the benchmark reports
RULES = [
    "No data leakage: only features available at issue creation are used.",
    "Repo‑prior must be learned on training data only (leave‑one‑out).",
    "Use calibrated probabilities for downstream cost calculations.",
    "When a model predicts Hard as Easy the cost is highest (3).",
    "Budget bound in CSP reduces expansion dramatically."
]

def get_rules():
    """Return the rule list for reporting purposes."""
    return RULES


def is_noise(title: str, labels: str | None = None) -> bool:
    """Simple heuristic to identify noisy issues.
    Args:
        title: Issue title.
        labels: Comma‑separated label string (optional).
    Returns:
        True if the issue is considered noise, False otherwise.
    """
    t = title.lower()
    if "dashboard" in t or t.startswith("tracking issue"):
        return True
    if labels:
        lbl = labels.lower()
        if "epic" in lbl or "kind/epic" in lbl:
            return True
    return False


def run_rules(facts: dict) -> dict:
    """Evaluate a set of simple rules on the provided fact dictionary.
    Returns a dict with keys: excluded (bool), flags (list), priority (float), trace (list).
    """
    result = {
        "excluded": False,
        "flags": [],
        "priority": 0.0,
        "trace": []
    }
    # Exclusion rules
    if facts.get("has_open_pr"):
        result["excluded"] = True
        result["trace"].append("has_open_pr")
    title = facts.get("title", "").lower()
    if title.startswith("tracking issue"):
        result["excluded"] = True
        result["trace"].append("tracking_title")
    if "wontfix" in facts.get("labels", "").lower():
        result["excluded"] = True
        result["trace"].append("wontfix_label")
    # Flags and priority adjustments
    if facts.get("age_days", 0) > 365 and facts.get("comments", 0) == 0:
        result["flags"].append("stale")
        result["priority"] -= 0.1
        result["trace"].extend(["stale_rule", "priority_adjust"])
    if facts.get("gfi") and facts.get("help_wanted"):
        result["priority"] += 0.5
        result["trace"].append("gfi_help_wanted")
    if facts.get("conf", 1.0) < 0.5:
        result["flags"].append("uncertain")
        result["trace"].append("uncertain_confidence")
    return result