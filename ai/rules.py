class Rule:
    def __init__(self, name, cond, effect):
        self.name, self.cond, self.effect = name, cond, effect


def _add(x):
    def f(s): s["priority"] += x
    return f


def _flag(name):
    def f(s): s["flags"].append(name)
    return f


def _exclude(s): s["excluded"] = True


def _slow(f): return f["reply_days"] is not None and f["reply_days"] > 60
def _fast(f): return f["reply_days"] is not None and f["reply_days"] < 3


RULES = [
    Rule("R1: open PR already exists -> EXCLUDE", lambda f, s: f["has_open_pr"], _exclude),
    Rule("R2: maintainers reply slower than 60 days -> priority -0.30", lambda f, s: _slow(f), _add(-0.30)),
    Rule("R3: 'help wanted' and unassigned -> priority +0.20", lambda f, s: f["help_wanted"] and f["unassigned"], _add(0.20)),
    Rule("R4: CONTRIBUTING.md present and replies under 3 days -> priority +0.15",
         lambda f, s: f["has_contributing"] and _fast(f), _add(0.15)),
    Rule("R5: older than 1 year with no activity -> flag STALE", lambda f, s: f["age_days"] > 365 and f["comments"] == 0, _flag("stale")),
    Rule("R6: flagged stale -> priority -0.20 (chained from R5)", lambda f, s: "stale" in s["flags"], _add(-0.20)),
]


def run_rules(facts):
    state = {"priority": 0.0, "excluded": False, "flags": [], "trace": []}
    fired, changed = set(), True
    while changed:                        # forward chaining: repeat until nothing new fires
        changed = False
        for r in RULES:
            if r.name not in fired and r.cond(facts, state):
                r.effect(state); fired.add(r.name); state["trace"].append(r.name); changed = True
    return state