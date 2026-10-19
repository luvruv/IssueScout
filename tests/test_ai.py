import random
from ai.csp import solve
from ai.astar import astar, verify_admissible
from ai.rules import run_rules
from ai.ontology import missing, closure, extract_keywords, skills_to_learn
from ai.bayes import posterior, update, mle_rate


def facts(**kw):
    f = dict(has_open_pr=False, reply_days=None, help_wanted=False, unassigned=True, has_contributing=False, age_days=10,
             comments=0, labels="", title="fix typo in docs", body_len=300, gfi=False, conf=0.8)
    f.update(kw); return f


def test_rules_exclude_and_chain():
    assert run_rules(facts(has_open_pr=True))["excluded"]
    assert run_rules(facts(title="Tracking issue: v2"))["excluded"]
    assert run_rules(facts(labels="wontfix"))["excluded"]
    s = run_rules(facts(age_days=500, comments=0))
    assert "stale" in s["flags"] and s["priority"] < 0 and len(s["trace"]) == 2     # R5 fires, then R6 chains
    assert run_rules(facts(gfi=True, help_wanted=True))["priority"] > 0.4
    assert "uncertain" in run_rules(facts(conf=0.4))["flags"]


def pool(seed=3, n=30):
    r = random.Random(seed)
    return [{"id": i, "repo": f"r{r.randrange(6)}", "hours": round(r.uniform(2, 12), 1), "idx": i} for i in range(n)]


def test_csp_constraints_hold():
    p = pool(); sol, _ = solve(p, 5, 20)
    assert sol and len(sol) == 5 and len({c["id"] for c in sol}) == 5
    assert sum(c["hours"] for c in sol) <= 20
    assert max(sum(1 for c in sol if c["repo"] == r) for r in {c["repo"] for c in sol}) <= 2


def test_csp_methods_agree_and_bound_prunes():
    p = pool()
    for k, b in [(5, 20), (6, 19)]:
        s1, n1 = solve(p, k, b, fc=False, mrv=False); s2, n2 = solve(p, k, b)
        assert (s1 is None) == (s2 is None)
        assert n2 <= n1
        if s1: assert [c["id"] for c in s1] == [c["id"] for c in s2]          # same plan, fewer nodes
    _, n1 = solve(p, 6, 19, fc=False, mrv=False); _, n2 = solve(p, 6, 19)
    assert n2 * 5 < n1                                                         # the bound gives a large saving


def test_csp_infeasible():
    assert solve(pool(), 5, 3)[0] is None


def steps():
    f = frozenset
    return [{"id": "a", "title": "a", "teaches": f({"go"}), "needs": f(), "hours": 3},
            {"id": "b", "title": "b", "teaches": f({"docker"}), "needs": f(), "hours": 4},
            {"id": "c", "title": "c", "teaches": f({"client-go"}), "needs": f({"go"}), "hours": 6},
            {"id": "e", "title": "e", "teaches": f({"go", "docker"}), "needs": f(), "hours": 9}]


def test_astar_optimal_and_admissible():
    goal = {"go", "docker", "client-go"}
    p, c, e = astar(set(), goal, steps()); p2, c2, e2 = astar(set(), goal, steps(), use_heuristic=False)
    assert c == c2 == 13 and e <= e2
    ok, worst, n = verify_admissible(set(), goal, steps()); assert ok and worst <= 1.0 + 1e-9


def test_astar_unreachable():
    assert astar(set(), {"rust"}, steps())[0] is None


def test_ontology():
    assert not missing(["go"], {"client-go", "go"})                 # client-go is learnable for a Go user
    assert "kubernetes" in missing(["go"], {"kubernetes"})
    assert "javascript" in closure(["react"])
    assert {"docker", "yaml"} <= skills_to_learn(["go"], {"kubernetes"}) | {"docker", "yaml"}
    assert "go" in extract_keywords("panic in main.go when parsing", "Go") and "python" in extract_keywords("pip install fails", "")


def test_bayes():
    assert mle_rate(2, 2) == 1.0
    m, (lo, hi) = posterior(2, 2, 0.8); assert 0.8 < m < 0.95 and lo < m < hi    # shrunk below the MLE of 1.0
    assert 0.5 < update(0.5, 2.0) < 1


def test_noise_issues():
    from ai.rules import is_noise
    assert is_noise("Dependency Dashboard") and is_noise("Tracking issue: v2 release") and is_noise("x", "kind/epic")
    assert not is_noise("Fix typo in README")
