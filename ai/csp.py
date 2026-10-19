"""Issue-selection CSP: backtracking + forward checking + MRV + a remaining-budget bound.

Variables  : the k plan slots.            Domain: pre-filtered candidate issues.
Constraints: all different, sum(hours) <= budget, at most `max_per_repo` per repo.
Symmetry   : slot order follows candidate order (idx), so permutations of one plan are not re-explored.
Candidates are sorted best-score-first, so the first solution found is the highest-ranked feasible plan.

Forward checking alone cannot see the SHARED budget (it checks each slot separately), so it explores the same
nodes as plain backtracking. `bound=True` adds the missing global check: the cheapest possible completion must fit.
"""


def solve(cands, k, budget, max_per_repo=2, fc=True, mrv=True, bound=True, max_nodes=200000):
    """cands: dicts with id, repo, hours, idx.  Returns (solution list or None, nodes explored)."""
    stats = {"nodes": 0}

    def ok(assign, var, c):
        total, same = c["hours"], 0
        for s, a in assign.items():
            if a["idx"] == c["idx"]:
                return False
            if (s < var) != (a["idx"] < c["idx"]):
                return False
            total += a["hours"]
            same += a["repo"] == c["repo"]
        return total <= budget and same < max_per_repo

    def bt(assign, domains):
        if len(assign) == k:
            return dict(assign)
        free = [v for v in range(k) if v not in assign]
        var = min(free, key=lambda v: len(domains[v])) if mrv else free[0]          # MRV
        for c in domains[var]:
            if stats["nodes"] >= max_nodes:
                return None
            if not ok(assign, var, c):
                continue
            stats["nodes"] += 1
            assign[var] = c
            nd, dead = domains, False
            if fc:                                                                  # forward checking
                nd = dict(domains)
                rest = [v for v in free if v != var]
                for v in rest:
                    nd[v] = [x for x in domains[v] if ok(assign, v, x)]
                    if not nd[v]:
                        dead = True
                        break
                if not dead and bound and rest:                                     # global budget bound
                    used = sum(a["hours"] for a in assign.values())
                    cheapest = sorted({x["idx"]: x["hours"] for v in rest for x in nd[v]}.items(), key=lambda t: t[1])
                    if len(cheapest) < len(rest) or used + sum(h for _, h in cheapest[:len(rest)]) > budget:
                        dead = True
            if not dead:
                res = bt(assign, nd)
                if res:
                    return res
            del assign[var]
        return None

    sol = bt({}, {i: list(cands) for i in range(k)})
    if not sol:
        return None, stats["nodes"]
    return [sol[v] for v in sorted(sol, key=lambda v: sol[v]["idx"])], stats["nodes"]