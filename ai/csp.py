def solve(cands, k, budget, max_per_repo=2, fc=True, mrv=True, max_nodes=200000):
    """
    cands: list of dicts with keys id, repo, hours, idx (idx = rank by score, best first).
    Variables: k plan slots. Domain: candidates.
    Constraints: all different, total hours <= budget, <= max_per_repo per repo.
    Symmetry breaking: slot order follows idx order (removes duplicate permutations).
    Returns (solution list or None, nodes explored).
    """
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
        var = min(free, key=lambda v: len(domains[v])) if mrv else free[0]     # MRV
        for c in domains[var]:
            if stats["nodes"] >= max_nodes:
                return None
            if not ok(assign, var, c):
                continue
            stats["nodes"] += 1
            assign[var] = c
            nd, dead = domains, False
            if fc:                                                               # forward checking
                nd = dict(domains)
                for v in free:
                    if v == var:
                        continue
                    nd[v] = [x for x in domains[v] if ok(assign, v, x)]
                    if not nd[v]:
                        dead = True
                        break
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