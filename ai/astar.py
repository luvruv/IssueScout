"""A* over 'skills gained' states.  State = set of skills known; action = finish a stepping-stone issue.
g = hours spent.   h = (#missing goal skills) x (cheapest hours-per-goal-skill any step offers).

ADMISSIBLE: a step costing H teaches at most |teaches ∩ goal| goal skills, so each missing goal skill costs at
least min_s H_s/|teaches_s ∩ goal|. h therefore never overestimates the true remaining cost. (h is also consistent.)
"""
import heapq, itertools


def _per_skill(goal, steps):
    vals = [s["hours"] / len(s["teaches"] & goal) for s in steps if s["teaches"] & goal]
    return min(vals) if vals else None


def astar(start, goal, steps, use_heuristic=True):
    """Returns (path, total_hours, nodes_expanded). use_heuristic=False gives uniform-cost search for comparison."""
    start, goal = frozenset(start), frozenset(goal)
    per = _per_skill(goal, steps)
    if per is None:
        return None, None, 0
    h = (lambda st: len(goal - st) * per) if use_heuristic else (lambda st: 0.0)
    tie = itertools.count()
    pq = [(h(start), next(tie), 0.0, start, [])]
    best, expanded = {start: 0.0}, 0
    while pq:
        f, _, g, state, path = heapq.heappop(pq)
        if goal <= state:
            return path, g, expanded
        if g > best.get(state, 1e18):
            continue
        expanded += 1
        for s in steps:
            if not s["needs"] <= state:
                continue
            new = state | s["teaches"]
            if new == state:
                continue
            g2 = g + s["hours"]
            if g2 < best.get(new, 1e18):
                best[new] = g2
                heapq.heappush(pq, (g2 + h(new), next(tie), g2, new, path + [s]))
    return None, None, expanded


def verify_admissible(start, goal, steps, limit=400):
    """Empirical proof on small instances: for every reachable state, h(state) <= true optimal cost-to-go."""
    goal = frozenset(goal); per = _per_skill(goal, steps)
    seen, stack, worst = set(), [frozenset(start)], 0.0
    while stack and len(seen) < limit:
        st = stack.pop()
        if st in seen:
            continue
        seen.add(st)
        for s in steps:
            if s["needs"] <= st and not s["teaches"] <= st:
                stack.append(st | s["teaches"])
    bad = []
    for st in seen:
        _, true_cost, _ = astar(st, goal, steps, use_heuristic=False)
        if true_cost is None:
            continue
        hv = len(goal - st) * per
        worst = max(worst, hv / true_cost if true_cost else 0)
        if hv > true_cost + 1e-9:
            bad.append((sorted(st), hv, true_cost))
    return (not bad), worst, len(seen)