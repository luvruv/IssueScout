import heapq, itertools


def astar(start, goal, steps):
    """
    start: skills the user has. goal: skills to gain.
    steps: dicts {id, title, teaches:frozenset, needs:frozenset, hours}.
    g = hours spent. h = (#missing goal skills) * (min hours per skill any step offers).
    Admissible: a step costing H teaches at most |teaches| skills, so every skill costs >= min(H/|teaches|).
    """
    start, goal = frozenset(start), frozenset(goal)
    per_skill = min(s["hours"] / max(len(s["teaches"]), 1) for s in steps)
    h = lambda state: len(goal - state) * per_skill
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