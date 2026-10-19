"""A* vs uniform-cost search (same optimal cost, fewer expansions) + empirical admissibility check."""
import random
from ai.astar import astar, verify_admissible


def instance(seed, n_skills=7, n_steps=22):
    rnd = random.Random(seed); sk = [f"s{i}" for i in range(n_skills)]; steps = []
    for i in range(n_steps):
        teaches = frozenset(rnd.sample(sk, rnd.choice([1, 1, 2])))
        needs = frozenset(rnd.sample([s for s in sk if s not in teaches], rnd.choice([0, 0, 1])))
        steps.append({"id": f"step{i}", "title": f"step{i}", "teaches": teaches, "needs": needs, "hours": float(rnd.randint(2, 12))})
    return set(), set(sk[:6]), steps


def main():
    rows, ok_all = [], True
    for seed in range(8):
        s, g, steps = instance(seed)
        pa, ca, ea = astar(s, g, steps); pu, cu, eu = astar(s, g, steps, use_heuristic=False)
        if ca is None:
            continue
        adm, worst, n = verify_admissible(s, g, steps, limit=120)
        ok_all &= adm and abs(ca - cu) < 1e-9
        rows.append(f"| {seed} | {ca:.0f} | {cu:.0f} | {ea} | {eu} | {'yes' if adm else 'NO'} (max h/true = {worst:.2f}, {n} states) |")
    txt = ("| instance | A* cost | UCS cost | A* expanded | UCS expanded | h admissible? |\n|---|---|---|---|---|---|\n" + "\n".join(rows)
           + f"\n\nA* always matched the optimal UCS cost: {ok_all}\n")
    import os; os.makedirs("report", exist_ok=True)
    open("report/astar_benchmark.md", "w").write(txt); print(txt)
