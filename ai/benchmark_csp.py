"""Node-count comparison: plain backtracking vs +FC vs +MRV vs +FC+MRV vs +bound.  Writes report/csp_benchmark.md"""
import random, time


def pools():
    out = []
    for seed, n, nrepos in [(3, 30, 6), (7, 40, 8), (11, 25, 5)]:
        rnd = random.Random(seed)
        out.append((f"synthetic n={n}", [{"id": i, "repo": f"r{rnd.randrange(nrepos)}", "hours": round(rnd.uniform(2, 12), 1), "idx": i} for i in range(n)]))
    try:                                           # a real pool from your own data, if the models exist
        from core.recommend import score_open
        df = score_open().sort_values("p_merge", ascending=False).head(40).reset_index(drop=True)
        out.append(("your data (top 40)", [{"id": r.id, "repo": r.repo, "hours": float(r.est_hours), "idx": i} for i, r in df.iterrows()]))
    except Exception as e:
        print("real pool skipped:", str(e)[:80])
    return out


def main():
    from ai.csp import solve
    cfgs = [("plain", dict(fc=False, mrv=False)), ("FC", dict(fc=True, mrv=False, bound=False)), ("MRV", dict(fc=False, mrv=True)),
            ("FC+MRV", dict(fc=True, mrv=True, bound=False)), ("FC+MRV+budget bound", dict(fc=True, mrv=True, bound=True))]
    lines = ["| pool | k | budget h | " + " | ".join(n for n, _ in cfgs) + " |", "|" + "---|" * (3 + len(cfgs))]
    for name, pool in pools():
        for k, budget in [(4, 16), (5, 20), (6, 22), (6, 19), (7, 24)]:
            cells = []
            for _, kw in cfgs:
                t = time.time(); sol, n = solve(pool, k, budget, max_nodes=300000, **kw)
                cells.append(f"{n}{'' if sol else ' (no plan)'} / {time.time()-t:.2f}s")
            lines.append(f"| {name} | {k} | {budget} | " + " | ".join(cells) + " |")
    txt = "Nodes explored / seconds (lower is better; all methods find a plan whenever one exists).\n\n" + "\n".join(lines) + "\n"
    import os; os.makedirs("report", exist_ok=True)
    open("report/csp_benchmark.md", "w").write(txt); print(txt)
