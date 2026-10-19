"""Interactive manual labelling. The automatic label is HIDDEN so it cannot bias you."""
import os, webbrowser
import pandas as pd
from collector.db import conn

PATH = "data/processed/manual_label_sample.csv"
KEYS = {"e": "Easy", "m": "Medium", "h": "Hard"}


def main():
    if not os.path.exists(PATH):
        raise SystemExit("Run `python cli.py features` first.")
    m = pd.read_csv(PATH)
    m["my_difficulty"] = m["my_difficulty"].fillna("").astype(str)
    todo = m.index[~m["my_difficulty"].isin(KEYS.values())].tolist()
    print(f"{len(m) - len(todo)} done, {len(todo)} to go.  Keys: e=Easy m=Medium h=Hard  o=open in browser  s=skip  q=quit")
    print("Judge how hard it would be for a beginner who knows the language, from the issue ALONE.\n")
    db = conn()
    for n, i in enumerate(todo, 1):
        r = m.loc[i]
        row = db.execute("SELECT body, labels FROM issues WHERE id=?", (r["id"],)).fetchone() or ("", "")
        print("=" * 78)
        print(f"[{n}/{len(todo)}] {r['id']}  |  {r['title']}\nlabels: {row[1]}\n{(row[0] or '')[:900]}\n{r['url']}")
        while True:
            k = input("e/m/h/o/s/q > ").strip().lower()
            if k == "o":
                webbrowser.open(r["url"]); continue
            break
        if k == "q":
            break
        if k in KEYS:
            m.loc[i, "my_difficulty"] = KEYS[k]
            m.to_csv(PATH, index=False)
    m.to_csv(PATH, index=False)
    print("saved. Run `python cli.py evaluate` to get agreement and kappa.")
