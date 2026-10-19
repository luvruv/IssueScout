"""Summarise the classmate study.  data/user_study.csv columns: participant,doable_1to5,explanation_clear_1to5,would_try_1to5,comments"""
import pandas as pd
d = pd.read_csv("data/user_study.csv")
for c in ["doable_1to5", "explanation_clear_1to5", "would_try_1to5"]:
    print(f"{c:26s} mean {d[c].mean():.2f}  sd {d[c].std():.2f}  n={d[c].notna().sum()}")
