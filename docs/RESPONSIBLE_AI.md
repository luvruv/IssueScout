# Responsible AI and PEAS

## PEAS (utility-based agent)
| | |
|---|---|
| Performance measure | expected merge probability of the recommended set, within the user's hour budget, penalised by expected difficulty |
| Environment | GitHub (partially observable, dynamic: issues get claimed), the user's profile |
| Actuators | ranked plan, exclusions with reasons, learning path, first-PR kit (drafts only) |
| Sensors | GitHub API data, user profile, user feedback and outcomes |

## Practices in the code
- Never posts to GitHub; drafts are labelled as drafts.
- Always shows probabilities and an "unsure" flag below 50% confidence; every pick shows rules fired and the CSP check.
- Leakage control: no PR-derived fields as inputs; temporal and unseen-repo evaluation.
- Honest limitations printed in `report/RESULTS.md`; negative results kept (time-to-close).
- Rate limits respected and responses cached; only public data is read.
- Bias: popular repos dominate the data; report the repo-size breakdown (`report/eda/02_repo_distribution.png`) and the language mix.
- LLM guardrails: facts-only prompt, invented numbers rejected, extractive RAG fallback, groundedness score on every answer.
- Fairness risk to discuss: over-recommending already popular repos and English-only issues.
