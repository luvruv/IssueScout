# IssueScout

Find open GitHub issues you can finish, and get a kit to make your first PR.
**ML predicts, rules and search decide, the LLM only explains.**

```
python cli.py status        # where is the project? (checks vs targets)
python cli.py recommend --skills go,docker --level intermediate --hours 6 --weeks 3 --verbose
python cli.py predict kubernetes/kubernetes#12345      # difficulty of ANY issue
python cli.py prepare owner/repo#123                   # first-PR kit
python cli.py ask owner/repo "How do I run the tests?" # RAG over CONTRIBUTING/README
python cli.py web                                      # browser UI
```
Start with `docs/STEP_BY_STEP.md`. Results land in `report/RESULTS.md`.

| Folder | What |
|---|---|
| collector/ | GitHub API -> SQLite (cached) |
| features/ | label definition (effort percentile), leakage-free features |
| ml/ | EDA, difficulty/merge/time models, clustering, anomaly, PCA, GD demo, evaluation, predict |
| ai/ | ontology, forward-chaining rules, CSP, A*, Bayes, benchmarks |
| llm/ | provider client, skill extraction, grounded explanations, RAG, first-PR kit, evaluations |
| core/ | recommend(), outcomes log |
| api/, app/, cli.py | FastAPI, Streamlit, command line |
| tests/ | `python cli.py test` |

Responsible use: IssueScout never comments on or opens anything on GitHub for you. Predictions are hints, shown with probabilities.
