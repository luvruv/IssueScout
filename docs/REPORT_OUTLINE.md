# Report outlines mapped to your two syllabi

## ML report
| Syllabus topic | Where | Evidence file |
|---|---|---|
| Types of learning, problems, applications | intro | write-up |
| Cost function, gradient descent, sampling, decision boundary, under/overfitting, bias-variance | `ml/gradient_descent_demo.py`, `ml/evaluate.py`, temporal + grouped splits | `report/gd_cost_curve.png`, `decision_boundary.png`, `learning_curves.png` |
| Regression, Ridge, Lasso | `ml/train_time.py` | `report/time_comparison.md`, `regularisation_effect.png` (negative result, explain why) |
| Logistic regression, SVM, decision tree | `ml/train_difficulty.py` | `report/model_comparison.md` |
| Ensembles: bagging, random forest, boosting | same | same |
| KNN, ANN intro | KNN(cosine), MLP(64) in the same table | same |
| Clustering: K-Means, DBSCAN, BIRCH; anomaly detection | `ml/cluster.py` | `pca_repos.png`, `cluster_profiles.csv`, `kdistance.png` |
| Probability, Bayes rule, MLE | `ai/bayes.py`, `python cli.py bayes-demo` | table of MLE vs Bayesian rates |
| PCA, kernel PCA | `ml/cluster.py` | `pca_repos.png` |
| Validation: accuracy, CI, confusion matrix, precision/recall, ROC, CV, bootstrap, tuning, R2 | `train_difficulty.py` | `confusion_difficulty.png`, `roc_difficulty.png`, `calibration_difficulty.png`, the CI column |
| Model deployment | joblib + `api/main.py` + `cli.py predict` | `/predict` endpoint |
| Reinforcement learning intro | theory section only (frame recommendation as a contextual bandit; not implemented) | write-up |
Also cover: label definition, leakage, label validation (kappa), cost-sensitive metric (`test_cost`, `hard_called_easy`), limitations.

## AI report
| Syllabus topic | Where | Evidence |
|---|---|---|
| AI intro, agents, PEAS, expert systems | `docs/RESPONSIBLE_AI.md`, `ai/rules.py` | rule trace in `recommend --verbose` |
| Search, heuristics, A*, admissibility | `ai/astar.py`, `core/recommend.py: learning_path` | `report/astar_benchmark.md` |
| Planning and CSP | `ai/csp.py` | `report/csp_benchmark.md` |
| Knowledge representation and reasoning | `ai/ontology.py` | graph + `missing()` examples |
| Generative AI, VAE/GAN, Transformer, attention | theory section (no demo needed); link to how the LLM/embeddings are used here | write-up |
| LLMs: prompting, context, evaluation, LangChain | `llm/` | `report/metrics.json: llm_eval, rag_eval` |
| Responsible AI | `docs/RESPONSIBLE_AI.md` | guardrail results |
