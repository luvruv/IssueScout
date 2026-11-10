# EDA summary

- issues in DB: 21330 (closed 4813, open 16517)
- repos: 127; languages: {'Go': 68, 'Python': 36, 'Rust': 10, 'JavaScript': 6, 'TypeScript': 4, 'Shell': 1, 'Jinja': 1, 'C++': 1}
- labelled (merged PR) issues used for training: 4000 in 121 repos
- duplicate ids: 0; duplicate titles within a repo: 54
- missing share: {'body': 0.0, 'labels': 0.0, 'first_response_hours': 0.864, 'pr_files': 0.774, 'pr_review_comments': 0.774}
- class distribution: {'Easy': 0.4, 'Medium': 0.398, 'Hard': 0.202}  (imbalance ratio max/min = 1.98)
- 'good first issue' share: open 0.006, labelled 0.027
- top repo holds 5.3% of labelled issues (repo skew)
- features most correlated with effort: kw_big (+0.15), n_labels (+0.13), repo_age_days (+0.10), has_contributing (+0.09), has_link (+0.08)
- maintainer first reply: median 13.1 h, 90th pct 591 h (n=2905)
- extreme body-length outliers (3xIQR): 136; kept (real issues), log-scaled in plots
- difficulty split by 'good first issue' label:
difficulty   Easy  Medium   Hard
is_gfi                          
0           0.399   0.399  0.203
1           0.450   0.376  0.174
- top terms Easy: github, https, com, version, test, file, error, python, bug, mocha
- top terms Medium: https, github, com, version, error, file, python, bug, test, cli
- top terms Hard: goreleaser, github, audit, https, com, pylint, 628c889d7f2a34066649833be419fdb9abe01174, version, error, blob
