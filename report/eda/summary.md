# EDA summary

- issues in DB: 8655 (closed 1576, open 7079)
- repos: 83; languages: {'Go': 67, 'Python': 11, 'TypeScript': 1, 'Shell': 1, 'Jinja': 1, 'C++': 1, 'Rust': 1}
- labelled (merged PR) issues used for training: 1359 in 77 repos
- duplicate ids: 0; duplicate titles within a repo: 24
- missing share: {'body': 0.0, 'labels': 0.0, 'first_response_hours': 0.909, 'pr_files': 0.818, 'pr_review_comments': 0.818}
- class distribution: {'Easy': 0.401, 'Medium': 0.397, 'Hard': 0.202}  (imbalance ratio max/min = 1.99)
- 'good first issue' share: open 0.009, labelled 0.023
- top repo holds 12.7% of labelled issues (repo skew)
- features most correlated with effort: is_docs (+0.12), kw_big (+0.11), n_labels (+0.08), kw_lint (+0.07), n_words (+0.06)
- maintainer first reply: median 13.6 h, 90th pct 459 h (n=789)
- extreme body-length outliers (3xIQR): 47; kept (real issues), log-scaled in plots
- difficulty split by 'good first issue' label:
difficulty   Easy  Medium   Hard
is_gfi                          
0           0.398   0.398  0.204
1           0.516   0.355  0.129
- top terms Easy: goreleaser, github, https, com, version, audit, test, blob, file, error
- top terms Medium: github, cli, https, com, version, http, error, file, test, v3
- top terms Hard: goreleaser, audit, github, com, https, 628c889d7f2a34066649833be419fdb9abe01174, agents, blob, source, example
