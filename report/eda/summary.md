# EDA summary

- issues in DB: 9276 (closed 1656, open 7620)
- repos: 87; languages: {'Go': 68, 'Python': 14, 'TypeScript': 1, 'Shell': 1, 'Jinja': 1, 'C++': 1, 'Rust': 1}
- labelled (merged PR) issues used for training: 1397 in 81 repos
- duplicate ids: 0; duplicate titles within a repo: 24
- missing share: {'body': 0.0, 'labels': 0.0, 'first_response_hours': 0.908, 'pr_files': 0.821, 'pr_review_comments': 0.821}
- class distribution: {'Easy': 0.401, 'Medium': 0.397, 'Hard': 0.202}  (imbalance ratio max/min = 1.99)
- 'good first issue' share: open 0.008, labelled 0.022
- top repo holds 12.3% of labelled issues (repo skew)
- features most correlated with effort: kw_big (+0.11), is_docs (+0.09), n_labels (+0.08), kw_lint (+0.06), is_help_wanted (+0.05)
- maintainer first reply: median 11.5 h, 90th pct 421 h (n=854)
- extreme body-length outliers (3xIQR): 50; kept (real issues), log-scaled in plots
- difficulty split by 'good first issue' label:
difficulty   Easy  Medium   Hard
is_gfi                          
0           0.398   0.398  0.204
1           0.516   0.355  0.129
- top terms Easy: goreleaser, github, https, com, version, audit, test, blob, file, error
- top terms Medium: github, https, com, cli, version, http, error, file, test, command
- top terms Hard: goreleaser, audit, github, com, https, 628c889d7f2a34066649833be419fdb9abe01174, agents, blob, source, example
