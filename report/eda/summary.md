# EDA summary

- issues in DB: 2709 (closed 551, open 2158)
- repos: 31; languages: {'Go': 24, 'Python': 7}
- labelled (merged PR) issues used for training: 466 in 30 repos
- duplicate ids: 0; duplicate titles within a repo: 5
- missing share: {'body': 0.0, 'labels': 0.0, 'first_response_hours': 0.883, 'pr_files': 0.797, 'pr_review_comments': 1.0}
- class distribution: {'Easy': 0.401, 'Medium': 0.399, 'Hard': 0.2}  (imbalance ratio max/min = 2.00)
- 'good first issue' share: open 0.012, labelled 0.049
- top repo holds 14.6% of labelled issues (repo skew)
- features most correlated with effort: kw_big (+0.14), is_docs (+0.14), is_gfi (+0.09), has_repro (+0.09), n_labels (+0.08)
- maintainer first reply: median 12.8 h, 90th pct 485 h (n=316)
- extreme body-length outliers (3xIQR): 7; kept (real issues), log-scaled in plots
- difficulty split by 'good first issue' label:
difficulty   Easy  Medium   Hard
is_gfi                          
0           0.391   0.409  0.201
1           0.609   0.217  0.174
- top terms Easy: github, cli, https, com, version, v3, test, http, urfave, error
- top terms Medium: cli, github, https, com, v3, command, cmd, version, flag, context
- top terms Hard: github, cli, https, com, helm, help, command, webhook, manager, pytest
