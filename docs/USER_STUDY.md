# Small user study (5-10 classmates)
Give each person a profile (skills, hours) and the plan IssueScout produced. Ask them to open the 2-3 issues and answer (1 = no, 5 = yes):
1. This plan looks doable for me. (`doable_1to5`)
2. I understand why each issue was chosen. (`explanation_clear_1to5`)
3. I would try one of these issues. (`would_try_1to5`)
4. Free comment.
Save as `data/user_study.csv` with columns `participant,doable_1to5,explanation_clear_1to5,would_try_1to5,comments`, then run `python scripts/user_study_summary.py`.
Report n, means and spread, and say it is a small convenience sample, not proof.
