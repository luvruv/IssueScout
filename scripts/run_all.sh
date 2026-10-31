#!/usr/bin/env bash
# Whole pipeline, in order. Run from the project root (Git Bash on Windows is fine).  bash scripts/run_all.sh
set -e
python cli.py add-repos repo_lists              # syncs repos from repo_lists/ into repos.txt
python cli.py collect --closed-pages 6 --open-pages 3 --pr-pages 6    # SLOW: hours. cached + resumable: just run again
python cli.py features                           # labels + manual-label sample
python cli.py eda
python cli.py train all
python cli.py cluster
python cli.py evaluate
python cli.py gd-demo
python cli.py benchmark
python cli.py test
python cli.py report
python cli.py status
