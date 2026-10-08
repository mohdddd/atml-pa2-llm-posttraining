#!/usr/bin/env bash
# Task 1 (DPO): every command that produces a reported number, split into two GPU lanes.
#   bash task1_dpo/run_all.sh lane0     # standard DPO -> eval -> length-confounding study
#   bash task1_dpo/run_all.sh lane1     # objective validation -> beta forks -> SFT reference eval
#   bash task1_dpo/run_all.sh summary   # tables + figures (CPU)
# Every step skips work whose outputs already exist, so a lane can be re-run after an interruption.
set -euo pipefail
CFG=configs/dpo.yaml

case "${1:-}" in
  lane0)
    python -m task1_dpo.train          --config $CFG --run-name standard
    python -m task1_dpo.evaluate       --config $CFG --adapter outputs/task1_dpo/standard --name standard --beta 0.10
    python -m task1_dpo.analyze_length --config $CFG --stage all
    ;;
  lane1)
    python -m task1_dpo.validate_objective --with-model
    python -m task1_dpo.ablate_beta    --config $CFG --stage all
    python -m task1_dpo.evaluate       --config $CFG --adapter none --name sft --parts heldout,generate,wordlimit,stratified
    ;;
  summary)
    python -m task1_dpo.summarize      --config $CFG
    ;;
  *)
    echo "usage: $0 lane0|lane1|summary" >&2; exit 2 ;;
esac
