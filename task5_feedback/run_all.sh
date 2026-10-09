#!/usr/bin/env bash
# Task 5 (RLVR vs RLAIF): every command that produces a reported number.
#   bash task5_feedback/run_all.sh gpu       # generate SFT/RLVR/RLAIF on GSM8K + SVAMP -> pairwise judge -> diagnostics -> tables
#   bash task5_feedback/run_all.sh summary   # metrics + tables + figure from saved files (CPU)
# PREFLIGHT=1: first 6 problems of each math set, 2 diagnostic problems; outputs only under results/preflight/.
# Every step skips work whose outputs already exist, so the lane can be re-run after an interruption.
set -euo pipefail
CFG=configs/feedback.yaml
SET=()
if [[ "${PREFLIGHT:-0}" == "1" ]]; then
  SET=(--set results_dir=results/preflight math_limit=6 diagnostic_limit=2)
fi

summary() {
  for DS in gsm transfer; do
    python -m task5_feedback.evaluate_math --config $CFG --dataset $DS --stage summary "${SET[@]}"
  done
  python -m task5_feedback.compare_feedback --config $CFG "${SET[@]}"
}

case "${1:-}" in
  gpu)
    for DS in gsm transfer; do
      for P in sft rlvr rlaif; do
        python -m task5_feedback.evaluate_math --config $CFG --dataset $DS --stage generate --policy $P "${SET[@]}"
      done
    done
    for DS in gsm transfer; do
      python -m task5_feedback.evaluate_math --config $CFG --dataset $DS --stage judge "${SET[@]}"
    done
    python -m task5_feedback.score_perturbations --config $CFG "${SET[@]}"
    summary
    ;;
  summary)
    summary ;;
  *)
    echo "usage: $0 gpu|summary" >&2; exit 2 ;;
esac
