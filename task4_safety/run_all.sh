#!/usr/bin/env bash
# Task 4 (safety calibration): every command that produces a reported number.
#   bash task4_safety/run_all.sh gpu     # link adapters -> generate (SFT/DPO/PPO/GRPO) -> audit sheet -> AI judge -> rates
#   bash task4_safety/run_all.sh rates   # tables + figure from saved generations/judgements (CPU)
#   bash task4_safety/run_all.sh audit   # judge/manual agreement, after manual_audit_sheet.csv is filled (CPU)
# PREFLIGHT=1: 4 SAFE + 4 UNSAFE prompts, 2+2 audit prompts; outputs only under results/preflight/ (git-ignored).
# Every step skips work whose outputs already exist, so the lane can be re-run after an interruption.
set -euo pipefail
CFG=configs/feedback.yaml
SET=(); RES=results
if [[ "${PREFLIGHT:-0}" == "1" ]]; then
  SET=(--set results_dir=results/preflight safety_limit_per_class=4 manual_audit_per_class=2)
  RES=results/preflight
fi

case "${1:-}" in
  gpu)
    python -m scripts.link_adapters --config $CFG --results-dir $RES
    for P in sft dpo ppo grpo; do
      python -m task4_safety.generate_responses --config $CFG --policy $P "${SET[@]}"
    done
    python -m task4_safety.make_audit_sheet --config $CFG "${SET[@]}"
    python -m task4_safety.judge_responses  --config $CFG "${SET[@]}"
    python -m task4_safety.evaluate_safety  --config $CFG --stage rates "${SET[@]}"
    ;;
  rates)
    python -m task4_safety.evaluate_safety  --config $CFG --stage rates "${SET[@]}"
    ;;
  audit)
    python -m task4_safety.evaluate_safety  --config $CFG --stage agreement "${SET[@]}"
    ;;
  *)
    echo "usage: $0 gpu|rates|audit" >&2; exit 2 ;;
esac
