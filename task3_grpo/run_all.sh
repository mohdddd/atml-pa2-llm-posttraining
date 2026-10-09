#!/usr/bin/env bash
# Task 3 (GRPO): every command that produces a reported number.
#   bash task3_grpo/run_all.sh lane1    # validation -> standard 20-update run -> evals (midpoint, standard, SFT) -> normalisation forks
#   bash task3_grpo/run_all.sh summary  # group-size study on the supplied cache + tables + figures (CPU)
# PREFLIGHT=1: 2 updates, 1-update forks, 8 eval prompts; outputs only under results/preflight/, outputs/preflight/.
set -euo pipefail
CFG=configs/grpo.yaml
SET=()
if [[ "${PREFLIGHT:-0}" == "1" ]]; then
  SET=(--set updates=2 fork_updates=1 eval_prompts=8
       results_dir=results/preflight/task3_grpo output=outputs/preflight/task3_grpo/standard)
fi
OUT=$([[ "${PREFLIGHT:-0}" == "1" ]] && echo outputs/preflight/task3_grpo || echo outputs/task3_grpo)

case "${1:-}" in
  lane1)
    python -m task3_grpo.validate_objective
    python -m task3_grpo.continue_train        --config $CFG --run-name standard "${SET[@]}"
    python -m task3_grpo.evaluate              --config $CFG --adapter checkpoints/grpo_midpoint_policy --name midpoint "${SET[@]}"
    python -m task3_grpo.evaluate              --config $CFG --adapter $OUT/standard --name standard "${SET[@]}"
    python -m task3_grpo.evaluate              --config $CFG --adapter none --name sft "${SET[@]}"
    python -m task3_grpo.compare_normalization --config $CFG --stage all "${SET[@]}"
    ;;
  summary)
    python -m task3_grpo.analyze_group_size    --config $CFG "${SET[@]}"
    python -m task3_grpo.summarize             --config $CFG "${SET[@]}"
    ;;
  *)
    echo "usage: $0 lane1|summary" >&2; exit 2 ;;
esac
