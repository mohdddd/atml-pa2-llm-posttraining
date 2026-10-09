#!/usr/bin/env bash
# Task 2 (PPO): every command that produces a reported number.
#   bash task2_ppo/run_all.sh lane0    # validation -> standard 20-update run -> evals -> clipping study (cached + eps forks)
#   bash task2_ppo/run_all.sh kl       # KL-pressure forks beta=0.0 and 0.20 (+ evals); beta=0.10 is the eps=0.20 fork
#   bash task2_ppo/run_all.sh summary  # tables + figures (CPU)
# PREFLIGHT=1 runs the same code paths with 2 updates, 1-update forks, 8 eval prompts and 8 cached rollouts,
# writing only to results/preflight/ and outputs/preflight/ (both git-ignored).
# Every step skips work whose outputs already exist, so a lane can be re-run after an interruption.
set -euo pipefail
CFG=configs/ppo.yaml
SET=()
if [[ "${PREFLIGHT:-0}" == "1" ]]; then
  SET=(--set updates=2 fork_updates=1 eval_prompts=8 cached_max_rollouts=8
       results_dir=results/preflight/task2_ppo output=outputs/preflight/task2_ppo/standard)
fi
OUT=$([[ "${PREFLIGHT:-0}" == "1" ]] && echo outputs/preflight/task2_ppo || echo outputs/task2_ppo)

case "${1:-}" in
  lane0)
    python -m task2_ppo.validate_objective
    python -m task2_ppo.continue_train   --config $CFG --run-name standard "${SET[@]}"
    python -m task2_ppo.evaluate         --config $CFG --adapter checkpoints/ppo_midpoint_policy --name midpoint "${SET[@]}"
    python -m task2_ppo.evaluate         --config $CFG --adapter $OUT/standard --name standard "${SET[@]}"
    python -m task2_ppo.analyze_clipping --config $CFG --stage all "${SET[@]}"
    ;;
  kl)
    python -m task2_ppo.ablate_kl        --config $CFG "${SET[@]}"
    ;;
  summary)
    python -m task2_ppo.summarize        --config $CFG "${SET[@]}"
    ;;
  *)
    echo "usage: $0 lane0|kl|summary" >&2; exit 2 ;;
esac
