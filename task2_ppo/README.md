# Task 2 — Proximal Policy Optimization (continuation from the supplied midpoint)

## Layout

| File | Role |
|---|---|
| `ppo.py` | PPO helpers. **Defect fixed:** the starter's clipped surrogate used `torch.maximum(rho*A, clip(rho)*A)` (optimistic branch, no trust region); the manual's objective is the `min`. Added `clipping_diagnostics` (clip fraction, affected-token fraction, clipped/unclipped surrogate) |
| `validate_objective.py` | Numerical checks of the surrogate, gradient behaviour when the clip binds, clip-fraction definition, GAE vs direct sum, reward shaping → `results/task2_ppo/objective_validation.json` |
| `continue_train.py` | PPO continuation loop: rollout → reward (−`missing_eos_penalty` without EOS) → old/ref log-probs, critic values → KL-shaped rewards → GAE → whitening → `ppo_epochs` policy + critic steps; per-update log, adapter save |
| `evaluate.py` | Held-out evaluation of one frozen policy (protocol in `common/rl.py`) |
| `analyze_clipping.py` | Step 2: cached-batch geometry per ε + matched ε forks (+ evals) |
| `ablate_kl.py` | Step 3: β_KL forks (+ evals); β = 0.10 reuses the ε = 0.20 fork |
| `summarize.py` | Tables → `report/tables/task2_*.csv`, figures → `report/figures/task2/` |
| `run_all.sh` | The exact command sequence (`lane0`, `kl`, `summary`; `PREFLIGHT=1` for a short check) — *student addition* |
| `NOTES.md` | Declared design choices and hypotheses — *student addition* |

Shared additions: `common/rl.py` (config overrides, prompt eligibility and schedule, dropout off,
held-out evaluation, subprocess runner), used by Tasks 2 and 3.

## Reproduce

```bash
python -m scripts.download_assets && python -m scripts.validate_assets
python -m scripts.run_lanes --tag task23_full "bash task2_ppo/run_all.sh lane0" "bash task3_grpo/run_all.sh lane1 && bash task2_ppo/run_all.sh kl"
bash task2_ppo/run_all.sh summary
```

## Outputs

| Path | Content |
|---|---|
| `results/task2_ppo/runs/<run>/train_log.jsonl` | one row per update: reward (raw, effective), KL, entropy, length, EOS/truncation, policy/value loss, clip and affected fraction, grad norms, critic value/return means, explained variance, per-epoch details |
| `results/task2_ppo/runs/<run>/rollouts.jsonl` | every training rollout (prompt id, response, reward, KL) |
| `results/task2_ppo/runs/<run>/schedule.json` | exact prompt ids per update |
| `results/task2_ppo/runs/<run>/train_summary.json` | config, git commit, wall-clock, peak VRAM, generated tokens |
| `results/task2_ppo/eval/<model>/{generations.jsonl,summary.json}` | held-out per-prompt records and aggregates |
| `results/task2_ppo/clipping/{cached_study.json,cached_steps.jsonl}` | cached-batch clipping geometry |
| `results/task2_ppo/objective_validation.json` | objective validation evidence |
| `outputs/task2_ppo/{standard,forks/*}` | LoRA adapters (git-ignored; kept in the Kaggle version output) |
