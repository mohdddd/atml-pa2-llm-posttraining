# Task 3 — Group Relative Policy Optimization (continuation from the supplied midpoint)

## Layout

| File | Role |
|---|---|
| `grpo.py` | GRPO helpers. **Defect fixed:** the starter's `group_relative_advantages` normalised with the mean/std of the whole batch, ignoring `group_ids`; the manual's advantage uses the mean/std inside each prompt's K-completion group. Added `group_reward_stats` (per-group std, uninformative flag) |
| `validate_objective.py` | Numerical checks of per-group advantages, the canonical and Dr. GRPO losses, the KL term and truncation masking → `results/task3_grpo/objective_validation.json` |
| `continue_train.py` | GRPO continuation loop: K completions per prompt → RM rewards → group advantages → truncated completions masked from the loss → clipped objective + β·KL; per-completion gradient-norm probe; per-update log, adapter save |
| `evaluate.py` | Held-out evaluation (same protocol as Task 2, `common/rl.py`) |
| `analyze_group_size.py` | Step 2 (CPU): equal-generation K ∈ {2,4,8} study on the supplied cache |
| `compare_normalization.py` | Step 3: canonical vs Dr. GRPO forks (+ evals) and length-conditioned analysis |
| `summarize.py` | Tables → `report/tables/task3_*.csv`, figures → `report/figures/task3/` |
| `run_all.sh` | `lane1`, `summary`; `PREFLIGHT=1` for a short check — *student addition* |
| `NOTES.md` | Declared design choices and hypotheses — *student addition* |

## Reproduce

```bash
python -m scripts.download_assets && python -m scripts.validate_assets
python -m scripts.run_lanes --tag task23_full "bash task2_ppo/run_all.sh lane0" "bash task3_grpo/run_all.sh lane1 && bash task2_ppo/run_all.sh kl"
bash task3_grpo/run_all.sh summary
```

## Outputs

| Path | Content |
|---|---|
| `results/task3_grpo/runs/<run>/train_log.jsonl` | per update: reward, KL, within-group reward std, uninformative-group fraction, policy loss, grad norm, entropy, length, truncation/masking, per-group stats |
| `results/task3_grpo/runs/<run>/completions.jsonl` | every completion: response, length, loss tokens, reward, advantage, own policy-gradient norm |
| `results/task3_grpo/eval/<model>/{generations.jsonl,summary.json}` | held-out records (models: sft, midpoint, standard, fork_grpo, fork_dr_grpo) |
| `results/task3_grpo/group_size/{group_size_study.json,groups.csv}` | group-size study |
| `results/task3_grpo/normalization/comparison.json` | length-conditioned gradient statistics + held-out comparison |
| `outputs/task3_grpo/{standard,forks/*}` | LoRA adapters (git-ignored; kept in the Kaggle version output) |
