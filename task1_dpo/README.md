# Task 1 — Direct Preference Optimization

## Layout

| File | Role |
|---|---|
| `dpo.py` | DPO objective. **Defect fixed:** the starter used `beta*(policy_margin + ref_margin)`; the manual's objective is `beta*(policy_margin - ref_margin)` |
| `validate_objective.py` | Numerical checks of the objective and of the log-prob plumbing → `results/task1_dpo/objective_validation.json` |
| `train.py` | DPO training loop (gradient accumulation, fp16 loss scaling, reference = same network with LoRA disabled, per-step log, adapter save) |
| `evaluate.py` | Common evaluation protocol for one policy: held-out pairs, generations (length / KL / reward), word-limit prompts, length strata |
| `ablate_beta.py` | Step 2: β ∈ {0.03, 0.10, 0.30} short forks + evaluation |
| `analyze_length.py` | Step 3: dataset length statistics, length-balanced training, stratified evaluation |
| `summarize.py` | Tables → `report/tables/task1_*.csv`, figures → `report/figures/task1/` |
| `diagnostics.py` | CPU-only: paired reward vs SFT (bootstrap CI), identical-to-SFT fraction, summed vs per-token stratum accuracy, fp16 skipped steps |
| `utils.py` | Shared Task 1 helpers (eligibility rule, pair log-probs, generation scoring) — *student addition* |
| `run_all.sh` | The exact command sequence, as two GPU lanes + summary — *student addition* |

Shared additions outside this folder: `common/experiment.py` (run metadata, smoke overrides, VRAM),
`common/generation.py::response_token_logprobs_lean` (memory-lean log-probs for generated batches),
`scripts/run_lanes.py` (one process per GPU).

## Reproduce

```bash
python -m scripts.download_assets && python -m scripts.validate_assets
python -m scripts.run_lanes --tag task1 "bash task1_dpo/run_all.sh lane0" "bash task1_dpo/run_all.sh lane1"
bash task1_dpo/run_all.sh summary
```
CPU plumbing test (0.5B model, minutes): add `--smoke` to any entry point; outputs go to `outputs/smoke/`, `results/smoke/` (git-ignored).

## Outputs

| Path | Content |
|---|---|
| `results/task1_dpo/runs/<run>/train_log.jsonl` | one row per optimizer step: loss, batch preference accuracy, margins, implicit rewards, grad norm |
| `results/task1_dpo/runs/<run>/train_summary.json` | config, git commit, wall-clock, peak VRAM, n pairs, steps |
| `results/task1_dpo/runs/<run>/train_ids.json` | exact training pair IDs used + pairs dropped by the prompt rule |
| `results/task1_dpo/eval/<model>/summary.json` | all aggregate metrics per part + metadata |
| `results/task1_dpo/eval/<model>/{heldout_pairs,generations,word_limit,stratified_pairs}.jsonl` | per-item records (qualitative examples come from these) |
| `results/task1_dpo/dataset_length_stats.json` | length structure of the preference files |

Models: `sft` (untouched base, the reference), `standard`, `beta_0.03`, `beta_0.10`, `beta_0.30`, `length_balanced`.

## External code

No external implementation reused. Library APIs only (PyTorch, Transformers, PEFT).
