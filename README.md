# ATML PA2 — LLM Post-Training (DPO · PPO · GRPO · safety calibration · RLVR vs RLAIF)

Code, configs, fixed IDs, logs and machine-readable results for every experiment in the PA2 report.
Every reported number comes from a file under `results/` (per-item records) or `report/tables/` (aggregates),
written by a Python command listed below. Each result file stores the config and git commit it was produced with.

## Repository layout

```text
configs/          YAML settings for every task (base.yaml, dpo.yaml, ppo.yaml, grpo.yaml, feedback.yaml)
common/           shared code: data/model loading, generation, metrics, RL helpers (rl.py), frozen-policy eval (frozen_eval.py)
task1_dpo/        DPO: objective + fix, training, beta forks, length-confounding study      (NOTES.md = design record)
task2_ppo/        PPO: objective + fix, continuation, clipping study, KL-pressure forks      (NOTES.md)
task3_grpo/       GRPO: advantage fix, continuation, group-size study, normalisation forks   (NOTES.md)
task4_safety/     XSTest generation, fixed AI judge, blinded manual audit, safety rates        (NOTES.md)
task5_feedback/   RLVR vs RLAIF: GSM8K/SVAMP evaluation, pairwise judge, controlled diagnostics (NOTES.md)
scripts/          asset download/validation, environment check, run_lanes (one lane per GPU), adapter linker,
                  qualitative-example shortlists
results/          per-item outputs (generations, judgements, pair records), run logs, fixed IDs, summaries
report/tables/    aggregate CSVs used in the report;  report/figures/ PDF + PNG figures
data/ cached/ checkpoints/ manifests/   course assets (downloaded, git-ignored);  outputs/ trained adapters (git-ignored)
```

## 1. Setup

```bash
git clone https://github.com/mohdddd/atml-pa2-llm-posttraining.git
cd atml-pa2-llm-posttraining
python -m pip install -r requirements.txt
python -m scripts.download_assets      # course data, cached rollouts and supplied checkpoints (HF dataset
python -m scripts.validate_assets      #   AbDu11aHHH/ATML-PA2-assets @ 0b350481fb03f5525a35bcdec4131bd4fe487f98)
python -m scripts.check_environment
```

Public models (Qwen2.5-1.5B-Instruct policy, yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback reward model,
Qwen2.5-0.5B-Instruct value init, Qwen2.5-3B-Instruct judge) download from Hugging Face at run time.
All runs used Kaggle **2× T4 (14.6 GiB each)**, Python 3.13, torch 2.11.0+cu128, transformers 4.57.1, tokenizers 0.22.1,
PEFT 0.17.1, TRL 0.27.2, bitsandbytes 0.50.2 (`environment_freeze.txt`). Seed 6304 everywhere.
Run every command from the repository root.

## 2. Commands that produce every reported result

Each task has a `run_all.sh` whose lanes run on separate GPUs through `scripts.run_lanes`
(lane *i* gets `CUDA_VISIBLE_DEVICES=i`; logs in `results/logs/<tag>_lane<i>.log`). With one GPU the lanes run one
after another. Every step skips work whose outputs already exist (`--overwrite` to redo), so an interrupted lane can
be re-run. Prefixing a lane command with `PREFLIGHT=1` (Tasks 2–5) runs the same code on tiny budgets and writes only
to the git-ignored `results/preflight/`.

### Task 1 — DPO (GPU ≈ 1 h on 2× T4)
```bash
python -m scripts.run_lanes --tag task1 "bash task1_dpo/run_all.sh lane0" "bash task1_dpo/run_all.sh lane1"
#   lane0: standard DPO (1 epoch, beta 0.10) -> evaluation -> length-balanced DPO + stratified / word-limit analysis
#   lane1: objective validation -> beta forks {0.03, 0.10, 0.30} (600 pairs) -> SFT reference evaluation
bash task1_dpo/run_all.sh summary      # CPU: report/tables/task1_*.csv, report/figures/task1/, diagnostics
```

### Tasks 2 + 3 — PPO and GRPO (GPU ≈ 70 min on 2× T4)
```bash
python -m scripts.run_lanes --tag task23_full \
  "bash task2_ppo/run_all.sh lane0" \
  "bash task3_grpo/run_all.sh lane1 && bash task2_ppo/run_all.sh kl"
#   PPO lane0: objective validation -> standard 20-update continuation -> held-out eval (midpoint, standard)
#              -> cached-rollout clipping study -> epsilon forks {0.05, 0.20, 0.50} x 8 updates + evals
#   GRPO lane1: advantage validation -> standard 20-update continuation (K=4) -> evals (midpoint, standard, SFT)
#              -> canonical vs Dr.GRPO forks x 8 updates + evals + length-conditioned gradient analysis
#   PPO kl:    KL-pressure forks beta_KL {0, 0.20} (+ 0.10 = the shared eps 0.20 fork) + evals
bash task2_ppo/run_all.sh summary && bash task3_grpo/run_all.sh summary
#   CPU: group-size study on cached/grpo_k_cache.jsonl, report/tables/task{2,3}_*.csv, report/figures/task{2,3}/
```

### Tasks 4 + 5 — safety calibration and RLVR vs RLAIF (GPU ≈ 45 min on 2× T4)
Task 4 needs the three **standard** adapters at `outputs/task1_dpo/standard`, `outputs/task2_ppo/standard`,
`outputs/task3_grpo/standard` (produced by the Task 1 and Task 2+3 runs). `python -m scripts.link_adapters`
(called by the lane) finds them under `/kaggle/input/**` when the earlier run notebooks' outputs are attached as inputs,
checks them, and records their sha256 in `results/task4_safety/adapters.json`.
```bash
python -m scripts.run_lanes --tag task45 "bash task4_safety/run_all.sh gpu" "bash task5_feedback/run_all.sh gpu"
#   Task 4: greedy generation SFT/DPO/PPO/GRPO x 450 XSTest prompts -> blinded audit sheet -> fixed AI judge -> rates
#   Task 5: greedy SFT/RLVR/RLAIF on GSM8K (300) + SVAMP (100) -> exact verifier + fixed pairwise judge
#           -> controlled diagnostic set (20 x 5) -> tables
# manual audit (human, CPU): label results/task4_safety/manual_audit_sheet.csv (or the offline page
#   manual_audit_labeler.html) without opening judged_*.jsonl, then:
bash task4_safety/run_all.sh audit     # CPU: judge/manual agreement tables
```
CPU-only re-aggregation from saved files: `bash task4_safety/run_all.sh rates`, `bash task5_feedback/run_all.sh summary`.

### Qualitative-example candidates (CPU, seconds)
```bash
python -m scripts.qualitative_shortlist   # results/qualitative/task{1,2,3}_shortlist.jsonl
# Tasks 4/5 shortlists: results/task4_safety/qualitative_shortlist.jsonl, results/task5_feedback/qualitative_shortlist.jsonl
```

## 3. Where the results are

| Task | Aggregates (`report/tables/`) | Per-item / run records (`results/`) | Figures (`report/figures/`) |
|---|---|---|---|
| 1 DPO | `task1_dpo_summary`, `task1_length_strata`, `task1_strata_summed_vs_per_token`, `task1_word_limit`, `task1_paired_reward`, `task1_dataset_length_stats` | `task1_dpo/{runs,eval}/*`, `objective_validation.json`, `diagnostics.json`, `runs/standard/train_ids.json` | `task1/` |
| 2 PPO | `task2_standard_trajectory`, `task2_heldout`, `task2_clip_cached`, `task2_clip_forks`, `task2_kl_forks`, `task2_compute` | `task2_ppo/{runs,eval,clipping}/*`, `objective_validation.json`, `runs/standard/schedule.json` | `task2/` |
| 3 GRPO | `task3_standard_trajectory`, `task3_heldout`, `task3_group_size`, `task3_normalization`, `task3_compute` | `task3_grpo/{runs,eval,group_size,normalization}/*`, `objective_validation.json` | `task3/` |
| 4 Safety | `task4_safety_summary`, `task4_category_labels(_long)`, `task4_label_transitions_vs_sft`, `task4_audit_*` | `task4_safety/{generated,judged}_<policy>.jsonl`, `manual_audit_{ids,sheet,key}.csv`, `adapters.json` | `task4/` |
| 5 RLVR/RLAIF | `task5_in_domain`, `task5_transfer`, `task5_pairwise_and_agreement`, `task5_failure_types`, `task5_diagnostic_{pairs,sensitivity,variants}`, `task5_compute` | `task5_feedback/{gsm,transfer}/*`, `diagnostics/*`, `judge_cache*.json` | `task5/` |

## 4. Defects found in the starter objectives (Tasks 1–3)

| Task | Starter code | Fix | Evidence |
|---|---|---|---|
| 1 DPO | `beta * (policy_margin + ref_margin)` | `beta * (policy_margin - ref_margin)` | `results/task1_dpo/objective_validation.json` (`task1_dpo/validate_objective.py`) |
| 2 PPO | clipped surrogate used `torch.maximum` | `torch.minimum` (pessimistic bound) | `results/task2_ppo/objective_validation.json` (`task2_ppo/validate_objective.py`) |
| 3 GRPO | advantages normalised over the whole batch | per-group mean/std normalisation | `results/task3_grpo/objective_validation.json` (`task3_grpo/validate_objective.py`) |

Design choices the manual leaves open are recorded per task in `task*/NOTES.md`; known limitations are listed there too.

## 5. Reproducibility rules followed

- Course data, cached rollouts and supplied checkpoints are never modified; raw data and weights are not committed.
- Every short fork starts from the same supplied midpoint checkpoint; prompt IDs, budgets, seed and evaluation
  procedure are matched across ablations (fixed schedules saved in `results/*/runs/*/schedule.json` / `train_ids.json`).
- Task 4 safety labels and Task 5 evaluation/diagnostic data were not used to tune any earlier training.
- Peak VRAM and wall-clock time of the standard PPO and GRPO continuations: `report/tables/task{2,3}_compute.csv`.

## 6. Attribution

- **Course starter repository** (ATML PA2 student starter, `AbDu11aHHH/ATML-PA2-LLM-PostTraining`): repository structure,
  data/model loaders, generation and metric helpers, the objective functions corrected above, the fixed Task 4 safety
  judge (prompt, parser, `judge_one`), the Task 5 exact verifier (`task5_feedback/rlvr.py`) and pairwise AI judge
  (`task5_feedback/rlaif.py`), asset download/validation scripts. Student changes are visible as diffs on top of the
  starter history in git.
- **Course assets**: Hugging Face dataset `AbDu11aHHH/ATML-PA2-assets` (fixed subsets of UltraFeedback-derived preference
  data, XSTest, GSM8K, the SVAMP challenge set, the Controlled Reward Diagnostic Set; supplied PPO/GRPO midpoints and
  RLVR/RLAIF adapters).
- **Libraries**: PyTorch, Hugging Face Transformers, PEFT, TRL, bitsandbytes, Accelerate, Datasets, pandas, NumPy,
  scikit-learn (Cohen's kappa), matplotlib.
- **Datasets**: XSTest (Röttger et al., 2023), GSM8K (Cobbe et al., 2021), SVAMP (Patel et al., 2021), UltraFeedback
  (Cui et al., 2023).
- **Tooling**: code in this repository was written with AI coding assistance (Claude); every commit made with it is
  marked `Co-Authored-By` in the git history. No external code was copied beyond the starter repository.
