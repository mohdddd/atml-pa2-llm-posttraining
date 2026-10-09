# Task 3 — design record

## Settings taken unchanged from the release (`configs/grpo.yaml`, `configs/base.yaml`)
20 updates (standard) / 8 updates (forks); 1 prompt per update; K = 4; 1 policy epoch; LR 5e-6; ε = 0.20; β = 0.10;
prompt cap 256; completion cap 512 with max-length completions masked from the loss; grad-norm clip 1.0;
sampling temperature 0.7, top-p 0.9; seed 6304.

## Declared design choices
1. **Prompt eligibility, schedule, sampling seeds, dropout off:** identical to Task 2 (`task2_ppo/NOTES.md` 1–3).
2. **Advantage:** A_k = (r_k − μ)/(σ + 1e-6) with population σ inside each prompt group. A group is uninformative when
   σ ≤ 1e-6 (the helper's ε). Rewards of truncated completions still enter the group statistics; only their loss is masked.
3. **KL in the loss:** the released k3 estimator averaged over loss tokens (unchanged for both normalisations).
   The reported KL is the common sampled-response estimator (token mean of log π − log π_ref over all generated tokens).
4. **Added config keys:** `reward_max_length: 1280`, `eval_max_response_length: 768` (same as Task 2 and the K-cache).
5. **Group-size study:** each prompt's 8 cached completions split into 8/K disjoint consecutive groups (generation_index
   order), so every K uses all 192 generations (96/48/24 groups). Robustness: expectation over all C(8,K) subsets.
   Difficulty bins: terciles of each prompt's 8-completion mean reward (8 prompts per bin). Additional thresholds
   σ > {0.10, 0.25, 0.50}, and a diagnostic with rewards binarised at the median of all 192 cached rewards.
   Note: for K = 8 the baseline MSE is 0 and sign agreement is 1 by construction (the group is the full 8-sample set).
6. **Length-conditioned statistic:** per completion, the L2 norm of the gradient of its own policy term (its share of the
   batch loss, without KL) w.r.t. the LoRA parameters, measured on the first-epoch forward pass. Length bins = terciles
   of the pooled lengths of loss-contributing completions of both forks. Also reported: implied per-token weight
   |A|/T (canonical) vs |A|/L_max (Dr. GRPO), Spearman(length, grad norm) overall and by advantage sign.
7. **Held-out protocol:** identical to Task 2 (165 prompts, cap 768). Also evaluated: SFT (no adapter) and the
   supplied GRPO midpoint.

## Hypotheses (student, written BEFORE the full run — fill in or delete; do not backdate)
- Group size:
- Normalisation:
- Dominant instability without a critic:
