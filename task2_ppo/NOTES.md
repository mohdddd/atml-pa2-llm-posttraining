# Task 2 — design record

## Settings taken unchanged from the release (`configs/ppo.yaml`, `configs/base.yaml`)
20 updates (standard) / 8 updates (forks); 1 prompt per update; 2 PPO epochs; policy LR 3e-6; critic LoRA LR 1e-4,
head LR 3e-4 (`lora_head` mode); ε = 0.20; β_KL = 0.10; γ = 1.0; λ = 0.95; value coefficient 0.5; missing-EOS
penalty 1.0; prompt cap 256 tokens; training response cap 512, evaluation cap 768; reward-model context 1280;
grad-norm clip 1.0; sampling temperature 0.7, top-p 0.9; seed 6304.

## Declared design choices (not fixed by the manual or the release)
1. **Prompt eligibility.** Prompts whose rendered chat prompt exceeds 256 tokens are excluded from training and
   evaluation (the starter generator cuts them from the right, removing the assistant header). Train pool
   1200 → 944 eligible, held-out pool 200 → 165. Same rule as Task 1.
2. **Prompt schedule.** Eligible train prompts shuffled once with seed 6304; update *u* takes the next prompt(s).
   Every run and fork uses the same sequence (a fork sees the first 8 prompts of the standard run).
   Sampling RNG is reset to `seed*1000 + u` before each rollout, so all forks start from the same first rollout.
3. **Dropout off during PPO updates** (policy LoRA and critic), so ρ = 1 exactly at the first epoch.
4. **Optimization granularity.** Each PPO epoch is one full-batch gradient step for the policy and one for the critic
   (separate optimizers; critic loss scaled by the value coefficient). Advantages whitened over the batch's valid tokens.
5. **Critic precision.** Trainable critic tensors kept in fp32; frozen critic body fp16; head fed fp32 hidden states.
   fp16 dynamic loss scaling for policy and critic (skipped steps are logged).
6. **Shared fork.** (ε = 0.20, β = 0.10) is identical in both studies and is run once.
7. **Cached-batch study.** Prompts of the 3 long cached rollouts are cut from the right to 256 tokens exactly as when the
   cache was produced (the rest is reconstructed verbatim; lengths checked against the cache). Advantages from the cached
   values and log-probs with β = 0.10. Step-0 geometry uses the released midpoint vs the cached old policy; the matched
   update per ε is 2 epochs × 4 shuffled minibatches of 8 rollouts (same order for every ε), LR 3e-6.
8. **Stability statistic (defined once):** maximum per-epoch KL(π_old‖π_new) (k3 estimator) over the fork; also reported:
   mean/max grad norm, policy-loss SD across updates, skipped steps.
9. **Held-out protocol:** 165 eligible held-out prompts, 1 sampled response each, cap 768, same seed and batch composition
   for every model; KL = token-averaged sampled-response estimator vs the reference (LoRA disabled); entropy = mean
   full-distribution token entropy over generated tokens. Also evaluated: the supplied midpoint (start of every run).

## Hypotheses (student, written BEFORE the full run — fill in or delete; do not backdate)
- Clipping study:
- KL-pressure study:
- Critic behaviour:

## Known limitation found after the run (applies to Task 1–3 held-out generation)
The sampling seed is reset once before the first evaluation batch, not per batch. When two models' first batch
ends at a different step (its longest response differs), the random stream of every later batch shifts, so later
batches are independent samples rather than common-random-number pairs. Verified on the Task 2/3 outputs: e.g.
midpoint vs ε-forks share 24/32 identical responses in batch 1 and ≤2/32 in later batches. Consequences:
- all per-model means/SEMs are valid (same prompts, same decoding, same protocol);
- paired differences vs the midpoint and their bootstrap CIs remain valid but are wider than a fully paired design;
- the `identical_text_vs_*` columns (and Task 1's identical-to-SFT diagnostic) do NOT measure how much a policy
  changed and should not be interpreted as such.
Not re-run because of the deadline; Tasks 4–5 use deterministic decoding and are unaffected.
