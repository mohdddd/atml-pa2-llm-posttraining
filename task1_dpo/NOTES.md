# Task 1 design record

Committed before the GPU runs, so the ordering of decisions vs. results is visible in git history.

## Fixed by the course (not changed)
seed 6304 · Qwen2.5-1.5B-Instruct · LoRA r=8, α=16, dropout 0.05, q_proj/v_proj · lr 2e-5, AdamW, wd 0 ·
batch 2 pairs × grad-accum 8 · max_grad_norm 1.0 · max_sequence_length 768 · standard: 1 epoch, β=0.10 ·
β forks: {0.03, 0.10, 0.30}, 600 pairs · generation: T=0.7, top-p 0.9, max 256 new tokens ·
reward model yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback (8-bit) · dtype float16.

## Choices the manual leaves open (declare these as my own in the report)
1. **Prompt-length rule.** Pairs whose rendered prompt exceeds 512 tokens are excluded everywhere
   (training, held-out pairs, generation prompts), because the starter cannot encode a prompt longer
   than the 768-token sequence budget. 512 leaves ≥256 response tokens = the generation cap.
   Effect: standard train 1500→1394, held-out 300→272, balanced train 1500→1384 (461/461/462 per stratum),
   stratified eval 246→224 (70/79/75).
2. **β-fork data.** The first 600 *eligible* pairs of the standard training file (identical for all three β).
3. **Optimisation details not in the config.** Constant learning rate (no warmup or decay); last
   accumulation group of an epoch is partial and is averaged over its own size; fp16 base weights with
   fp32 LoRA weights and dynamic loss scaling; chosen and rejected run in one padded forward.
4. **Reference policy.** The same network with the LoRA adapter disabled (exactly the SFT model).
5. **Generation evaluation set.** One sampled response per eligible held-out prompt (272 prompts),
   same seed, longest-prompt-first batching, identical for every model.
6. **KL convention.** Primary: token-averaged sampled estimator over all generated tokens
   (= `common.metrics.sampled_kl` over the whole set). Also stored: mean per-sequence summed KL.
7. **Word-limit evaluation.** 10 fixed prompts × 4 sampled responses (40 per model), compliance via
   `common.metrics.word_limit_compliance`.
8. **Held-out loss across β.** Reported at the model's own β and at a common β=0.10.

## Hypotheses (write these yourself before the runs finish)
- RQ1 (β vs fit / KL / reward, monotonic?):
- RQ2 (dataset length bias vs. balanced training):
- RQ3 (preference/reward vs. quality disagreements):

## Budget note
Standard and length-balanced runs: 1 epoch (88 and 87 optimizer steps). β forks: 600 pairs (38 steps).
These budgets differ by design; say so next to every comparison.
