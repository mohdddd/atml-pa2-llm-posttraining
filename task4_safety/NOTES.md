# Task 4 design record

Committed before the GPU run, so the ordering of decisions vs. results is visible in git history.

## Fixed by the course (not changed)
Policies: SFT = untouched Qwen2.5-1.5B-Instruct; DPO = Task 1 Step 1 standard adapter; PPO = Task 2 Step 1 standard
20-update adapter; GRPO = Task 3 Step 1 standard 20-update adapter (never a fork / beta-sweep model) ·
fixed XSTest CSV (450 prompts: 250 SAFE, 200 UNSAFE, 18 types) · `safety_max_new_tokens` 256 · fixed AI judge
Qwen2.5-3B-Instruct (4-bit), its prompt, parser and `judge_one` (greedy, 64 new tokens) · audit subset = starter
`fixed_audit_ids` (seed 6304, 30 SAFE + 30 UNSAFE prompts).

## Choices the manual leaves open (declare these as my own in the report)
1. **Decoding.** Greedy (`do_sample=False`), one response per policy and prompt, fixed XSTest order, batches of 32
   (identical batches for every policy). The model's own generation config still applies its repetition penalty
   (1.05) under greedy decoding — the same for every policy; recorded in `generation_<policy>.json`.
2. **Prompt fit.** All XSTest prompts are ≤48 rendered tokens; the run stops if any exceeded the 256-token prompt limit.
3. **Judge reuse.** The judge is deterministic, so a (prompt, response) text produced by several policies is judged
   once and the label reused (`judge_cache.jsonl`). This changes cost, not labels.
4. **Rates.** Safe-answer and over-refusal over all SAFE prompts; unsafe-compliance and justified-refusal over all
   UNSAFE prompts; ambiguous over all prompts (and per class); labels that presuppose the other prompt class
   (e.g. JUSTIFIED_REFUSAL on a SAFE prompt) reported separately as "class mismatch"; 95% Wilson intervals.
5. **Diagnostics (not metrics).** XSTest-style refusal-prefix string match, identical-to-SFT rate, label
   transitions vs SFT.
6. **Manual audit unit.** The 60 fixed prompts × 4 policies = 240 policy–prompt pairs. Identical response texts are
   labelled once (blinded sheet, shuffled, no policy name, no AI label; the XSTest class/type is shown). Agreement is
   reported both per unique text and per policy–prompt pair: accuracy, Cohen's κ, 5×5 confusion (manual rows × AI
   columns), ambiguous rates, and the four headline rates on the audited subset under manual vs AI labels.
7. **Secondary class-conditioned behaviour reading (added after the full run, before the manual audit).** The fixed
   judge never sees the XSTest prompt class, so a refusal of a SAFE prompt can be labelled JUSTIFIED_REFUSAL rather
   than OVER_REFUSAL (and an answer to an UNSAFE prompt SAFE_ANSWER rather than UNSAFE_COMPLIANCE). In addition to the
   required label-based rates (unchanged, primary), the judge label is collapsed to behaviour (refuse =
   JUSTIFIED_REFUSAL|OVER_REFUSAL, answer = SAFE_ANSWER|UNSAFE_COMPLIANCE) and read against the XSTest class:
   `safe_refusal_behaviour_rate`, `unsafe_answer_behaviour_rate` (Wilson CIs). The audit additionally reports
   refuse/answer agreement and a behaviour confusion table per prompt class. No labels or generations were changed.

## Hypotheses (write these yourself before you look at the results)
