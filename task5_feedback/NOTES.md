# Task 5 design record

Committed before the GPU run, so the ordering of decisions vs. results is visible in git history.

## Fixed by the course (not changed)
Policies: SFT = untouched Qwen2.5-1.5B-Instruct; RLVR = `checkpoints/rlvr_policy`; RLAIF = `checkpoints/rlaif_policy`
(supplied, frozen) · fixed GSM8K eval subset (300) · fixed SVAMP transfer file (first 100 of the challenge set,
unchanged) · `math_max_new_tokens` 512 · released exact verifier (`rlvr.exact_reward`, last `#### <number>` only) ·
released pairwise judge (`rlaif.PairwiseAIJudge`: rubric, greedy, 4 new tokens, hash-based A/B orientation balancing,
cache) · Controlled Reward Diagnostic Set (20 problems × 5 variants).

## Choices the manual leaves open (declare these as my own in the report)
1. **Decoding.** Greedy, one response per policy and problem, fixed order, batches of 32, identical for all policies
   (repetition penalty 1.05 from the model's generation config applies to every policy). Prompts ≤239 tokens (limit 512).
2. **Pairwise protocol.** Per problem: RLVR vs SFT, RLAIF vs SFT (the required "against SFT" win rates) and RLVR vs
   RLAIF. Win 1 / tie 0.5 / loss 0. The judge sees the problem text, never the gold answer (as in RLAIF training).
   Identical response pairs are still judged; how the judge labels them is reported (a free position/noise check).
3. **Verifier–judge agreement.** Verifier preference = which response has the higher exact reward (tie if equal).
   Reported: 3-way agreement; on verifier-decisive pairs the judge's agree / tie / opposite rates; on verifier-tied
   pairs the judge's tie rate.
4. **Failure types.** correct · wrong designated final · no `####` final (truncated at the cap) · no `####` final (other).
5. **Controlled pairs** (better response first): reasoning = clean vs corrupt-reasoning (S_reason); outcome = clean vs
   good-reasoning-wrong-final (S_outcome); filler = clean vs persuasive-filler (both correct — preferring the filler
   counts as wrong, a tie as tie); distractor = clean vs gold-distractor-wrong-final; conflict = corrupt-reasoning-
   correct-final vs good-reasoning-wrong-final (no designated better side; preference split only). A pooled S_outcome
   (outcome + distractor) is also stored. All five variants of a problem are additionally scored as one K=5 group
   with the released `group_rewards` (the direct-RLAIF reward) and with the verifier.
6. **Position check.** Every controlled pair is also judged in both presentation orders without the orientation swap
   (same rubric and decoding) → order-consistency and position-A rate. This is diagnostic; the released `compare`
   result is the primary judge preference.
7. **Judge output audit (added after the preflight, before the full run).** `PairwiseAIJudge.compare` additionally
   stores the raw decoded text (`judge_cache_raw.json`; judgements unchanged). A TIE whose output contains no
   A/B/TIE token (the released parser's fallback) is counted separately as an unparsed tie. A secondary win rate
   scores byte-identical response pairs as ties instead of by the judge (the preflight showed the judge picking a
   side on identical texts); the primary win rate stays the fixed-judge protocol.
8. **Paired accuracy differences vs SFT** with a 10 000-sample paired bootstrap (seed 6304).

## Hypotheses (write these yourself before you look at the results)
