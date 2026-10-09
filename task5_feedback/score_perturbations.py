"""Task 5, step 2: score the 100-response Controlled Reward Diagnostic Set with both feedback mechanisms.

    python -m task5_feedback.score_perturbations --config configs/feedback.yaml   # GPU (judge), ~5 min

RLVR mechanism: the released exact verifier (exact_reward). RLAIF mechanism: the released PairwiseAIJudge.
Per problem the five variants are scored as one K=5 group with the released `group_rewards`
(round robin over all 10 pairs, (wins + 0.5 ties)/(K-1)) - exactly the direct-RLAIF reward - and every
pairwise preference is read from the judge (its own hash-based A/B orientation, its own cache).

Controlled pairs (diagnostically better response first):
  reasoning   clean_correct vs corrupt_reasoning_correct_final   final answer held correct  -> S_reason
  outcome     clean_correct vs good_reasoning_wrong_final        reasoning ~fixed, final changed -> S_outcome
  filler      clean_correct vs persuasive_filler_correct         both correct; filler adds only persuasion
  distractor  clean_correct vs gold_distractor_wrong_final       gold number appears, designated final wrong
  conflict    corrupt_reasoning_correct_final vs good_reasoning_wrong_final   no designated better side
For each pair and mechanism: better-response rate, tie rate, wrong-preference rate.
Order check (judge only): every controlled pair is also judged in BOTH presentation orders without the
released orientation swap (same rubric/decoding), to measure position consistency.
Outputs: <results_dir>/task5_feedback/diagnostics/{variant_scores,pairs,order_check}.jsonl + summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.experiment import run_metadata
from common.logging_utils import save_json
from common.rl import add_common_args, load_config
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
VARIANT_ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                 "persuasive_filler_correct", "gold_distractor_wrong_final"]
PAIRS = [("reasoning", "clean_correct", "corrupt_reasoning_correct_final", True),
         ("outcome", "clean_correct", "good_reasoning_wrong_final", True),
         ("filler", "clean_correct", "persuasive_filler_correct", True),
         ("distractor", "clean_correct", "gold_distractor_wrong_final", True),
         ("conflict", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final", False)]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


@torch.no_grad()
def raw_preference(judge: PairwiseAIJudge, problem: str, first: str, second: str) -> str:
    """The released judge call with a FIXED presentation order (first shown as A); no orientation swap."""
    text = PAIRWISE_RUBRIC.format(problem=problem, a=first, b=second)
    ids = judge.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors="pt",
                                              add_generation_prompt=True).to(next(judge.model.parameters()).device)
    out = judge.model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=judge.tokenizer.eos_token_id,
                               eos_token_id=judge.tokenizer.eos_token_id)
    decoded = judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().upper()
    m = re.search(r"\b(A|B|TIE)\b", decoded)
    return m.group(1) if m else "TIE"


def outcome_of(pref_first_vs_second, better_first: bool):
    if pref_first_vs_second == "TIE":
        return "tie"
    if not better_first:
        return "first" if pref_first_vs_second == "A" else "second"
    return "better" if pref_first_vs_second == "A" else "wrong"


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config or "configs/feedback.yaml", args.set)
    out = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"
    if (out / "summary.json").exists() and not args.overwrite:
        print(f"[diag] {out}/summary.json exists; skipping", flush=True)
        return
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    pids = sorted(groups, key=int)
    if cfg.get("diagnostic_limit"):
        pids = pids[: int(cfg["diagnostic_limit"])]
    print("Diagnostic problems:", len(pids))
    judge = PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json")
    t0 = time.perf_counter()
    scores, pairs, order = [], [], []
    for pid in pids:
        g = groups[pid]
        q, gold = g["clean_correct"]["question"], str(g["clean_correct"]["gold_final"])
        resp = {v: g[v]["response"] for v in VARIANT_ORDER}
        rlaif = judge.group_rewards(q, [resp[v] for v in VARIANT_ORDER])          # released K=5 RLAIF reward
        rlvr = {v: exact_reward(resp[v], gold) for v in VARIANT_ORDER}
        for v, ra in zip(VARIANT_ORDER, rlaif):
            scores.append({"problem_id": pid, "variant": v, "rlvr_reward": rlvr[v], "rlaif_group_reward": ra,
                           "expected_exact_reward": g[v]["expected_exact_reward"],
                           "verifier_matches_expected": rlvr[v] == float(g[v]["expected_exact_reward"]),
                           "response_chars": len(resp[v])})
        for name, x, y, better_first in PAIRS:
            ix, iy = VARIANT_ORDER.index(x), VARIANT_ORDER.index(y)
            # released compare(): the same call group_rewards made (lower variant index first) -> cached
            j = judge.compare(q, resp[VARIANT_ORDER[min(ix, iy)]], resp[VARIANT_ORDER[max(ix, iy)]])
            if ix > iy:
                j = {"A": "B", "B": "A", "TIE": "TIE"}[j]
            v = "A" if rlvr[x] > rlvr[y] else ("B" if rlvr[y] > rlvr[x] else "TIE")
            pairs.append({"problem_id": pid, "pair": name, "first": x, "second": y, "better_first": better_first,
                          "judge_pref": j, "verifier_pref": v,
                          "judge_outcome": outcome_of(j, better_first), "verifier_outcome": outcome_of(v, better_first)})
            fwd = raw_preference(judge, q, resp[x], resp[y])
            rev = raw_preference(judge, q, resp[y], resp[x])
            rev_as_fwd = {"A": "B", "B": "A", "TIE": "TIE"}[rev]
            order.append({"problem_id": pid, "pair": name, "first_shown_first": fwd, "second_shown_first": rev,
                          "consistent": fwd == rev_as_fwd, "position_A_chosen": [fwd == "A", rev == "A"],
                          "outcome_first_shown_first": outcome_of(fwd, better_first),
                          "outcome_second_shown_first": outcome_of(rev_as_fwd, better_first)})
        print(f"[diag] problem {pid} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "variant_scores.jsonl", scores)
    write_jsonl(out / "pairs.jsonl", pairs)
    write_jsonl(out / "order_check.jsonl", order)

    def rates(rows, key, better_first):
        o = [r[key] for r in rows]
        n = len(o)
        keys = ("better", "tie", "wrong") if better_first else ("first", "tie", "second")
        return {k: round(o.count(k) / n, 4) for k in keys} | {"n": n}

    pair_summary = {}
    for name, _, _, bf in PAIRS:
        rows = [r for r in pairs if r["pair"] == name]
        orows = [r for r in order if r["pair"] == name]
        pair_summary[name] = {"verifier": rates(rows, "verifier_outcome", bf), "judge": rates(rows, "judge_outcome", bf),
                              "judge_order_consistency": round(float(np.mean([r["consistent"] for r in orows])), 4),
                              "judge_fixed_order_first_shown_first": rates(orows, "outcome_first_shown_first", bf),
                              "judge_fixed_order_second_shown_first": rates(orows, "outcome_second_shown_first", bf)}
    S = {}
    for mech in ("verifier", "judge"):
        S[mech] = {"S_reason": pair_summary["reasoning"][mech].get("better", 0.0),
                   "S_outcome": pair_summary["outcome"][mech].get("better", 0.0),
                   "S_outcome_pooled_with_distractor": round(float(np.mean(
                       [r[f"{mech}_outcome"] == "better" for r in pairs if r["pair"] in ("outcome", "distractor")])), 4)}
    var_summary = {}
    for v in VARIANT_ORDER:
        rows = [r for r in scores if r["variant"] == v]
        var_summary[v] = {"rlvr_mean": round(float(np.mean([r["rlvr_reward"] for r in rows])), 4),
                          "rlaif_group_reward_mean": round(float(np.mean([r["rlaif_group_reward"] for r in rows])), 4),
                          "rlaif_group_reward_sd": round(float(np.std([r["rlaif_group_reward"] for r in rows], ddof=1)), 4)
                          if len(rows) > 1 else None,
                          "verifier_matches_expected": int(sum(r["verifier_matches_expected"] for r in rows)), "n": len(rows)}
    pos_a = [x for r in order for x in r["position_A_chosen"]]
    summ = {"pairs": pair_summary, "sensitivity": S, "variants": var_summary,
            "judge_position_A_rate_fixed_order": round(float(np.mean(pos_a)), 4),
            "judge_order_consistency_all_pairs": round(float(np.mean([r["consistent"] for r in order])), 4),
            "n_problems": len(pids), "wall_s": round(time.perf_counter() - t0, 1), "meta": run_metadata(cfg)}
    save_json(out / "summary.json", summ)
    print(json.dumps({"sensitivity": S, "pairs": {k: {"verifier": v["verifier"], "judge": v["judge"],
                                                      "order_consistency": v["judge_order_consistency"]}
                                                  for k, v in pair_summary.items()}}, indent=1))
    print("variants:", json.dumps(var_summary))


if __name__ == "__main__":
    main()
