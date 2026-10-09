"""Task 1 CPU diagnostics computed from saved per-item records (no model is loaded).

1. Paired reward-model difference vs SFT on the identical generation prompts (bootstrap 95% CI),
   and the fraction of generations that are character-identical to the SFT generation
   (all models sample with the same seed, so identical text means the policy did not change that sample).
2. Length-stratified preference accuracy with the summed implicit-reward margin (the manual's
   m_theta) and with a per-token (length-normalised) margin, from stratified_pairs.jsonl.
3. Optimizer steps skipped by fp16 dynamic loss scaling.
Writes report/tables/task1_paired_reward.csv, task1_strata_summed_vs_per_token.csv,
results/task1_dpo/diagnostics.json.  Runtime: CPU, seconds.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import read_jsonl, repo_path
from common.logging_utils import save_json

MODELS = ["standard", "beta_0.03", "beta_0.10", "beta_0.30", "length_balanced"]
STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def bootstrap_ci(x, n=5000, seed=6304):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/task1_dpo")
    args = ap.parse_args()
    R = repo_path(args.results)
    tab = repo_path("report/tables"); tab.mkdir(parents=True, exist_ok=True)
    out = {}

    sft = read_jsonl(R / "eval/sft/generations.jsonl")
    rows = []
    for m in MODELS:
        g = read_jsonl(R / f"eval/{m}/generations.jsonl")
        assert [a["id"] for a in sft] == [b["id"] for b in g], f"{m}: prompt order differs from SFT"
        d = np.array([b["reward"] - a["reward"] for a, b in zip(sft, g)])
        lo, hi = bootstrap_ci(d)
        rows.append({"model": m, "n": len(d), "reward_diff_vs_sft_mean": float(d.mean()), "ci95_low": lo, "ci95_high": hi,
                     "frac_identical_to_sft": float(np.mean([a["response"] == b["response"] for a, b in zip(sft, g)])),
                     "len_diff_vs_sft_mean": float(np.mean([b["response_tokens"] - a["response_tokens"] for a, b in zip(sft, g)]))})
    pd.DataFrame(rows).to_csv(tab / "task1_paired_reward.csv", index=False)
    out["paired_reward_vs_sft"] = rows

    srows = []
    for m in ("standard", "length_balanced"):
        P = read_jsonl(R / f"eval/{m}/stratified_pairs.jsonl")
        for s in STRATA + ["all"]:
            q = P if s == "all" else [p for p in P if p["length_stratum"] == s]
            summed = np.array([p["margin"] for p in q])
            per_tok = np.array([(p["pol_c"] - p["ref_c"]) / p["len_c"] - (p["pol_r"] - p["ref_r"]) / p["len_r"] for p in q])
            srows.append({"model": m, "stratum": s, "n": len(q),
                          "acc_summed_margin": float(np.mean(summed > 0)), "acc_per_token_margin": float(np.mean(per_tok > 0)),
                          "chosen_logratio_mean": float(np.mean([p["pol_c"] - p["ref_c"] for p in q])),
                          "rejected_logratio_mean": float(np.mean([p["pol_r"] - p["ref_r"] for p in q])),
                          "resp_len_diff_mean": float(np.mean([p["len_c"] - p["len_r"] for p in q]))})
    pd.DataFrame(srows).to_csv(tab / "task1_strata_summed_vs_per_token.csv", index=False)
    out["strata_summed_vs_per_token"] = srows

    out["fp16_skipped_steps"] = {m: [r["step"] for r in read_jsonl(R / f"runs/{m}/train_log.jsonl") if r["skipped_nonfinite"]]
                                 for m in MODELS}
    save_json(R / "diagnostics.json", out)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(srows).to_string(index=False))
    print("fp16 skipped steps:", out["fp16_skipped_steps"])


if __name__ == "__main__":
    main()
