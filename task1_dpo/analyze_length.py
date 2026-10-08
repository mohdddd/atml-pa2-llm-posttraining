"""Task 1 Step 3: length-confounding study.

Stages
  stats     (CPU) length structure of the preference files themselves: how often the preferred
            response is longer, by how much, and how that relates to the score gap
  train     one DPO run on the supplied length-balanced training file (same budget as standard:
            one epoch over the eligible pairs, beta from the config)
  evaluate  length-stratified held-out accuracy for standard and length-balanced DPO, plus the
            generation and word-limit evaluation of the length-balanced model
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import preference_responses, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_tokenizer
from task1_dpo.utils import add_common_args, eligible_pairs, load_task_config, run_module

FILES = {
    "standard_train": "dpo_standard_train",
    "standard_eval": "dpo_standard_eval",
    "length_balanced_train": "dpo_length_train",
    "length_stratified_eval": "dpo_length_eval",
}
MATCH_REL = 0.10   # approximate "length matched" band used only for describing the standard files


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def dataset_stats(cfg):
    tok = load_tokenizer(cfg["base_model"])
    out = {"tokenizer": cfg["base_model"], "matched_band_relative": MATCH_REL,
           "note": "Response lengths in tokens of the raw responses (before training-time truncation)."}
    for name, key in FILES.items():
        rows_all = read_jsonl(cfg["paths"][key])
        rows, _ = eligible_pairs(rows_all, tok, int(cfg["max_prompt_tokens"]))
        lc, lr, gap = [], [], []
        for r in rows:
            yc, yr = preference_responses(r)
            lc.append(len(tok(yc, add_special_tokens=False)["input_ids"]))
            lr.append(len(tok(yr, add_special_tokens=False)["input_ids"]))
            gap.append(float(r.get("score_chosen", np.nan)) - float(r.get("score_rejected", np.nan)))
        lc, lr, gap = np.array(lc, float), np.array(lr, float), np.array(gap, float)
        d = lc - lr
        rel = d / np.maximum(np.maximum(lc, lr), 1)
        st = {
            "n_file": len(rows_all), "n_eligible": len(rows),
            "chosen_tokens_mean": float(lc.mean()), "rejected_tokens_mean": float(lr.mean()),
            "length_diff_mean": float(d.mean()), "length_diff_median": float(np.median(d)),
            "frac_chosen_longer": float(np.mean(d > 0)), "frac_rejected_longer": float(np.mean(d < 0)),
            "frac_within_matched_band": float(np.mean(np.abs(rel) <= MATCH_REL)),
            "frac_chosen_longer_outside_band": float(np.mean(rel > MATCH_REL)),
            "frac_rejected_longer_outside_band": float(np.mean(rel < -MATCH_REL)),
            "spearman_lengthdiff_vs_scoregap": _spearman(d, gap),
            "frac_score_ties": float(np.mean(gap == 0)),
            "length_only_predictor_accuracy": float(np.mean(d > 0)),   # "always prefer the longer response"
        }
        if "length_stratum" in rows[0]:
            st["stratum_counts"] = {s: int(sum(r["length_stratum"] == s for r in rows))
                                    for s in ("preferred_longer", "length_matched", "rejected_longer")}
        out[name] = st
        print(f"[stats] {name}: n={len(rows)} chosen_longer={st['frac_chosen_longer']:.3f} "
              f"diff_mean={st['length_diff_mean']:.1f} spearman(diff,gap)={st['spearman_lengthdiff_vs_scoregap']:.3f}", flush=True)
    save_json(repo_path(cfg["results_dir"]) / "dataset_length_stats.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--stage", choices=["stats", "train", "evaluate", "all"], default="all")
    ap.add_argument("--overwrite", action="store_true")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_task_config(args.config, smoke=args.smoke, overrides=args.set)
    beta = float(cfg["beta"])
    max_ex = cfg.get("train_examples") if args.smoke else None
    if args.stage in ("stats", "all"):
        dataset_stats(cfg)
    ow = ["--overwrite"] if args.overwrite else []
    common = dict(smoke=args.smoke, overrides=args.set)
    if args.stage in ("train", "all"):
        extra = ["--max-examples", str(max_ex)] if max_ex else []
        run_module("task1_dpo.train", "--config", args.config, "--run-name", "length_balanced", "--dataset", "dpo_length_train",
                   "--output", cfg["length_output"], "--beta", str(beta), *extra, *ow, **common)
    if args.stage in ("evaluate", "all"):
        run_module("task1_dpo.evaluate", "--config", args.config, "--adapter", cfg["standard_output"], "--name", "standard",
                   "--beta", str(beta), "--parts", "stratified", *ow, **common)
        run_module("task1_dpo.evaluate", "--config", args.config, "--adapter", cfg["length_output"], "--name", "length_balanced",
                   "--beta", str(beta), "--parts", "heldout,stratified,generate,wordlimit", *ow, **common)


if __name__ == "__main__":
    main()
