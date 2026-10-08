"""Task 1 evaluation of one policy (SFT base or a DPO adapter) under the common protocol.

Parts (each writes per-item records + a summary entry; results/task1_dpo/eval/<name>/):
  heldout    held-out DPO pairs: margin m_theta, preference accuracy, DPO loss
  generate   one sampled response per eligible held-out prompt: length, reference KL, RM score
  wordlimit  the 10 fixed word-limit prompts x word_limit_samples samples: compliance, length
  stratified length-stratified held-out pairs: accuracy per stratum
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common.data import read_jsonl, repo_path, write_jsonl
from common.experiment import peak_vram_gib, reset_peak_vram, run_metadata
from common.logging_utils import load_json, save_json
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from task1_dpo.utils import (
    add_common_args,
    load_task_config,
    eligible_pairs,
    eval_generation_items,
    generate_and_score,
    pair_logps,
    preference_metrics,
    row_id,
    summarize_generations,
)

ALL_PARTS = ("heldout", "generate", "wordlimit", "stratified")


def _out_dir(cfg, name):
    d = repo_path(cfg["results_dir"]) / "eval" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _update_summary(path, key, value, meta):
    s = load_json(path) if path.exists() else {}
    s[key] = value
    s.setdefault("meta", {})[key] = meta
    save_json(path, s)


def _heldout_rows(cfg, tok, key, n=None):
    rows, dropped = eligible_pairs(read_jsonl(cfg["paths"][key]), tok, int(cfg["max_prompt_tokens"]))
    return (rows[: int(n)] if n else rows), dropped


def eval_heldout(cfg, model, tok, beta, out):
    rows, dropped = _heldout_rows(cfg, tok, "dpo_standard_eval", cfg.get("eval_pairs"))
    lp = pair_logps(model, tok, rows, int(cfg["max_sequence_length"]), int(cfg.get("eval_pair_batch", 4)))
    m = preference_metrics(lp, beta)
    m["n_dropped_by_prompt_rule"] = len(dropped)
    recs = [{"id": row_id(r), "prompt_id": r["prompt_id"], **{k: float(lp[k][i]) for k in lp},
             "margin": float((lp["pol_c"][i] - lp["ref_c"][i]) - (lp["pol_r"][i] - lp["ref_r"][i])),
             "score_chosen": r.get("score_chosen"), "score_rejected": r.get("score_rejected")} for i, r in enumerate(rows)]
    write_jsonl(out / "heldout_pairs.jsonl", recs)
    return m


def eval_stratified(cfg, model, tok, beta, out):
    rows, dropped = _heldout_rows(cfg, tok, "dpo_length_eval", cfg.get("eval_pairs"))
    lp = pair_logps(model, tok, rows, int(cfg["max_sequence_length"]), int(cfg.get("eval_pair_batch", 4)))
    margin = (lp["pol_c"] - lp["ref_c"]) - (lp["pol_r"] - lp["ref_r"])
    strata = np.array([r["length_stratum"] for r in rows])
    res = {"all": preference_metrics(lp, beta), "n_dropped_by_prompt_rule": len(dropped)}
    for s in ("preferred_longer", "length_matched", "rejected_longer"):
        sel = strata == s
        sub = {k: v[sel] for k, v in lp.items()}
        res[s] = preference_metrics(sub, beta) if sel.any() else {"n": 0}
    recs = [{"id": row_id(r), "length_stratum": r["length_stratum"], "length_difference": r.get("length_difference"),
             **{k: float(lp[k][i]) for k in lp}, "margin": float(margin[i])} for i, r in enumerate(rows)]
    write_jsonl(out / "stratified_pairs.jsonl", recs)
    return res


def eval_generate(cfg, model, tok, rm, rm_tok, out):
    items = eval_generation_items(cfg, tok)
    recs = generate_and_score(model, tok, items, cfg, rm, rm_tok, batch_size=int(cfg.get("eval_gen_batch", 32)),
                              seed=int(cfg["seed"]))
    write_jsonl(out / "generations.jsonl", recs)
    return summarize_generations(recs)


def eval_wordlimit(cfg, model, tok, out):
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    k = int(cfg.get("word_limit_samples", 4))
    items = [{"id": f"{r['prompt_id']}#s{j}", "prompt_id": r["prompt_id"], "messages": r["messages"]} for r in rows for j in range(k)]
    recs = generate_and_score(model, tok, items, cfg, None, None, batch_size=int(cfg.get("eval_gen_batch", 32)),
                              seed=int(cfg["seed"]))
    for it, rec in zip(items, recs):
        prompt = it["messages"][-1]["content"]
        rec.update({"prompt_id": it["prompt_id"], "word_limit": parse_word_limit(prompt),
                    "words": word_count(rec["response"]), "compliant": word_limit_compliance(prompt, rec["response"])})
    write_jsonl(out / "word_limit.jsonl", recs)
    comp = [r["compliant"] for r in recs if r["compliant"] is not None]
    words = np.array([r["words"] for r in recs], dtype=float)
    per_prompt = {}
    for r in recs:
        per_prompt.setdefault(r["prompt_id"], []).append(r["compliant"])
    return {"n_responses": len(recs), "samples_per_prompt": k, "compliance_rate": float(np.mean(comp)),
            "words_mean": float(words.mean()), "words_std": float(words.std(ddof=1)) if len(words) > 1 else 0.0,
            "prompts_all_compliant": int(sum(all(v) for v in per_prompt.values())),
            "tokens_mean": float(np.mean([r["response_tokens"] for r in recs]))}


def evaluate_policy(cfg, adapter: str | None, name: str, beta: float, parts=ALL_PARTS, overwrite=False):
    out = _out_dir(cfg, name)
    summary_path = out / "summary.json"
    done = load_json(summary_path) if summary_path.exists() else {}
    todo = [p for p in parts if overwrite or p not in done]
    if not todo:
        print(f"[eval {name}] all requested parts already done; skipping.")
        return
    tok = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm = rm_tok = None
    for part in todo:
        reset_peak_vram()
        t0 = time.perf_counter()
        if part == "heldout":
            res = eval_heldout(cfg, model, tok, beta, out)
        elif part == "stratified":
            res = eval_stratified(cfg, model, tok, beta, out)
        elif part == "generate":
            rm, rm_tok = load_reward_model(cfg)
            res = eval_generate(cfg, model, tok, rm, rm_tok, out)
            del rm, rm_tok                      # free the reward model before the next part
            rm = rm_tok = None
            clear_gpu()
        elif part == "wordlimit":
            res = eval_wordlimit(cfg, model, tok, out)
        else:
            raise ValueError(part)
        meta = run_metadata({k: cfg[k] for k in ("seed", "base_model", "reward_model", "generation", "max_generation_tokens",
                                                 "max_sequence_length", "max_prompt_tokens", "dtype", "cli_overrides", "smoke") if k in cfg},
                            adapter=adapter, beta=beta, wall_clock_s=round(time.perf_counter() - t0, 1), peak_vram_gib=peak_vram_gib())
        _update_summary(summary_path, part, res, meta)
        short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, dict)}
        print(f"[eval {name}] {part}: {short} ({meta['wall_clock_s']}s)", flush=True)
    del model
    clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the untouched SFT policy")
    ap.add_argument("--name", required=True)
    ap.add_argument("--beta", type=float, help="beta the policy was trained with (for the held-out DPO loss)")
    ap.add_argument("--parts", default=",".join(ALL_PARTS))
    ap.add_argument("--overwrite", action="store_true")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_task_config(args.config, smoke=args.smoke, overrides=args.set)
    adapter = None if args.adapter.lower() == "none" else args.adapter
    beta = args.beta if args.beta is not None else float(cfg["beta"])
    evaluate_policy(cfg, adapter, args.name, beta, [p for p in args.parts.split(",") if p], args.overwrite)


if __name__ == "__main__":
    main()
