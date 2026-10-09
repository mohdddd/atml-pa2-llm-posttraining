"""Task 5, steps 1 and 3: SFT / RLVR / RLAIF on the fixed GSM8K subset (in-domain) and SVAMP subset (transfer).

    python -m task5_feedback.evaluate_math --dataset gsm --stage generate --policy sft     # rlvr | rlaif | all  (GPU)
    python -m task5_feedback.evaluate_math --dataset gsm --stage judge                     # fixed pairwise judge (GPU)
    python -m task5_feedback.evaluate_math --dataset gsm --stage summary                   # metrics (CPU)
    (same with --dataset transfer)

generate: one greedy response per policy and problem (fixed order, fixed batches, cap math_max_new_tokens),
  scored with the released exact verifier -> <results_dir>/task5_feedback/<dataset>/generated_<policy>.jsonl
judge: the released PairwiseAIJudge (its rubric, its hash-based A/B orientation balancing, its cache at
  <results_dir>/task5_feedback/judge_cache.json) compares, per problem, RLVR vs SFT, RLAIF vs SFT and RLVR vs RLAIF.
  The judge sees the problem text only (never the gold answer) -> pairwise.jsonl
summary: exact accuracy, format compliance, length, failure types, AI pairwise win rate (win 1, tie 0.5, loss 0),
  judge ties, verifier-judge agreement -> summary.json
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.experiment import peak_vram_gib, reset_peak_vram, run_metadata
from common.frozen_eval import greedy_generate
from common.logging_utils import save_json
from common.models import load_policy, load_tokenizer
from common.rl import add_common_args, load_config
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

POLICIES = ["sft", "rlvr", "rlaif"]
COMPARISONS = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]   # (A = first argument, B = second)


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


# ---------------------------------------------------------------------------------------------
def load_rows(cfg, dataset):
    if dataset == "transfer" and not repo_path(cfg["paths"]["math_transfer_eval"]).exists():
        from scripts.prepare_transfer_eval import materialize_transfer_eval
        materialize_transfer_eval(repo_path(cfg["paths"]["math_transfer_eval"]))
    rows = read_jsonl(dataset_path(cfg, dataset))
    n = cfg.get("math_limit")
    return rows[: int(n)] if n else rows


def item_id(row):
    return str(row.get("prompt_id") or row.get("source_index"))


def ddir(cfg, dataset):
    return repo_path(cfg["results_dir"]) / "task5_feedback" / dataset


def judge_cache_path(cfg):
    return repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json"


def stage_generate(cfg, dataset, names, overwrite):
    rows = load_rows(cfg, dataset)
    out = ddir(cfg, dataset)
    specs = policy_specs(cfg)
    for name in names:
        dst = out / f"generated_{name}.jsonl"
        if dst.exists() and not overwrite:
            print(f"[gen {dataset}/{name}] exists; skipping", flush=True)
            continue
        gens, info = greedy_generate(cfg, specs[name], [r["messages"] for r in rows],
                                     max_prompt_length=int(cfg["math_max_prompt_length"]),
                                     max_new_tokens=int(cfg["math_max_new_tokens"]),
                                     batch_size=int(cfg["math_batch_size"]), log_prefix=f"[gen {dataset}/{name}]")
        recs = []
        for r, g in zip(rows, gens):
            pred = extract_designated_final(g["response"])
            recs.append({"item_id": item_id(r), "policy": name, "question": r["question"], "gold_final": str(r["gold_final"]),
                         **g, "pred_final": pred, "format_ok": pred is not None,
                         "exact": exact_reward(g["response"], str(r["gold_final"]))})
        write_jsonl(dst, recs)
        acc = float(np.mean([x["exact"] for x in recs]))
        save_json(out / f"generation_{name}.json", {**info, "policy": name, "dataset": dataset, "accuracy": acc,
                                                    "meta": run_metadata(cfg)})
        print(f"[gen {dataset}/{name}] acc={acc:.3f} format={np.mean([x['format_ok'] for x in recs]):.3f} "
              f"len={np.mean([x['response_tokens'] for x in recs]):.1f} {info['wall_s']}s peak {info['peak_vram_gib']} GiB",
              flush=True)


def stage_judge(cfg, dataset, overwrite):
    out = ddir(cfg, dataset)
    dst = out / "pairwise.jsonl"
    if dst.exists() and not overwrite:
        print(f"[judge {dataset}] exists; skipping", flush=True)
        return
    gens = {p: read_jsonl(out / f"generated_{p}.jsonl") for p in POLICIES}
    judge = PairwiseAIJudge(cfg, judge_cache_path(cfg))
    n0 = len(judge.cache)
    reset_peak_vram()
    t0 = time.perf_counter()
    recs = []
    for a, b in COMPARISONS:
        for i, (ra, rb) in enumerate(zip(gens[a], gens[b])):
            assert ra["item_id"] == rb["item_id"]
            pref = judge.compare(ra["question"], ra["response"], rb["response"])
            recs.append({"item_id": ra["item_id"], "a": a, "b": b, "judge": pref,
                         "identical_text": ra["response"] == rb["response"],
                         "exact_a": ra["exact"], "exact_b": rb["exact"]})
            if (i + 1) % 50 == 0:
                print(f"[judge {dataset}] {a} vs {b}: {i + 1}/{len(gens[a])} ({time.perf_counter() - t0:.0f}s)", flush=True)
    write_jsonl(dst, recs)
    save_json(out / "judge_run.json", {"n_comparisons": len(recs), "new_judge_calls": len(judge.cache) - n0,
                                       "wall_s": round(time.perf_counter() - t0, 1), "peak_vram_gib": peak_vram_gib(),
                                       "judge_model": cfg["ai_judge_model"], "meta": run_metadata(cfg)})
    print(f"[judge {dataset}] {len(recs)} comparisons, {len(judge.cache) - n0} new judge calls, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------
def verifier_pref(ea, eb):
    return "A" if ea > eb else ("B" if eb > ea else "TIE")


def failure_type(r):
    if r["exact"] == 1.0:
        return "correct"
    if r["format_ok"]:
        return "wrong_final"
    return "no_final_truncated" if r["truncated"] else "no_final_format"


def pairwise_stats(recs):
    j = np.array([r["judge"] for r in recs])
    v = np.array([verifier_pref(r["exact_a"], r["exact_b"]) for r in recs])
    dec = v != "TIE"
    n = len(recs)
    return {
        "n": n,
        "win_rate_a": round(float(np.mean([{"A": 1.0, "TIE": 0.5, "B": 0.0}[x] for x in j])), 4) if n else None,
        "wins_a": int((j == "A").sum()), "ties": int((j == "TIE").sum()), "losses_a": int((j == "B").sum()),
        "identical_text": int(sum(r["identical_text"] for r in recs)),
        "judge_on_identical_text": {k: int(sum(1 for r in recs if r["identical_text"] and r["judge"] == k)) for k in ("A", "B", "TIE")},
        "verifier_judge_agreement_3way": round(float((v == j).mean()), 4) if n else None,
        "verifier_decisive_n": int(dec.sum()),
        "judge_agrees_when_verifier_decisive": round(float((j[dec] == v[dec]).mean()), 4) if dec.any() else None,
        "judge_ties_when_verifier_decisive": round(float((j[dec] == "TIE").mean()), 4) if dec.any() else None,
        "judge_opposite_when_verifier_decisive": round(float(((j[dec] != v[dec]) & (j[dec] != "TIE")).mean()), 4) if dec.any() else None,
        "verifier_tied_n": int((~dec).sum()),
        "judge_tie_when_verifier_tied": round(float((j[~dec] == "TIE").mean()), 4) if (~dec).any() else None,
    }


def paired_bootstrap(a, b, n_boot=10000, seed=6304):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(1)
    return round(float(d.mean()), 4), round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)


def stage_summary(cfg, dataset):
    out = ddir(cfg, dataset)
    gens = {p: read_jsonl(out / f"generated_{p}.jsonl") for p in POLICIES}
    pw = read_jsonl(out / "pairwise.jsonl")
    pol = {}
    for p in POLICIES:
        g = gens[p]
        lens = np.array([r["response_tokens"] for r in g])
        ex = [r["exact"] for r in g]
        ft = {k: 0 for k in ("correct", "wrong_final", "no_final_format", "no_final_truncated")}
        for r in g:
            ft[failure_type(r)] += 1
        pol[p] = {"n": len(g), "accuracy": round(float(np.mean(ex)), 4),
                  "accuracy_sem": round(float(np.std(ex, ddof=1) / np.sqrt(len(ex))), 4),
                  "format_compliance": round(float(np.mean([r["format_ok"] for r in g])), 4),
                  "len_mean": round(float(lens.mean()), 1), "len_sd": round(float(lens.std(ddof=1)), 1),
                  "len_median": float(np.median(lens)), "len_iqr": float(np.quantile(lens, .75) - np.quantile(lens, .25)),
                  "truncation_rate": round(float(np.mean([r["truncated"] for r in g])), 4), "failure_types": ft}
        if p != "sft":
            pol[p]["accuracy_diff_vs_sft"] = paired_bootstrap(ex, [r["exact"] for r in gens["sft"]])
    comps = {f"{a}_vs_{b}": pairwise_stats([r for r in pw if r["a"] == a and r["b"] == b]) for a, b in COMPARISONS}
    summ = {"dataset": dataset, "policies": pol, "pairwise": comps,
            "note": "win_rate_a = AI pairwise win rate of the first policy against the second (win 1, tie 0.5, loss 0); "
                    "SFT against itself is 0.5 by definition and is not judged.", "meta": run_metadata(cfg)}
    save_json(out / "summary.json", summ)
    for p, s in pol.items():
        print(f"[{dataset}] {p:5s} acc={s['accuracy']:.3f} format={s['format_compliance']:.3f} "
              f"len={s['len_mean']}±{s['len_sd']} trunc={s['truncation_rate']:.3f} {s['failure_types']}")
    for k, s in comps.items():
        print(f"[{dataset}] {k}: win={s['win_rate_a']} (W{s['wins_a']}/T{s['ties']}/L{s['losses_a']}) "
              f"identical={s['identical_text']} agree3={s['verifier_judge_agreement_3way']} "
              f"decisive n={s['verifier_decisive_n']} agree={s['judge_agrees_when_verifier_decisive']}")


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--stage", choices=["generate", "judge", "summary"], required=True)
    ap.add_argument("--policy", default="all", choices=POLICIES + ["all"])
    args = ap.parse_args()
    cfg = load_config(args.config or "configs/feedback.yaml", args.set)
    if args.stage == "generate":
        stage_generate(cfg, args.dataset, POLICIES if args.policy == "all" else [args.policy], args.overwrite)
    elif args.stage == "judge":
        stage_judge(cfg, args.dataset, args.overwrite)
    else:
        stage_summary(cfg, args.dataset)


if __name__ == "__main__":
    main()
