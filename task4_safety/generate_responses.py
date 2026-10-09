"""Task 4, step 1: one deterministic response per fixed policy and XSTest prompt.

    python -m task4_safety.generate_responses --config configs/feedback.yaml --policy sft   # dpo | ppo | grpo | all

Writes <results_dir>/task4_safety/generated_<policy>.jsonl (fixed XSTest order) and
generation_<policy>.json (decoding settings, wall-clock, peak VRAM, run metadata). Each policy runs in
its own process when called by run_all.sh, so GPU memory is released between policies.
Skips a policy whose output exists unless --overwrite.
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.data import repo_path, write_jsonl
from common.experiment import run_metadata
from common.frozen_eval import greedy_generate
from common.logging_utils import save_json
from common.rl import add_common_args, load_config

POLICIES = ["sft", "dpo", "ppo", "grpo"]


def policy_specs(cfg):
    # Fixed by the manual: untouched SFT + the three STANDARD adapters (never a fork / beta-sweep model).
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    df = pd.read_csv(repo_path(cfg["paths"]["xstest"]))
    n = cfg.get("safety_limit_per_class")
    if n:  # preflight only
        df = pd.concat([df[df["benchmark_class"] == c].head(int(n)) for c in ("SAFE", "UNSAFE")])
    return df.reset_index(drop=True)


def out_dir(cfg):
    return repo_path(cfg["results_dir"]) / "task4_safety"


def generate_for_policy(cfg, policy_name: str):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    df = load_xstest(cfg)
    prompts = [[{"role": "user", "content": str(x)}] for x in df["prompt"].tolist()]
    gens, info = greedy_generate(cfg, specs[policy_name], prompts,
                                 max_prompt_length=int(cfg["safety_max_prompt_length"]),
                                 max_new_tokens=int(cfg["safety_max_new_tokens"]),
                                 batch_size=int(cfg["safety_batch_size"]), log_prefix=f"[gen {policy_name}]")
    records = []
    for (_, row), g in zip(df.iterrows(), gens):
        records.append({
            "xstest_id": int(row["xstest_id"]),
            "policy": policy_name,
            "prompt": str(row["prompt"]),
            "benchmark_class": str(row["benchmark_class"]),
            "type": str(row["type"]),
            "focus": str(row["focus"]),
            **g,
        })
    return records, info


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--policy", default="all", choices=POLICIES + ["all"])
    args = ap.parse_args()
    cfg = load_config(args.config or "configs/feedback.yaml", args.set)
    out = out_dir(cfg)
    for name in POLICIES if args.policy == "all" else [args.policy]:
        dst = out / f"generated_{name}.jsonl"
        if dst.exists() and not args.overwrite:
            print(f"[gen {name}] {dst} exists; skipping (--overwrite to redo)", flush=True)
            continue
        records, info = generate_for_policy(cfg, name)
        write_jsonl(dst, records)
        n_tok = [r["response_tokens"] for r in records]
        info.update({"policy": name, "n_responses": len(records),
                     "mean_response_tokens": sum(n_tok) / len(n_tok),
                     "truncation_rate": sum(r["truncated"] for r in records) / len(records)})
        save_json(out / f"generation_{name}.json", {**info, "meta": run_metadata(cfg)})
        print(f"[gen {name}] {len(records)} responses, mean len {info['mean_response_tokens']:.1f}, "
              f"trunc {info['truncation_rate']:.3f}, {info['wall_s']}s, peak {info['peak_vram_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
