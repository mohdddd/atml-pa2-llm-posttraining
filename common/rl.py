"""Helpers shared by the online-RL tasks (Task 2 PPO, Task 3 GRPO). Student addition.

Contents
- load_config        : YAML (+ base config) with `key=value` / `a.b=value` overrides recorded in the config
- eligible_prompts   : the fixed prompt-length rule (rendered prompt <= max_prompt_length tokens)
- update_schedule    : the fixed, seeded order of training prompts consumed by every continuation/fork
- disable_dropout    : dropout off during RL updates, so the PPO/GRPO ratio is exactly 1 before the first step
- evaluate_policy    : the common held-out protocol (one sampled response per eligible eval prompt,
                       same seed and batch composition for every model) -> per-prompt records
- summarize_eval     : aggregate metrics with the conventions of the manual
- run_module         : run `python -m module ...` in a fresh process (GPU memory released on exit)
"""
from __future__ import annotations

import random
import subprocess
import sys

import numpy as np
import torch
import yaml
from peft import PeftModel

from common.data import load_yaml, read_jsonl
from common.generation import batch_generate, response_token_logprobs_lean, score_reward_pairs
from common.models import reference_mode


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load a task config. Each override is `key=value` or `a.b=value` (value parsed as YAML).
    Overrides are for preflight/timing runs and plumbing tests; all are kept in cfg['cli_overrides']."""
    cfg = load_yaml(path)
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
        cfg.setdefault("cli_overrides", []).append(item)
    return cfg


def add_common_args(ap):
    ap.add_argument("--config", required=False)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides (preflight/tests only)")
    ap.add_argument("--overwrite", action="store_true", help="redo work whose outputs already exist")


def override_args(cfg: dict) -> list[str]:
    """Forward the overrides of this process to a child process."""
    return ["--set", *cfg["cli_overrides"]] if cfg.get("cli_overrides") else []


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
def prompt_tokens(tokenizer, messages) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def eligible_prompts(rows: list[dict], tokenizer, max_prompt_length: int):
    """Keep prompts whose rendered chat prompt fits max_prompt_length tokens.

    The starter generator truncates longer prompts from the right, which removes the end of the
    user turn and the assistant header. Such prompts are excluded everywhere (training, forks and
    held-out evaluation), for every condition. Returns (kept, dropped_records)."""
    kept, dropped = [], []
    for r in rows:
        n = prompt_tokens(tokenizer, r["messages"])
        if n <= max_prompt_length:
            kept.append(r)
        else:
            dropped.append({"prompt_id": r["prompt_id"], "source_index": r.get("source_index"), "prompt_tokens": n})
    return kept, dropped


def update_schedule(cfg: dict, tokenizer, n_updates: int, per_update: int):
    """The fixed sequence of training prompts: eligible train prompts shuffled once with the
    assignment seed. Update u uses positions [u*per_update, (u+1)*per_update). Every continuation and
    every fork therefore sees the same prompts in the same order (a fork = a prefix of the schedule)."""
    rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    kept, dropped = eligible_prompts(rows, tokenizer, int(cfg["max_prompt_length"]))
    order = list(range(len(kept)))
    random.Random(int(cfg["seed"])).shuffle(order)
    need = n_updates * per_update
    if need > len(order):
        raise ValueError(f"schedule needs {need} prompts, only {len(order)} eligible")
    sched = [kept[i] for i in order[:need]]
    return [sched[u * per_update:(u + 1) * per_update] for u in range(n_updates)], {
        "n_train_file": len(rows), "n_eligible": len(kept), "n_dropped": len(dropped)}


def eval_items(cfg: dict, tokenizer) -> tuple[list[dict], dict]:
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    kept, dropped = eligible_prompts(rows, tokenizer, int(cfg["max_prompt_length"]))
    n = cfg.get("eval_prompts")
    if n:
        kept = kept[: int(n)]
    items = [{"id": r["prompt_id"], "source_index": r.get("source_index"), "messages": r["messages"]} for r in kept]
    return items, {"n_eval_file": len(rows), "n_eligible": len(kept), "dropped": dropped}


# --------------------------------------------------------------------------------------
# Model utilities
# --------------------------------------------------------------------------------------
def disable_dropout(model) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) and m.p > 0:
            m.p = 0.0
            n += 1
    return n


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum())
    return total ** 0.5


# --------------------------------------------------------------------------------------
# Held-out evaluation (common to Task 2 and Task 3)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_policy(model, tokenizer, items, cfg, rm=None, rm_tok=None, max_new_tokens: int = 768,
                    batch_size: int = 32, logp_batch: int = 2, seed: int | None = None):
    """One sampled response per item (cfg['generation'] decoding), then per response:
    length, EOS/truncation, reward-model score, sampled-response KL to the reference
    (sum and token count of log pi - log ref over generated tokens) and mean token entropy
    (sum over valid tokens). Items are processed longest-prompt first, so batch composition is
    identical for every model; the generator seed is reset once before the first batch."""
    gen = cfg["generation"]
    is_peft = isinstance(model, PeftModel)
    order = sorted(range(len(items)), key=lambda i: -prompt_tokens(tokenizer, items[i]["messages"]))
    if seed is not None:
        torch.manual_seed(seed)
    out = {}
    for b in range(0, len(order), batch_size):
        idx = order[b: b + batch_size]
        prompts = [items[i]["messages"] for i in idx]
        g = batch_generate(model, tokenizer, prompts, int(cfg["max_prompt_length"]), int(max_new_tokens),
                           temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
        seq, pw, rids, rmask = g["sequences"], g["prompt_width"], g["response_ids"], g["response_mask"]
        attn = g["attention_mask"].clone()
        attn[:, pw:] = rmask.long()
        kl_sum, ent_sum, n_tok = [], [], []
        for s in range(0, len(idx), logp_batch):
            sl = slice(s, s + logp_batch)
            pol, ent = response_token_logprobs_lean(model, seq[sl], attn[sl], pw, rids[sl], with_entropy=True)
            if is_peft:
                with reference_mode(model):
                    ref, _ = response_token_logprobs_lean(model, seq[sl], attn[sl], pw, rids[sl])
            else:
                ref = pol
            m = rmask[sl]
            kl_sum += ((pol - ref) * m).sum(-1).tolist()
            ent_sum += (ent * m).sum(-1).tolist()
            n_tok += m.sum(-1).tolist()
        rewards = [None] * len(idx)
        if rm is not None:
            rewards = []
            for s in range(0, len(idx), 8):
                rewards += score_reward_pairs(rm, rm_tok, prompts[s:s + 8], g["responses"][s:s + 8],
                                              max_length=int(cfg.get("reward_max_length", 1280))).tolist()
        for j, i in enumerate(idx):
            out[i] = {
                "id": items[i]["id"],
                "source_index": items[i].get("source_index"),
                "response": g["responses"][j],
                "response_tokens": int(g["response_lengths"][j]),
                "terminated_with_eos": bool(g["terminated_with_eos"][j]),
                "truncated": bool(g["truncated"][j]),
                "reward": None if rewards[j] is None else float(rewards[j]),
                "kl_seq_sum": float(kl_sum[j]),
                "entropy_seq_sum": float(ent_sum[j]),
                "n_tokens": int(n_tok[j]),
            }
        print(f"  eval batch {b // batch_size + 1}/{(len(order) + batch_size - 1) // batch_size} done", flush=True)
    return [out[i] for i in range(len(items))]


def summarize_eval(recs: list[dict], missing_eos_penalty: float | None = None) -> dict:
    lens = np.array([r["response_tokens"] for r in recs], float)
    kl = np.array([r["kl_seq_sum"] for r in recs]); ent = np.array([r["entropy_seq_sum"] for r in recs])
    nt = np.array([r["n_tokens"] for r in recs], float)
    s = {
        "n": len(recs),
        "length_mean": float(lens.mean()), "length_std": float(lens.std(ddof=1)) if len(lens) > 1 else 0.0,
        "length_median": float(np.median(lens)),
        "length_iqr": float(np.percentile(lens, 75) - np.percentile(lens, 25)),
        "truncation_rate": float(np.mean([r["truncated"] for r in recs])),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in recs])),
        # token-averaged sampled-response estimator over all generated tokens (== common.metrics.sampled_kl)
        "kl_token_mean": float(kl.sum() / max(nt.sum(), 1)),
        "kl_seq_mean": float(kl.mean()),
        "entropy_token_mean": float(ent.sum() / max(nt.sum(), 1)),
    }
    rw = np.array([r["reward"] for r in recs if r["reward"] is not None], float)
    if len(rw):
        s.update({"reward_mean": float(rw.mean()), "reward_std": float(rw.std(ddof=1)) if len(rw) > 1 else 0.0,
                  "reward_sem": float(rw.std(ddof=1) / np.sqrt(len(rw))) if len(rw) > 1 else 0.0})
        if missing_eos_penalty:
            eff = np.array([r["reward"] - (0.0 if r["terminated_with_eos"] else missing_eos_penalty) for r in recs], float)
            s["effective_reward_mean"] = float(eff.mean())
    return s


def evaluate_adapter(cfg: dict, adapter: str | None, name: str, overwrite: bool = False) -> None:
    """Held-out evaluation of one frozen policy -> <results_dir>/eval/<name>/{generations.jsonl, summary.json}.
    adapter None / 'none' = the untouched SFT policy (KL is then 0 by construction)."""
    import time

    from common.data import repo_path, write_jsonl
    from common.experiment import peak_vram_gib, reset_peak_vram, run_metadata
    from common.logging_utils import save_json
    from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer

    out = repo_path(cfg["results_dir"]) / "eval" / name
    if (out / "summary.json").exists() and not overwrite:
        print(f"[eval {name}] found {out}/summary.json; skipping (pass --overwrite to redo).", flush=True)
        return
    adapter = None if adapter in (None, "", "none", "None") else adapter
    tok = load_tokenizer(cfg["base_model"])
    items, info = eval_items(cfg, tok)
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = load_reward_model(cfg)
    max_new = int(cfg["eval_max_response_length"])
    print(f"[eval {name}] adapter={adapter} prompts={len(items)} max_new_tokens={max_new}", flush=True)
    reset_peak_vram()
    t0 = time.perf_counter()
    recs = evaluate_policy(model, tok, items, cfg, rm, rm_tok, max_new_tokens=max_new,
                           batch_size=int(cfg.get("eval_batch_size", 32)), seed=int(cfg["seed"]))
    summ = summarize_eval(recs, cfg.get("missing_eos_penalty"))
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "generations.jsonl", recs)
    save_json(out / "summary.json", {"metrics": summ, "adapter": adapter, "name": name, "eval_set": info,
                                     "max_new_tokens": max_new, "eval_wall_s": round(time.perf_counter() - t0, 1),
                                     "meta": run_metadata(cfg, peak_vram_gib=peak_vram_gib())})
    print(f"[eval {name}] reward={summ.get('reward_mean', float('nan')):.3f}±{summ.get('reward_sem', 0):.3f} "
          f"KL={summ['kl_token_mean']:.5f} H={summ['entropy_token_mean']:.3f} len={summ['length_mean']:.1f}±{summ['length_std']:.1f} "
          f"trunc={summ['truncation_rate']:.3f}", flush=True)
    del model, rm
    clear_gpu()


def evaluate_cli(default_config: str) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the SFT policy")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config or default_config, args.set)
    evaluate_adapter(cfg, args.adapter, args.name, args.overwrite)


# --------------------------------------------------------------------------------------
# Processes
# --------------------------------------------------------------------------------------
def run_module(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)
