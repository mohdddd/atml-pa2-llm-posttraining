"""Task 1 helpers shared by train / evaluate / ablate_beta / analyze_length (student addition).

Contents
- load_task_config: YAML + optional smoke overrides
- eligible_pairs: the fixed prompt-length rule applied to every DPO file
- pair_logps: sequence log-probs of chosen/rejected under policy and reference (held-out metrics)
- generate_and_score: sampled generations with length, reference KL and reward-model score
"""
from __future__ import annotations

import numpy as np
import torch
from peft import PeftModel

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
)
from common.experiment import apply_smoke
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs_lean, score_reward_pairs
from common.models import reference_mode

SMOKE_EXTRA = {
    "short_ablation_examples": 6,
    "max_sequence_length": 192,
    "max_prompt_tokens": 96,
    "max_generation_tokens": 24,
    "grad_accum_steps": 2,
    "eval_pairs": 6,
    "eval_generation_prompts": 6,
    "word_limit_samples": 1,
    "train_examples": 8,
}


def load_task_config(path: str, smoke: bool = False, overrides: list[str] | None = None) -> dict:
    """YAML config, optional smoke overrides, then optional `key=value` overrides (values parsed as YAML).
    Overrides are meant for preflight/timing runs; every override is recorded in run metadata."""
    import yaml

    cfg = load_yaml(path)
    if smoke:
        cfg = apply_smoke(cfg, SMOKE_EXTRA)
    for item in overrides or []:
        key, _, val = item.partition("=")
        cfg[key] = yaml.safe_load(val)
        cfg.setdefault("cli_overrides", []).append(item)
    return cfg


def add_common_args(ap):
    ap.add_argument("--smoke", action="store_true", help="CPU plumbing test with the 0.5B model")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides (preflight runs only)")


# --------------------------------------------------------------------------------------
# Data eligibility
# --------------------------------------------------------------------------------------
def prompt_token_count(tokenizer, row: dict) -> int:
    ids = tokenizer.apply_chat_template(prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True)
    return len(ids)


def eligible_pairs(rows: list[dict], tokenizer, max_prompt_tokens: int):
    """Keep pairs whose rendered prompt has <= max_prompt_tokens tokens.

    With max_sequence_length=768 and max_prompt_tokens=512, every kept pair has at least 256
    response tokens of budget, i.e. at least the generation cap. The same rule is used for
    training, held-out pairs and generation prompts, for every condition.
    Returns (kept_rows, dropped_records).
    """
    kept, dropped = [], []
    for row in rows:
        n = prompt_token_count(tokenizer, row)
        if n <= max_prompt_tokens:
            kept.append(row)
        else:
            dropped.append({"prompt_id": row.get("prompt_id"), "source_index": row.get("source_index"), "prompt_tokens": n})
    return kept, dropped


def row_id(row: dict) -> str:
    return f"{row.get('source_split', '')}:{row.get('source_index', '')}"


# --------------------------------------------------------------------------------------
# Held-out preference metrics
# --------------------------------------------------------------------------------------
def encode_pairs(tokenizer, rows, max_length):
    chosen, rejected = [], []
    for row in rows:
        prompt = prompt_messages_from_preference(row)
        yc, yr = preference_responses(row)
        chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
        rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
    return chosen, rejected


@torch.no_grad()
def pair_logps(model, tokenizer, rows, max_length: int, batch_pairs: int = 4):
    """Summed response-token log-probs for chosen/rejected under the policy and the frozen reference.

    The reference is the same network with the LoRA adapter disabled (for the untouched SFT model,
    policy == reference). Returns a dict of numpy arrays aligned with `rows`.
    """
    device = next(model.parameters()).device
    out = {k: [] for k in ("pol_c", "pol_r", "ref_c", "ref_r", "len_c", "len_r")}
    is_peft = isinstance(model, PeftModel)
    model.eval()
    for i in range(0, len(rows), batch_pairs):
        chunk = rows[i : i + batch_pairs]
        ch, rj = encode_pairs(tokenizer, chunk, max_length)
        batch = pad_batch(tokenizer, ch + rj)
        batch = {k: v.to(device) for k, v in batch.items()}
        n = len(chunk)
        pol, _, mask = response_sequence_logprobs(model, batch)
        if is_peft:
            with reference_mode(model):
                ref, _, _ = response_sequence_logprobs(model, batch)
        else:
            ref = pol
        lens = mask.sum(-1)
        out["pol_c"] += pol[:n].tolist(); out["pol_r"] += pol[n:].tolist()
        out["ref_c"] += ref[:n].tolist(); out["ref_r"] += ref[n:].tolist()
        out["len_c"] += lens[:n].tolist(); out["len_r"] += lens[n:].tolist()
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def preference_metrics(lp: dict, beta: float, ref_beta: float = 0.10) -> dict:
    """Manual metrics from pair log-probs: margin m_theta, accuracy (m > 0), DPO loss."""
    margin = (lp["pol_c"] - lp["ref_c"]) - (lp["pol_r"] - lp["ref_r"])
    loss = lambda b: float(np.mean(np.logaddexp(0.0, -b * margin)))   # -log sigmoid(b*m), stable
    return {
        "n": int(len(margin)),
        "preference_accuracy": float(np.mean(margin > 0)),
        "margin_mean": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "dpo_loss_own_beta": loss(beta),
        "dpo_loss_beta_0.10": loss(ref_beta),
        "beta": float(beta),
        "chosen_logratio_mean": float(np.mean(lp["pol_c"] - lp["ref_c"])),
        "rejected_logratio_mean": float(np.mean(lp["pol_r"] - lp["ref_r"])),
    }


# --------------------------------------------------------------------------------------
# Generation-based metrics
# --------------------------------------------------------------------------------------
@torch.no_grad()
def generate_and_score(model, tokenizer, items, cfg, rm=None, rm_tok=None, batch_size=32, logp_batch=8, seed=None):
    """Generate one sampled response per item and compute length, reference KL and RM score.

    items: list of dicts with "id" and "messages". Decoding uses cfg["generation"] and
    cfg["max_generation_tokens"] for every condition. Items are processed longest-prompt first,
    so the batch composition is identical for every model.
    KL uses the manual's sampled-response estimator: per-token log pi - log ref on the generated
    tokens. Stored per sequence as (sum, n_tokens) so the token-averaged convention of
    common.metrics.sampled_kl can be reproduced exactly over the whole set.
    """
    gen = cfg["generation"]
    max_new = int(cfg["max_generation_tokens"])
    max_prompt = int(cfg.get("max_prompt_tokens", 512)) + 64     # rendered prompts are already filtered to fit
    is_peft = isinstance(model, PeftModel)
    order = sorted(range(len(items)), key=lambda i: -len(str(items[i]["messages"])))
    if seed is not None:
        torch.manual_seed(seed)
    records = {}
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        prompts = [items[i]["messages"] for i in idx]
        g = batch_generate(model, tokenizer, prompts, max_prompt, max_new,
                           temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
        seq, attn, pw, rids, rmask = g["sequences"], g["attention_mask"], g["prompt_width"], g["response_ids"], g["response_mask"]
        # zero out tokens after EOS (they are padding) so they never count toward KL
        attn = attn.clone()
        attn[:, pw:] = rmask.long()
        kl_sum, n_tok = [], []
        for s in range(0, len(idx), logp_batch):
            sl = slice(s, s + logp_batch)
            pol, _ = response_token_logprobs_lean(model, seq[sl], attn[sl], pw, rids[sl])
            if is_peft:
                with reference_mode(model):
                    ref, _ = response_token_logprobs_lean(model, seq[sl], attn[sl], pw, rids[sl])
            else:
                ref = pol
            m = rmask[sl]
            kl_sum += ((pol - ref) * m).sum(-1).tolist()
            n_tok += m.sum(-1).tolist()
        rewards = [None] * len(idx)
        if rm is not None:
            rewards = []
            for s in range(0, len(idx), 8):
                rewards += score_reward_pairs(rm, rm_tok, prompts[s : s + 8], g["responses"][s : s + 8],
                                              max_length=int(cfg.get("reward_max_length", 1024))).tolist()
        for j, i in enumerate(idx):
            records[i] = {
                "id": items[i]["id"],
                "response": g["responses"][j],
                "response_tokens": int(g["response_lengths"][j]),
                "terminated_with_eos": bool(g["terminated_with_eos"][j]),
                "truncated": bool(g["truncated"][j]),
                "kl_seq_sum": float(kl_sum[j]),
                "kl_n_tokens": int(n_tok[j]),
                "reward": None if rewards[j] is None else float(rewards[j]),
            }
    return [records[i] for i in range(len(items))]


def summarize_generations(recs: list[dict]) -> dict:
    lens = np.array([r["response_tokens"] for r in recs], dtype=float)
    kl = np.array([r["kl_seq_sum"] for r in recs]); nt = np.array([r["kl_n_tokens"] for r in recs])
    out = {
        "n": len(recs),
        "length_mean": float(lens.mean()),
        "length_std": float(lens.std(ddof=1)) if len(lens) > 1 else 0.0,
        "length_median": float(np.median(lens)),
        "length_iqr": float(np.percentile(lens, 75) - np.percentile(lens, 25)),
        "truncation_rate": float(np.mean([r["truncated"] for r in recs])),
        "kl_token_mean": float(kl.sum() / max(nt.sum(), 1)),   # == common.metrics.sampled_kl over all tokens
        "kl_seq_mean": float(kl.mean()),
    }
    rw = [r["reward"] for r in recs if r["reward"] is not None]
    if rw:
        rw = np.asarray(rw)
        out.update({"reward_mean": float(rw.mean()), "reward_std": float(rw.std(ddof=1)) if len(rw) > 1 else 0.0,
                    "reward_sem": float(rw.std(ddof=1) / np.sqrt(len(rw))) if len(rw) > 1 else 0.0})
    return out


def eval_generation_items(cfg, tokenizer) -> list[dict]:
    """Generation prompts: prompts of the eligible held-out DPO pairs (fixed order and IDs)."""
    rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    rows, _ = eligible_pairs(rows, tokenizer, int(cfg["max_prompt_tokens"]))
    n = cfg.get("eval_generation_prompts")
    if n:
        rows = rows[: int(n)]
    return [{"id": row_id(r), "messages": prompt_messages_from_preference(r)} for r in rows]


def run_module(module: str, *args: str, smoke: bool = False, overrides: list[str] | None = None) -> None:
    """Run `python -m module args...` in a fresh process (all CPU/GPU memory is released when it exits)."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", module, *args]
    if smoke:
        cmd.append("--smoke")
    if overrides:
        cmd += ["--set", *overrides]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)
