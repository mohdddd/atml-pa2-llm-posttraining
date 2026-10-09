"""Task 3, Step 3 — canonical GRPO vs Dr. GRPO-style normalisation.

Stage `forks` (GPU): two matched continuations from the identical GRPO midpoint, `fork_updates` each,
same prompt schedule, sampling seeds, reward, beta, epsilon, K and generation cap; only loss_type differs.
Each fork is then evaluated with the common held-out protocol.

Stage `analyze` (CPU): length-conditioned statistics from the forks' per-completion logs
(results/task3_grpo/runs/fork_<type>/completions.jsonl):
  - per-completion policy-gradient norm (measured, see continue_train.py) vs response length:
    Spearman correlation, and means in three length bins whose edges are the terciles of the POOLED
    lengths of both forks (so both forks use identical bins), split by advantage sign;
  - share of the summed per-completion gradient norm carried by each length bin;
  - the per-token loss weight implied by the normaliser, |A_k|/T_k (grpo) or |A_k|/L_max (dr_grpo);
  - training-time trajectories (reward, KL, length) and the held-out summaries of both forks.
  -> results/task3_grpo/normalization/comparison.json

    python -m task3_grpo.compare_normalization --config configs/grpo.yaml --stage all
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import read_jsonl, repo_path
from common.experiment import run_metadata
from common.logging_utils import load_json, save_json
from common.rl import add_common_args, load_config, override_args, run_module

LOSS_TYPES = ["grpo", "dr_grpo"]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def stage_forks(cfg, path, overwrite=False):
    root = cfg["output"].rsplit("/", 1)[0]
    extra = override_args(cfg) + (["--overwrite"] if overwrite else [])
    for lt in LOSS_TYPES:
        out = f"{root}/forks/{lt}"
        run_module("task3_grpo.continue_train", "--config", path, "--run-name", f"fork_{lt}",
                   "--updates", str(cfg["fork_updates"]), "--loss-type", lt, "--output", out, *extra)
        run_module("task3_grpo.evaluate", "--config", path, "--adapter", out, "--name", f"fork_{lt}", *extra)


def stage_analyze(cfg):
    res_dir = repo_path(cfg["results_dir"])
    comps = {lt: read_jsonl(res_dir / "runs" / f"fork_{lt}" / "completions.jsonl") for lt in LOSS_TYPES}
    logs = {lt: read_jsonl(res_dir / "runs" / f"fork_{lt}" / "train_log.jsonl") for lt in LOSS_TYPES}
    L = int(cfg["max_completion_length"])
    pooled = [c["response_tokens"] for lt in LOSS_TYPES for c in comps[lt] if c["loss_tokens"] > 0]
    edges = [float(np.percentile(pooled, 100 / 3)), float(np.percentile(pooled, 200 / 3))]
    bin_of = lambda n: 0 if n <= edges[0] else (1 if n <= edges[1] else 2)
    names = ["short", "medium", "long"]

    out = {"length_bin_edges_tokens": edges, "bin_rule": "terciles of pooled lengths of loss-contributing completions of both forks",
           "max_completion_length": L, "per_loss_type": {}}
    for lt in LOSS_TYPES:
        rows = [c for c in comps[lt] if c["loss_tokens"] > 0 and c["advantage"] != 0.0 and c["seq_policy_grad_norm"] is not None]
        T = np.array([c["response_tokens"] for c in rows], float)
        G = np.array([c["seq_policy_grad_norm"] for c in rows], float)
        A = np.array([c["advantage"] for c in rows], float)
        W = np.abs(A) / (T if lt == "grpo" else L)
        bins = {}
        for bi, nm in enumerate(names):
            sel = np.array([bin_of(t) == bi for t in T], bool)
            entry = {"n": int(sel.sum()),
                     "grad_norm_mean": float(G[sel].mean()) if sel.any() else None,
                     "grad_norm_share": float(G[sel].sum() / G.sum()) if G.sum() > 0 else None,
                     "token_weight_mean": float(W[sel].mean()) if sel.any() else None}
            for sign, m in (("pos_adv", A > 0), ("neg_adv", A < 0)):
                s2 = sel & m
                entry[f"grad_norm_mean_{sign}"] = float(G[s2].mean()) if s2.any() else None
                entry[f"n_{sign}"] = int(s2.sum())
            bins[nm] = entry
        tr = logs[lt]
        allc = comps[lt]
        out["per_loss_type"][lt] = {
            "n_completions_total": len(allc),
            "n_masked_truncated": int(sum(c["loss_tokens"] == 0 for c in allc)),
            "n_in_gradient_stats": len(rows),
            "spearman_length_vs_seq_grad_norm": spearman(T, G),
            "spearman_length_vs_seq_grad_norm_pos_adv": spearman(T[A > 0], G[A > 0]),
            "spearman_length_vs_seq_grad_norm_neg_adv": spearman(T[A < 0], G[A < 0]),
            "length_bins": bins,
            "train_reward_first_last": [tr[0]["reward_mean"], tr[-1]["reward_mean"]],
            "train_reward_mean": float(np.mean([r["reward_mean"] for r in tr])),
            "train_length_mean": float(np.mean([c["response_tokens"] for c in allc])),
            "train_length_by_update": [r["response_length"] for r in tr],
            "train_kl_by_update": [r["kl_token_mean"] for r in tr],
            "generated_tokens": tr[-1]["generated_tokens_cum"],
        }
        ev = res_dir / "eval" / f"fork_{lt}" / "summary.json"
        if ev.exists():
            out["per_loss_type"][lt]["heldout"] = load_json(ev)["metrics"]
    out["meta"] = run_metadata(cfg)
    (res_dir / "normalization").mkdir(parents=True, exist_ok=True)
    save_json(res_dir / "normalization" / "comparison.json", out)
    for lt in LOSS_TYPES:
        d = out["per_loss_type"][lt]
        print(f"[norm {lt}] rho(len, grad)={d['spearman_length_vs_seq_grad_norm']:.3f} "
              f"share short/med/long={[round(d['length_bins'][n]['grad_norm_share'] or 0, 3) for n in names]} "
              f"heldout={ {k: round(v, 4) for k, v in d.get('heldout', {}).items() if k in ('reward_mean', 'kl_token_mean', 'length_mean')} }", flush=True)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--stage", choices=["forks", "analyze", "all"], default="all")
    args = ap.parse_args()
    path = args.config or "configs/grpo.yaml"
    cfg = load_config(path, args.set)
    if args.stage in ("forks", "all"):
        stage_forks(cfg, path, args.overwrite)
    if args.stage in ("analyze", "all"):
        stage_analyze(cfg)


if __name__ == "__main__":
    main()
