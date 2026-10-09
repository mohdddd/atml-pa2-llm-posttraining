"""Collect Task 2 (PPO) results into tables (report/tables/task2_*.csv) and figures (report/figures/task2/).

Runtime: CPU, seconds. Reads only results/task2_ppo/** (and results/task3_grpo/eval/sft if present
for the shared SFT reference row); never touches a model.

Tables
  task2_standard_trajectory.csv  per-update diagnostics of the 20-update continuation
  task2_heldout.csv              held-out metrics of every evaluated policy (+ paired reward difference
                                 vs the midpoint on identical prompts/seed, bootstrap 95% CI)
  task2_clip_cached.csv          cached-batch geometry per epsilon (step 0 and after the matched update)
  task2_clip_forks.csv           epsilon forks: held-out metrics + training stability statistics
  task2_kl_forks.csv             beta forks: held-out reward/KL/entropy/length + training trajectory ends
  task2_compute.csv              wall-clock and peak VRAM of every run
Figures: standard_trajectory, fork_trajectories, cached_clip_steps (PDF + PNG).
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import read_jsonl, repo_path
from common.rl import add_common_args, load_config
from task2_ppo.analyze_clipping import fork_name

STD_METRICS = [("reward_raw", "learned reward (raw RM)"), ("kl_token_mean", "KL to reference (token mean)"),
               ("policy_loss", "policy loss"), ("value_loss", "value loss"), ("entropy", "entropy (nats/token)"),
               ("clip_fraction", "clip fraction"), ("grad_norm", "policy grad norm (pre-clip)"),
               ("response_length", "response length (tokens)")]


def _load(p):
    return json.loads(p.read_text()) if p.exists() else None


def bootstrap_ci(x, n=5000, seed=6304):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_vs(res, name, base):
    a, b = res / "eval" / name / "generations.jsonl", res / "eval" / base / "generations.jsonl"
    if not (a.exists() and b.exists()) or name == base:
        return {}
    A = {r["id"]: r for r in read_jsonl(a)}; B = {r["id"]: r for r in read_jsonl(b)}
    ids = [i for i in A if i in B]
    d = np.array([A[i]["reward"] - B[i]["reward"] for i in ids])
    lo, hi = bootstrap_ci(d)
    return {f"reward_diff_vs_{base}": float(d.mean()), "ci95_low": lo, "ci95_high": hi,
            f"identical_text_vs_{base}": float(np.mean([A[i]["response"] == B[i]["response"] for i in ids]))}


def heldout_row(res, name, label, base="midpoint"):
    ev = _load(res / "eval" / name / "summary.json")
    if ev is None:
        return None
    m = ev["metrics"]
    return {"model": name, "label": label, "n_prompts": m["n"], "reward_mean": m.get("reward_mean"), "reward_sem": m.get("reward_sem"),
            "effective_reward_mean": m.get("effective_reward_mean"), "kl_token_mean": m["kl_token_mean"], "kl_seq_mean": m["kl_seq_mean"],
            "entropy_token_mean": m["entropy_token_mean"], "len_mean": m["length_mean"], "len_std": m["length_std"],
            "len_iqr": m["length_iqr"], "truncation_rate": m["truncation_rate"], "eos_rate": m["eos_rate"],
            **paired_vs(res, name, base)}


def stability(log):
    """Training stability statistics of one continuation (defined once, used for every fork)."""
    ep = [e for r in log for e in r["epochs"]]
    return {"updates": len(log),
            "max_kl_old_new": float(np.max([e["approx_kl_old_new"] for e in ep])),
            "mean_kl_old_new_epoch2": float(np.mean([r["epochs"][-1]["approx_kl_old_new"] for r in log])),
            "mean_clip_fraction": float(np.mean([r["clip_fraction"] for r in log])),
            "mean_affected_fraction": float(np.mean([r["affected_fraction"] for r in log])),
            "grad_norm_mean": float(np.mean([e["grad_norm"] for e in ep])), "grad_norm_max": float(np.max([e["grad_norm"] for e in ep])),
            "policy_loss_std": float(np.std([r["policy_loss"] for r in log])),
            "value_loss_mean": float(np.mean([r["value_loss"] for r in log])),
            "skipped_steps": int(sum(r["skipped_steps"] for r in log)),
            "train_reward_mean": float(np.mean([r["reward_raw"] for r in log])),
            "train_kl_last": log[-1]["kl_token_mean"], "train_entropy_last": log[-1]["entropy"],
            "train_len_mean": float(np.mean([r["response_length"] for r in log]))}


def savefig(fig, d, name):
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config or "configs/ppo.yaml", args.set)
    res = repo_path(cfg["results_dir"])
    real = not cfg.get("cli_overrides")
    tab = repo_path("report/tables") if real else res / "tables"
    fig_dir = repo_path("report/figures/task2") if real else res / "figures"
    tab.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    eps0, beta0 = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])

    # standard trajectory
    std_log = read_jsonl(res / "runs" / "standard" / "train_log.jsonl")
    cols = ["update", "reward_raw", "reward_effective", "kl_token_mean", "kl_seq_sum_mean", "policy_loss", "value_loss",
            "entropy", "sample_entropy", "clip_fraction", "affected_fraction", "grad_norm", "value_grad_norm_mean",
            "response_length", "eos_rate", "truncation_rate", "value_mean", "return_mean", "explained_variance",
            "approx_kl_old_new_last", "skipped_steps", "update_s"]
    for r in std_log:
        r["value_grad_norm_mean"] = float(np.mean([e["value_grad_norm"] for e in r["epochs"]]))
    pd.DataFrame([{c: r.get(c) for c in cols} for r in std_log]).to_csv(tab / "task2_standard_trajectory.csv", index=False)

    # held-out
    rows = []
    sft = heldout_row(repo_path("results/task3_grpo"), "sft", "SFT (no adapter; shared eval set)")
    for name, label in [("midpoint", "PPO midpoint (start of every run)"), ("standard", "PPO standard (20 updates)")] + \
            [(f"fork_{fork_name(e, beta0)}", f"fork eps={e:.2f} beta={beta0:.2f} ({cfg['fork_updates']} upd.)") for e in cfg["clip_values"]] + \
            [(f"fork_{fork_name(eps0, b)}", f"fork eps={eps0:.2f} beta={b:.2f} ({cfg['fork_updates']} upd.)") for b in cfg["kl_values"] if abs(b - beta0) > 1e-12]:
        r = heldout_row(res, name, label)
        if r:
            rows.append(r)
    if sft:
        rows.insert(0, sft)
    held = pd.DataFrame(rows)
    held.to_csv(tab / "task2_heldout.csv", index=False)

    # cached clip study
    cs = _load(res / "clipping" / "cached_study.json")
    if cs:
        crow = []
        steps = pd.DataFrame(read_jsonl(res / "clipping" / "cached_steps.jsonl"))
        for e in cfg["clip_values"]:
            k = f"{float(e):.2f}"
            s0 = cs["step0_geometry"][k]; after = cs["per_epsilon"][k]["after_update_geometry_all_eps"][k]
            st = steps[np.isclose(steps["epsilon"], float(e))]
            crow.append({"epsilon": float(e), "n_tokens": cs["n_tokens"],
                         "step0_clip_fraction": s0["clip_fraction"], "step0_affected_fraction": s0["affected_fraction"],
                         "step0_surrogate_clipped": s0["surrogate_clipped"], "step0_surrogate_unclipped": s0["surrogate_unclipped"],
                         "update_mean_clip_fraction": float(st["clip_fraction"].mean()), "update_mean_affected_fraction": float(st["affected_fraction"].mean()),
                         "update_mean_grad_norm": float(st["grad_norm"].mean()),
                         "after_clip_fraction": after["clip_fraction"], "after_affected_fraction": after["affected_fraction"],
                         "after_surrogate_clipped": after["surrogate_clipped"], "after_surrogate_unclipped": after["surrogate_unclipped"],
                         "after_kl_old_new": cs["per_epsilon"][k]["after_update_ratio_stats"]["kl_old_new_k3"],
                         "after_abs_log_ratio_mean": cs["per_epsilon"][k]["after_update_ratio_stats"]["abs_log_ratio_mean"]})
        pd.DataFrame(crow).to_csv(tab / "task2_clip_cached.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for e in cfg["clip_values"]:
            st = steps[np.isclose(steps["epsilon"], float(e))]
            for ax, c in zip(axes, ["clip_fraction", "affected_fraction", "grad_norm"]):
                ax.plot(st["step"], st[c], marker="o", label=f"eps={e}")
        for ax, t in zip(axes, ["clip fraction (minibatch)", "affected-token fraction", "grad norm (pre-clip)"]):
            ax.set_title(t, fontsize=9); ax.set_xlabel("optimizer step on cached batch")
        axes[0].legend(fontsize=8)
        savefig(fig, fig_dir, "cached_clip_steps")

    # forks
    def fork_table(conds, fname):
        out = []
        for e, b in conds:
            name = f"fork_{fork_name(e, b)}"
            lp = res / "runs" / name / "train_log.jsonl"
            if not lp.exists():
                continue
            hr = heldout_row(res, name, name) or {}
            out.append({"epsilon": e, "kl_beta": b, **{k: v for k, v in hr.items() if k not in ("model", "label")}, **stability(read_jsonl(lp))})
        pd.DataFrame(out).to_csv(tab / fname, index=False)
        return out
    fork_table([(float(e), beta0) for e in cfg["clip_values"]], "task2_clip_forks.csv")
    fork_table([(eps0, float(b)) for b in cfg["kl_values"]], "task2_kl_forks.csv")

    # compute
    comp = []
    for d in sorted((res / "runs").glob("*/train_summary.json")):
        s = json.loads(d.read_text())
        comp.append({"run": s["run_name"], "updates": s["updates"], "clip_epsilon": s["clip_epsilon"], "kl_beta": s["kl_beta"],
                     "generated_tokens": s["generated_tokens"], "wall_clock_min": round(s["wall_clock_s"] / 60, 2),
                     "peak_vram_gib": s["peak_vram_gib"], "gpu": s.get("gpu"), "git_commit": s.get("git_commit")})
    pd.DataFrame(comp).to_csv(tab / "task2_compute.csv", index=False)

    # figures
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    x = [r["update"] for r in std_log]
    for ax, (k, t) in zip(axes.flat, STD_METRICS):
        ax.plot(x, [r[k] for r in std_log], marker="o", ms=3)
        ax.set_title(t, fontsize=9); ax.set_xlabel("update")
    savefig(fig, fig_dir, "standard_trajectory")

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for row, (conds, title) in enumerate([([(float(e), beta0) for e in cfg["clip_values"]], "eps"),
                                          ([(eps0, float(b)) for b in cfg["kl_values"]], "beta")]):
        for e, b in conds:
            lp = res / "runs" / f"fork_{fork_name(e, b)}" / "train_log.jsonl"
            if not lp.exists():
                continue
            lg = read_jsonl(lp)
            lab = f"eps={e:.2f}" if title == "eps" else f"beta={b:.2f}"
            for ax, k in zip(axes[row], ["reward_raw", "kl_token_mean", "entropy", "response_length"]):
                ax.plot([r["update"] for r in lg], [r[k] for r in lg], marker="o", ms=3, label=lab)
        for ax, k in zip(axes[row], ["learned reward", "KL to reference", "entropy", "response length"]):
            ax.set_title(f"{title} forks: {k}", fontsize=9); ax.set_xlabel("update")
        axes[row][0].legend(fontsize=8)
    savefig(fig, fig_dir, "fork_trajectories")
    print(held[["model", "reward_mean", "kl_token_mean", "entropy_token_mean", "len_mean", "truncation_rate"]].to_string(index=False))
    print(f"tables -> {tab}, figures -> {fig_dir}")


if __name__ == "__main__":
    main()
