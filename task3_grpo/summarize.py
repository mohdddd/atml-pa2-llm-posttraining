"""Collect Task 3 (GRPO) results into tables (report/tables/task3_*.csv) and figures (report/figures/task3/).

Runtime: CPU, seconds. Reads only results/task3_grpo/**; never touches a model.

Tables
  task3_standard_trajectory.csv  per-update diagnostics of the 20-update continuation
  task3_heldout.csv              held-out metrics of SFT, midpoint, standard and both normalisation forks
                                 (+ paired reward difference vs the midpoint, bootstrap 95% CI)
  task3_group_size.csv           K in {2,4,8} x {all, difficulty bins}, RM reward and binarised diagnostic,
                                 disjoint regrouping (primary) and all-subsets expectation
  task3_normalization.csv        canonical vs Dr. GRPO: held-out metrics + length-conditioned gradient stats
  task3_compute.csv              wall-clock and peak VRAM of every run
Figures: standard_trajectory, group_size, normalization_grad_vs_length (PDF + PNG).
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
from task2_ppo.summarize import heldout_row, savefig

STD_METRICS = [("reward_mean", "reward (RM)"), ("kl_token_mean", "KL to reference (token mean)"),
               ("group_reward_std_mean", "within-group reward std"), ("uninformative_group_fraction", "uninformative-group fraction"),
               ("policy_loss", "policy loss"), ("grad_norm", "grad norm (pre-clip)"), ("entropy", "entropy (nats/token)"),
               ("response_length", "response length (tokens)")]
BINS = ["all", "low_reward_hard", "mid", "high_reward_easy"]


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config or "configs/grpo.yaml", args.set)
    res = repo_path(cfg["results_dir"])
    real = not cfg.get("cli_overrides")
    tab = repo_path("report/tables") if real else res / "tables"
    fig_dir = repo_path("report/figures/task3") if real else res / "figures"
    tab.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)

    std_log = read_jsonl(res / "runs" / "standard" / "train_log.jsonl")
    cols = ["update", "reward_mean", "kl_token_mean", "kl_k3_loss_mask", "group_reward_std_mean", "uninformative_group_fraction",
            "policy_loss", "loss", "grad_norm", "clip_fraction", "entropy", "sample_entropy", "response_length",
            "truncation_rate", "masked_completion_fraction", "skipped_steps", "update_s"]
    pd.DataFrame([{c: r.get(c) for c in cols} for r in std_log]).to_csv(tab / "task3_standard_trajectory.csv", index=False)

    rows = [heldout_row(res, n, l) for n, l in [("sft", "SFT (no adapter)"), ("midpoint", "GRPO midpoint (start of every run)"),
                                                 ("standard", "GRPO standard (20 updates)"),
                                                 ("fork_grpo", f"fork canonical GRPO ({cfg['fork_updates']} upd.)"),
                                                 ("fork_dr_grpo", f"fork Dr. GRPO ({cfg['fork_updates']} upd.)")]]
    held = pd.DataFrame([r for r in rows if r])
    held.to_csv(tab / "task3_heldout.csv", index=False)

    gs_path = res / "group_size" / "group_size_study.json"
    if gs_path.exists():
        gs = json.loads(gs_path.read_text())
        grow = []
        for sig, sigd in (("rm_reward", gs["rm_reward"]), ("binarized_at_median", gs["binarized_at_median"])):
            for scheme in ("disjoint", "exhaustive_subsets"):
                for k in ("2", "4", "8"):
                    for b in BINS:
                        d = sigd[scheme][k].get(b)
                        if d:
                            grow.append({"reward": sig, "grouping": scheme, "K": int(k), "bin": b, **d})
        pd.DataFrame(grow).to_csv(tab / "task3_group_size.csv", index=False)
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
        for ax, (metric, title) in zip(axes, [("informative_group_rate", "informative-group rate (std > 1e-6)"),
                                              ("informative_rate_std_gt_0.25", "groups with std > 0.25"),
                                              ("mean_group_reward_std", "mean within-group reward std"),
                                              ("baseline_mse", "baseline MSE vs 8-sample mean")]):
            for j, b in enumerate(BINS):
                vals = [gs["rm_reward"]["disjoint"][k][b][metric] for k in ("2", "4", "8")]
                ax.bar(np.arange(3) + (j - 1.5) * 0.2, vals, width=0.2, label=b)
            ax.set_xticks(range(3)); ax.set_xticklabels(["K=2", "K=4", "K=8"]); ax.set_title(title, fontsize=9)
        axes[0].legend(fontsize=7)
        savefig(fig, fig_dir, "group_size")

    nc = res / "normalization" / "comparison.json"
    if nc.exists():
        comp = json.loads(nc.read_text())
        nrow = []
        for lt, d in comp["per_loss_type"].items():
            h = d.get("heldout", {})
            r = {"loss_type": lt, "heldout_reward_mean": h.get("reward_mean"), "heldout_reward_sem": h.get("reward_sem"),
                 "heldout_kl_token_mean": h.get("kl_token_mean"), "heldout_entropy": h.get("entropy_token_mean"),
                 "heldout_len_mean": h.get("length_mean"), "heldout_len_std": h.get("length_std"), "heldout_truncation": h.get("truncation_rate"),
                 "train_len_mean": d["train_length_mean"], "train_reward_mean": d["train_reward_mean"], "generated_tokens": d["generated_tokens"],
                 "n_masked_truncated": d["n_masked_truncated"], "spearman_len_grad": d["spearman_length_vs_seq_grad_norm"],
                 "spearman_len_grad_pos_adv": d["spearman_length_vs_seq_grad_norm_pos_adv"],
                 "spearman_len_grad_neg_adv": d["spearman_length_vs_seq_grad_norm_neg_adv"]}
            for b, bd in d["length_bins"].items():
                r[f"{b}_n"] = bd["n"]; r[f"{b}_grad_norm_mean"] = bd["grad_norm_mean"]; r[f"{b}_grad_share"] = bd["grad_norm_share"]
                r[f"{b}_token_weight_mean"] = bd["token_weight_mean"]
            nrow.append(r)
        pd.DataFrame(nrow).to_csv(tab / "task3_normalization.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
        for ax, lt in zip(axes, ("grpo", "dr_grpo")):
            cp = res / "runs" / f"fork_{lt}" / "completions.jsonl"
            if not cp.exists():
                continue
            c = [x for x in read_jsonl(cp) if x["loss_tokens"] > 0 and x["seq_policy_grad_norm"]]
            for sign, col in ((1, "tab:blue"), (-1, "tab:red")):
                s = [x for x in c if np.sign(x["advantage"]) == sign]
                ax.scatter([x["response_tokens"] for x in s], [x["seq_policy_grad_norm"] for x in s], s=12, c=col,
                           label="A>0" if sign > 0 else "A<0", alpha=0.7)
            for e in comp["length_bin_edges_tokens"]:
                ax.axvline(e, color="gray", lw=0.8, ls="--")
            ax.set_title(f"{lt}: per-completion policy-gradient norm", fontsize=9); ax.set_xlabel("response tokens")
        axes[0].set_ylabel("grad norm"); axes[0].legend(fontsize=8)
        savefig(fig, fig_dir, "normalization_grad_vs_length")

    comp_rows = []
    for d in sorted((res / "runs").glob("*/train_summary.json")):
        s = json.loads(d.read_text())
        comp_rows.append({"run": s["run_name"], "updates": s["updates"], "loss_type": s["loss_type"], "generated_tokens": s["generated_tokens"],
                          "wall_clock_min": round(s["wall_clock_s"] / 60, 2), "peak_vram_gib": s["peak_vram_gib"],
                          "gpu": s.get("gpu"), "git_commit": s.get("git_commit")})
    pd.DataFrame(comp_rows).to_csv(tab / "task3_compute.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    x = [r["update"] for r in std_log]
    for ax, (k, t) in zip(axes.flat, STD_METRICS):
        ax.plot(x, [r[k] for r in std_log], marker="o", ms=3)
        ax.set_title(t, fontsize=9); ax.set_xlabel("update")
    savefig(fig, fig_dir, "standard_trajectory")
    if len(held):
        print(held[["model", "reward_mean", "kl_token_mean", "entropy_token_mean", "len_mean", "truncation_rate"]].to_string(index=False))
    print(f"tables -> {tab}, figures -> {fig_dir}")


if __name__ == "__main__":
    main()
