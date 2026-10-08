"""Collect Task 1 results into tables (report/tables/) and figures (report/figures/task1/).

Runtime: CPU, seconds. Reads only results/task1_dpo/**; never touches a model.
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
from task1_dpo.utils import add_common_args, load_task_config

MODELS = ["sft", "standard", "beta_0.03", "beta_0.10", "beta_0.30", "length_balanced"]
LABELS = {"sft": "SFT (reference)", "standard": "DPO standard (1 epoch)", "beta_0.03": "DPO beta=0.03 (short)",
          "beta_0.10": "DPO beta=0.10 (short)", "beta_0.30": "DPO beta=0.30 (short)", "length_balanced": "DPO length-balanced (1 epoch)"}


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_task_config(args.config, smoke=args.smoke, overrides=args.set)
    res = repo_path(cfg["results_dir"])
    tab_dir = repo_path("report/tables") if not args.smoke else res / "tables"
    fig_dir = repo_path("report/figures/task1") if not args.smoke else res / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)

    rows, strata_rows, wl_rows = [], [], []
    for m in MODELS:
        ev = _load(res / "eval" / m / "summary.json")
        tr = _load(res / "runs" / m / "train_summary.json")
        if ev is None and tr is None:
            continue
        ev = ev or {}
        h, g, w, s = ev.get("heldout", {}), ev.get("generate", {}), ev.get("wordlimit", {}), ev.get("stratified", {})
        rows.append({
            "model": m, "label": LABELS[m],
            "train_file": (tr or {}).get("dataset", "-"), "beta": (tr or {}).get("beta", None),
            "train_pairs": (tr or {}).get("n_pairs", 0), "optimizer_steps": (tr or {}).get("optimizer_steps", 0),
            "heldout_n": h.get("n"), "pref_accuracy": h.get("preference_accuracy"), "margin_mean": h.get("margin_mean"),
            "dpo_loss_own_beta": h.get("dpo_loss_own_beta"), "dpo_loss_beta_0.10": h.get("dpo_loss_beta_0.10"),
            "gen_n": g.get("n"), "kl_token_mean": g.get("kl_token_mean"), "kl_seq_mean": g.get("kl_seq_mean"),
            "reward_mean": g.get("reward_mean"), "reward_sem": g.get("reward_sem"),
            "len_mean": g.get("length_mean"), "len_std": g.get("length_std"), "len_iqr": g.get("length_iqr"),
            "truncation_rate": g.get("truncation_rate"),
            "train_wall_min": round(tr["wall_clock_s"] / 60, 1) if tr else None,
            "train_peak_vram_gib": (tr or {}).get("peak_vram_gib"),
        })
        for st in ("preferred_longer", "length_matched", "rejected_longer", "all"):
            if st in s:
                strata_rows.append({"model": m, "stratum": st, "n": s[st].get("n"), "pref_accuracy": s[st].get("preference_accuracy"),
                                    "margin_mean": s[st].get("margin_mean"), "dpo_loss_beta_0.10": s[st].get("dpo_loss_beta_0.10")})
        if w:
            wl_rows.append({"model": m, "n_responses": w["n_responses"], "compliance_rate": w["compliance_rate"],
                            "words_mean": w["words_mean"], "words_std": w["words_std"], "tokens_mean": w["tokens_mean"],
                            "prompts_all_compliant_of_10": w["prompts_all_compliant"]})

    pd.DataFrame(rows).to_csv(tab_dir / "task1_dpo_summary.csv", index=False)
    if strata_rows:
        pd.DataFrame(strata_rows).to_csv(tab_dir / "task1_length_strata.csv", index=False)
    if wl_rows:
        pd.DataFrame(wl_rows).to_csv(tab_dir / "task1_word_limit.csv", index=False)
    stats = _load(res / "dataset_length_stats.json")
    if stats:
        pd.DataFrame({k: v for k, v in stats.items() if isinstance(v, dict)}).T.drop(columns=["stratum_counts"], errors="ignore") \
            .to_csv(tab_dir / "task1_dataset_length_stats.csv")

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 4):
        print(pd.DataFrame(rows).drop(columns=["label", "train_file"]).to_string(index=False))
        if strata_rows:
            print(pd.DataFrame(strata_rows).pivot(index="model", columns="stratum", values="pref_accuracy").to_string())
        if wl_rows:
            print(pd.DataFrame(wl_rows).to_string(index=False))

    # Figure: training curves (loss and batch preference accuracy per optimizer step)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for m in MODELS[1:]:
        p = res / "runs" / m / "train_log.jsonl"
        if p.exists():
            log = pd.DataFrame(read_jsonl(p))
            ax[0].plot(log["step"], log["loss"], label=LABELS[m], lw=1.2)
            ax[1].plot(log["step"], log["preference_accuracy"].rolling(5, min_periods=1).mean(), lw=1.2)
    ax[0].set(xlabel="optimizer step", ylabel="training DPO loss (own beta)"); ax[0].axhline(np.log(2), ls=":", c="grey", lw=0.8)
    ax[1].set(xlabel="optimizer step", ylabel="train pref. accuracy (5-step mean)")
    ax[0].legend(fontsize=7); fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"train_curves.{ext}", dpi=200)

    # Figure: generated-length distributions
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for m in MODELS:
        p = res / "eval" / m / "generations.jsonl"
        if p.exists():
            lens = [r["response_tokens"] for r in read_jsonl(p)]
            ax.hist(lens, bins=30, histtype="step", label=LABELS[m], lw=1.2)
    ax.set(xlabel="generated response tokens", ylabel="count"); ax.legend(fontsize=7); fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"gen_length_hist.{ext}", dpi=200)
    print("tables ->", tab_dir, "| figures ->", fig_dir)


if __name__ == "__main__":
    main()
