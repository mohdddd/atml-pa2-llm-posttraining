"""Task 5: combine in-domain, diagnostic and transfer results into report tables (CPU, seconds).

    python -m task5_feedback.compare_feedback --config configs/feedback.yaml

Reads <results_dir>/task5_feedback/{gsm,transfer}/summary.json, diagnostics/*.  Writes
report/tables/task5_*.csv, report/figures/task5/diagnostic_pairs.{pdf,png} (preflight/override runs:
under <results_dir>/task5_feedback/{tables,figures}) and a qualitative-example shortlist.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import read_jsonl, repo_path, write_jsonl
from common.rl import add_common_args, load_config
from task5_feedback.evaluate_math import POLICIES


def _j(p):
    return json.loads(p.read_text())


def policy_table(summ, dataset):
    rows = []
    for p in POLICIES:
        s = summ["policies"][p]
        row = {"dataset": dataset, "policy": p, "n": s["n"], "accuracy": s["accuracy"], "accuracy_sem": s["accuracy_sem"],
               "format_compliance": s["format_compliance"], "len_mean": s["len_mean"], "len_sd": s["len_sd"],
               "len_median": s["len_median"], "len_iqr": s["len_iqr"], "truncation_rate": s["truncation_rate"]}
        if p == "sft":
            row.update({"ai_win_rate_vs_sft": 0.5, "wins": None, "ties": None, "losses": None,
                        "acc_diff_vs_sft": 0.0, "acc_diff_ci_lo": None, "acc_diff_ci_hi": None})
        else:
            c = summ["pairwise"][f"{p}_vs_sft"]
            d = s["accuracy_diff_vs_sft"]
            row.update({"ai_win_rate_vs_sft": c["win_rate_a"], "wins": c["wins_a"], "ties": c["ties"], "losses": c["losses_a"],
                        "acc_diff_vs_sft": d[0], "acc_diff_ci_lo": d[1], "acc_diff_ci_hi": d[2],
                        "verifier_judge_agreement_3way": c["verifier_judge_agreement_3way"],
                        "identical_text_vs_sft": c["identical_text"]})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config or "configs/feedback.yaml", args.set)
    res = repo_path(cfg["results_dir"]) / "task5_feedback"
    real = not cfg.get("cli_overrides")
    tab = repo_path("report/tables") if real else res / "tables"
    fig_dir = repo_path("report/figures/task5") if real else res / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    gsm, tr = _j(res / "gsm" / "summary.json"), _j(res / "transfer" / "summary.json")
    ind, ood = pd.DataFrame(policy_table(gsm, "gsm8k")), pd.DataFrame(policy_table(tr, "svamp"))
    ind.to_csv(tab / "task5_in_domain.csv", index=False)
    ood = ood.merge(ind[["policy", "accuracy", "ai_win_rate_vs_sft", "len_mean"]].rename(
        columns={"accuracy": "acc_in_domain", "ai_win_rate_vs_sft": "win_in_domain", "len_mean": "len_in_domain"}), on="policy")
    ood["accuracy_drop"] = (ood["acc_in_domain"] - ood["accuracy"]).round(4)
    ood["win_rate_drop"] = (ood["win_in_domain"] - ood["ai_win_rate_vs_sft"]).round(4)
    ood.to_csv(tab / "task5_transfer.csv", index=False)

    agr, fail = [], []
    for ds, s in (("gsm8k", gsm), ("svamp", tr)):
        for comp, c in s["pairwise"].items():
            agr.append({"dataset": ds, "comparison": comp, **{k: v for k, v in c.items() if k != "judge_on_identical_text"},
                        **{f"judge_on_identical_{k}": v for k, v in c["judge_on_identical_text"].items()}})
        for p in POLICIES:
            fail.append({"dataset": ds, "policy": p, **s["policies"][p]["failure_types"]})
    pd.DataFrame(agr).to_csv(tab / "task5_pairwise_and_agreement.csv", index=False)
    pd.DataFrame(fail).to_csv(tab / "task5_failure_types.csv", index=False)

    d = _j(res / "diagnostics" / "summary.json")
    prow = []
    for pair, ps in d["pairs"].items():
        for mech in ("verifier", "judge"):
            r = ps[mech]
            prow.append({"pair": pair, "mechanism": "RLVR_exact_verifier" if mech == "verifier" else "RLAIF_pairwise_judge",
                         "n": r["n"], "better_rate": r.get("better"), "tie_rate": r.get("tie"), "wrong_rate": r.get("wrong"),
                         "prefers_first_rate": r.get("first"), "prefers_second_rate": r.get("second"),
                         "judge_order_A_shown_first": json.dumps(ps["judge_fixed_order_first_shown_first"]) if mech == "judge" else None,
                         "judge_order_B_shown_first": json.dumps(ps["judge_fixed_order_second_shown_first"]) if mech == "judge" else None,
                         "judge_order_consistency": ps["judge_order_consistency"] if mech == "judge" else None})
    pairs_df = pd.DataFrame(prow)
    pairs_df.to_csv(tab / "task5_diagnostic_pairs.csv", index=False)
    srows = [{"mechanism": m, **v} for m, v in d["sensitivity"].items()]
    srows[1].update({"judge_position_A_rate_fixed_order": d["judge_position_A_rate_fixed_order"],
                     "judge_order_consistency_controlled_pairs": d["judge_order_consistency_all_pairs"]})
    pd.DataFrame(srows).to_csv(tab / "task5_diagnostic_sensitivity.csv", index=False)
    pd.DataFrame([{"variant": k, **v} for k, v in d["variants"].items()]).to_csv(tab / "task5_diagnostic_variants.csv", index=False)

    comp = []
    for ds, sub in (("gsm", "gsm8k"), ("transfer", "svamp")):
        for p in POLICIES:
            g = res / ds / f"generation_{p}.json"
            if g.exists():
                x = _j(g)
                comp.append({"dataset": sub, "stage": f"generate_{p}", "wall_s": x["wall_s"], "peak_vram_gib": x["peak_vram_gib"],
                             "n": x["n_prompts"]})
        jr = res / ds / "judge_run.json"
        if jr.exists():
            x = _j(jr)
            comp.append({"dataset": sub, "stage": "pairwise_judge", "wall_s": x["wall_s"], "peak_vram_gib": x["peak_vram_gib"],
                         "n": x["n_comparisons"], "new_judge_calls": x["new_judge_calls"]})
    comp.append({"dataset": "diagnostics", "stage": "judge+verifier", "wall_s": d["wall_s"], "n": d["n_problems"]})
    pd.DataFrame(comp).to_csv(tab / "task5_compute.csv", index=False)

    # figure: better / tie / wrong per controlled pair and mechanism
    sub = pairs_df[pairs_df.pair != "conflict"]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    labels = [f"{r.pair}\n{'verifier' if r.mechanism.startswith('RLVR') else 'judge'}" for r in sub.itertuples()]
    left = np.zeros(len(sub))
    for col, color, lab in (("better_rate", "#4C9F70", "prefers better"), ("tie_rate", "#B0B0B0", "tie"),
                            ("wrong_rate", "#C8553D", "prefers worse")):
        v = pd.to_numeric(sub[col], errors="coerce").fillna(0.0).to_numpy(float)
        ax.barh(range(len(sub)), v, left=left, color=color, label=lab)
        left += v
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of the 20 controlled pairs", fontsize=8)
    ax.legend(fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"diagnostic_pairs.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # qualitative shortlist (candidates only; the student picks and quotes the minimum text)
    short = []
    diag_rows = {(str(r["problem_id"]), r["variant_type"]): r for r in read_jsonl(cfg["paths"]["task5_diagnostics"])}
    for r in read_jsonl(res / "diagnostics" / "pairs.jsonl"):
        if r["judge_outcome"] in ("wrong", "tie") and r["better_first"]:
            short.append({"kind": f"diagnostic_judge_{r['judge_outcome']}_{r['pair']}", "problem_id": r["problem_id"],
                          "first": r["first"], "second": r["second"], "verifier_pref": r["verifier_pref"],
                          "judge_pref": r["judge_pref"],
                          "first_tail": diag_rows[(r["problem_id"], r["first"])]["response"][-400:],
                          "second_tail": diag_rows[(r["problem_id"], r["second"])]["response"][-400:]})
    for ds in ("gsm", "transfer"):
        gens = {p: {g["item_id"]: g for g in read_jsonl(res / ds / f"generated_{p}.jsonl")} for p in POLICIES}
        for r in read_jsonl(res / ds / "pairwise.jsonl"):
            if r["exact_a"] != r["exact_b"] and r["judge"] in ("A", "B"):
                judged_wrong = (r["judge"] == "A") != (r["exact_a"] > r["exact_b"])
                if judged_wrong:
                    a, b = gens[r["a"]][r["item_id"]], gens[r["b"]][r["item_id"]]
                    short.append({"kind": f"{ds}_judge_prefers_incorrect", "item_id": r["item_id"], "a": r["a"], "b": r["b"],
                                  "judge": r["judge"], "gold": a["gold_final"], "pred_a": a["pred_final"], "pred_b": b["pred_final"],
                                  "question": a["question"], "a_tail": a["response"][-400:], "b_tail": b["response"][-400:]})
        for p in ("rlvr", "rlaif"):
            for iid, g in gens[p].items():
                if g["exact"] == 0 and gens["sft"][iid]["exact"] == 1 and ds == "transfer":
                    short.append({"kind": f"transfer_{p}_wrong_sft_right", "item_id": iid, "gold": g["gold_final"],
                                  "pred": g["pred_final"], "truncated": g["truncated"], "question": g["question"],
                                  "tail": g["response"][-400:]})
    write_jsonl(res / "qualitative_shortlist.jsonl", short)
    pd.set_option("display.width", 220)
    print(ind[["policy", "accuracy", "format_compliance", "len_mean", "len_sd", "truncation_rate", "ai_win_rate_vs_sft",
               "acc_diff_vs_sft", "acc_diff_ci_lo", "acc_diff_ci_hi"]].to_string(index=False))
    print(ood[["policy", "accuracy", "accuracy_drop", "ai_win_rate_vs_sft", "win_rate_drop", "len_mean"]].to_string(index=False))
    print(pairs_df.to_string(index=False))
    print(pd.DataFrame(srows).to_string(index=False))
    print(f"shortlist: {len(short)} candidates; tables -> {tab}, figures -> {fig_dir}")


if __name__ == "__main__":
    main()
