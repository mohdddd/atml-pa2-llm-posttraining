"""Task 3, Step 2 — equal-generation group-size study on the supplied cache (CPU only, < 1 min).

Cache: 24 held-out prompts x 8 completions with reward-model scores (192 generations).

Regrouping rule (equal total generations): for each K in {2, 4, 8}, every prompt's 8 completions,
in generation_index order, are split into 8/K disjoint groups of K consecutive completions. Every K
therefore uses all 192 generations: 96 / 48 / 24 groups. A robustness version averages each statistic
over ALL C(8, K) subsets of a prompt's completions, which removes the dependence on generation order.

Per group: population reward std sigma; informative if sigma > tol (tol = 1e-6, the eps of the
released advantage helper); advantages A = (r - mu) / (sigma + eps) exactly as the fixed helper.
Reported per K (all prompts and per difficulty bin):
  informative_group_rate, uninformative_group_fraction, nonzero_advantage_fraction (share of generations
  that receive a non-zero advantage), mean_group_reward_std, advantage_variance (variance of the
  normalised group-relative signal over all generations), centered_reward_variance (mean within-group
  variance of r - mu_group), baseline_mse ((mu_group - mu_prompt8)^2, the error of the K-sample baseline
  against the 8-sample prompt mean), sign_agreement (fraction of generations whose sign(r - mu_group)
  equals sign(r - mu_prompt8)), and informative rates at practical thresholds sigma > tau.
Difficulty bins (defined once): terciles of the 8-completion mean reward of each prompt
  -> low-reward ("hard"), mid, high-reward ("easy"), 8 prompts each.
Low-resolution diagnostic: the same statistics after binarising every reward at the median of all 192
cached rewards (reward > median -> 1), to show how a coarse reward changes group informativeness.

    python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
-> results/task3_grpo/group_size/group_size_study.json, groups.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations

import numpy as np

from common.data import read_jsonl, repo_path
from common.experiment import run_metadata
from common.logging_utils import save_json
from common.rl import add_common_args, load_config

TOL = 1e-6
TAUS = [0.10, 0.25, 0.50]
BIN_NAMES = ["low_reward_hard", "mid", "high_reward_easy"]


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Disjoint consecutive K-groups inside each prompt (generation_index order). Every K uses all
    cached completions, so the total number of generations is identical across K."""
    if 8 % k:
        raise ValueError("K must divide 8")
    groups = []
    for pid, comps in by_prompt.items():
        comps = comps[:8]
        for j in range(0, 8, k):
            groups.append({"prompt": pid, "members": comps[j:j + k], "slot": j // k})
    return groups


def group_stats(rewards: np.ndarray, prompt_mean: float):
    mu = rewards.mean()
    sd = rewards.std()                       # population std, as in the advantage helper
    adv = (rewards - mu) / (sd + TOL)
    centered = rewards - mu
    ref = rewards - prompt_mean
    nz = (np.abs(centered) > 0) & (np.abs(ref) > 0)
    return {"std": float(sd), "informative": bool(sd > TOL), "adv": adv, "centered_var": float(centered.var()),
            "baseline_sq_err": float((mu - prompt_mean) ** 2),
            "sign_agree": (np.sign(centered[nz]) == np.sign(ref[nz])).tolist()}


def aggregate(stats_list, n_gen_per_group):
    if not stats_list:
        return None
    adv = np.concatenate([s["adv"] for s in stats_list])
    sa = [x for s in stats_list for x in s["sign_agree"]]
    stds = np.array([s["std"] for s in stats_list])
    inf = np.array([s["informative"] for s in stats_list])
    out = {"n_groups": len(stats_list), "n_generations": int(len(stats_list) * n_gen_per_group),
           "informative_group_rate": float(inf.mean()), "uninformative_group_fraction": float(1 - inf.mean()),
           "nonzero_advantage_fraction": float(np.mean(np.abs(adv) > 0)),
           "mean_group_reward_std": float(stds.mean()), "sd_of_group_reward_std": float(stds.std()),
           "advantage_variance": float(adv.var()),
           "centered_reward_variance": float(np.mean([s["centered_var"] for s in stats_list])),
           "baseline_mse": float(np.mean([s["baseline_sq_err"] for s in stats_list])),
           "sign_agreement": float(np.mean(sa)) if sa else None}
    for t in TAUS:
        out[f"informative_rate_std_gt_{t}"] = float((stds > t).mean())
    return out


def study(by_prompt, reward_of, bins_of, exhaustive=False):
    """reward_of(completion) -> float. Returns {K: {"all": ..., bin: ...}}."""
    prompt_mean = {pid: float(np.mean([reward_of(c) for c in comps[:8]])) for pid, comps in by_prompt.items()}
    res = {}
    for k in (2, 4, 8):
        per_bin = defaultdict(list)
        if exhaustive:
            for pid, comps in by_prompt.items():
                r8 = np.array([reward_of(c) for c in comps[:8]])
                for subset in combinations(range(8), k):
                    s = group_stats(r8[list(subset)], prompt_mean[pid])
                    per_bin["all"].append(s); per_bin[bins_of[pid]].append(s)
        else:
            for grp in regroup_equal_generation_budget(by_prompt, k):
                r = np.array([reward_of(c) for c in grp["members"]])
                s = group_stats(r, prompt_mean[grp["prompt"]])
                per_bin["all"].append(s); per_bin[bins_of[grp["prompt"]]].append(s)
        res[str(k)] = {b: aggregate(v, k) for b, v in per_bin.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config or "configs/grpo.yaml", args.set)
    by_prompt = load_k8_cache(cfg["group_cache"])
    out_dir = repo_path(cfg["results_dir"]) / "group_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    means = {pid: float(np.mean([c["reward"] for c in comps[:8]])) for pid, comps in by_prompt.items()}
    edges = [float(np.percentile(list(means.values()), 100 / 3)), float(np.percentile(list(means.values()), 200 / 3))]
    bins_of = {pid: BIN_NAMES[0] if m <= edges[0] else (BIN_NAMES[1] if m <= edges[1] else BIN_NAMES[2]) for pid, m in means.items()}
    all_r = np.array([c["reward"] for comps in by_prompt.values() for c in comps[:8]])
    median = float(np.median(all_r))

    result = {
        "n_prompts": len(by_prompt), "n_generations": int(len(all_r)), "tolerance": TOL, "taus": TAUS,
        "regrouping_rule": "disjoint consecutive groups of K within each prompt's 8 completions (generation_index order); all 192 generations used for every K",
        "difficulty_rule": "terciles of each prompt's mean reward over its 8 cached completions",
        "difficulty_edges": edges, "bin_sizes": {b: sum(v == b for v in bins_of.values()) for b in BIN_NAMES},
        "cache_truncated_completions": int(sum(c.get("clipped_at_max", False) for comps in by_prompt.values() for c in comps[:8])),
        "reward_summary": {"mean": float(all_r.mean()), "std": float(all_r.std()), "min": float(all_r.min()), "max": float(all_r.max()),
                           "n_distinct": int(len(np.unique(all_r)))},
        "rm_reward": {"disjoint": study(by_prompt, lambda c: c["reward"], bins_of),
                      "exhaustive_subsets": study(by_prompt, lambda c: c["reward"], bins_of, exhaustive=True)},
        "binarized_at_median": {"median": median,
                                "disjoint": study(by_prompt, lambda c: float(c["reward"] > median), bins_of),
                                "exhaustive_subsets": study(by_prompt, lambda c: float(c["reward"] > median), bins_of, exhaustive=True)},
        "prompts": [{"source_index": pid, "prompt_id": comps[0]["prompt_id"], "mean_reward": means[pid], "bin": bins_of[pid],
                     "rewards": [c["reward"] for c in comps[:8]], "completion_tokens": [c["completion_tokens"] for c in comps[:8]]}
                    for pid, comps in by_prompt.items()],
        "meta": run_metadata(cfg),
    }
    save_json(out_dir / "group_size_study.json", result)
    with open(out_dir / "groups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "source_index", "bin", "slot", "rewards", "std", "informative"])
        for k in (2, 4, 8):
            for grp in regroup_equal_generation_budget(by_prompt, k):
                r = np.array([c["reward"] for c in grp["members"]])
                w.writerow([k, grp["prompt"], bins_of[grp["prompt"]], grp["slot"], " ".join(f"{x:.4f}" for x in r),
                            f"{r.std():.6f}", int(r.std() > TOL)])
    for k in ("2", "4", "8"):
        a = result["rm_reward"]["disjoint"][k]["all"]; bz = result["binarized_at_median"]["disjoint"][k]["all"]
        print(f"[K={k}] groups={a['n_groups']} informative={a['informative_group_rate']:.3f} std={a['mean_group_reward_std']:.3f} "
              f"adv_var={a['advantage_variance']:.3f} baseline_mse={a['baseline_mse']:.3f} sign_agree={a['sign_agreement']:.3f} "
              f"| binarized informative={bz['informative_group_rate']:.3f}", flush=True)
    print(f"wrote {out_dir}/group_size_study.json", flush=True)


if __name__ == "__main__":
    main()
