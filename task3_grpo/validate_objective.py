"""Validate the GRPO objective helpers against the manual before any run (CPU, seconds).

Checks
 1. group_relative_advantages equals A_k = (r_k - mu_r)/(sigma_r + eps) computed separately inside each
    prompt group, for batches holding several groups with different reward scales (interleaved ids too);
    each group's advantages have mean 0. The released starter normalised over the whole batch, so a
    group whose rewards are all identical received non-zero advantages and groups leaked into each other.
 2. A zero-variance group gets all-zero advantages (uninformative group).
 3. grpo_policy_loss ('grpo') equals the manual's -mean_k[(1/T_k) sum_t min(rho A, clip(rho) A)] + beta*KL
    written independently; 'dr_grpo' replaces 1/T_k by 1/max_completion_length.
 4. The KL term is the k3 estimator exp(ref-pi) - (ref-pi) - 1 >= 0, zero when pi = ref.
 5. mask_truncated_sequences zeroes exactly the truncated completions.
Writes results/task3_grpo/objective_validation.json.
"""
from __future__ import annotations

import torch

from common.logging_utils import save_json
from common.metrics import masked_mean
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def manual_adv(rewards, gids, eps=1e-6):
    out = torch.zeros_like(rewards)
    for gid in set(gids.tolist()):
        idx = [i for i, x in enumerate(gids.tolist()) if x == gid]
        r = rewards[idx]
        mu = r.sum() / len(idx)
        sigma = torch.sqrt(((r - mu) ** 2).sum() / len(idx))
        out[idx] = (r - mu) / (sigma + eps)
    return out


def starter_adv(rewards, eps=1e-6):
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(eps)


def manual_loss(new, old, adv, mask, ref, eps, beta, denom):
    rho = torch.exp(new - old)
    a = adv[:, None]
    obj = torch.minimum(rho * a, torch.clamp(rho, 1 - eps, 1 + eps) * a)
    per_seq = (obj * mask).sum(-1) / denom
    d = ref - new
    kl = ((torch.exp(d) - d - 1) * mask).sum() / mask.sum()
    return -per_seq.mean() + beta * kl


def main():
    g = torch.Generator().manual_seed(0)
    res = {}
    rewards = torch.tensor([0.1, 0.5, 0.3, 0.9, 10.0, 12.0, 11.0, 15.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    gids = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    ours = group_relative_advantages(rewards, gids).double()
    res["advantages_vs_manual_max_abs"] = float((ours - manual_adv(rewards, gids)).abs().max())
    res["advantages_group_means"] = [float(ours[gids == i].mean()) for i in range(3)]
    res["zero_variance_group_advantages"] = ours[gids == 2].tolist()
    res["starter_zero_variance_group_advantages"] = starter_adv(rewards)[gids == 2].tolist()
    res["starter_minus_manual_max_abs"] = float((starter_adv(rewards) - manual_adv(rewards, gids)).abs().max())
    perm = torch.randperm(12, generator=g)
    ours_p = group_relative_advantages(rewards[perm], gids[perm]).double()
    res["interleaved_groups_max_abs_diff"] = float((ours_p - ours[perm]).abs().max())
    one = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    res["single_group_equals_starter"] = float((group_relative_advantages(one, torch.zeros(4)).double() - starter_adv(one)).abs().max()) < 1e-5

    B, T, L = 4, 10, 16
    mask = torch.ones(B, T, dtype=torch.float64); mask[0, 6:] = 0; mask[2, 3:] = 0
    old = -2 * torch.rand(B, T, generator=g, dtype=torch.float64)
    new = old + 0.6 * (torch.rand(B, T, generator=g, dtype=torch.float64) - 0.5)
    ref = old + 0.3 * (torch.rand(B, T, generator=g, dtype=torch.float64) - 0.5)
    adv = torch.randn(B, generator=g, dtype=torch.float64)
    for eps, beta in ((0.2, 0.1), (0.05, 0.0)):
        l, st = grpo_policy_loss(new, old, adv, mask, ref, eps, beta, "grpo")
        res[f"grpo_eps{eps}_beta{beta}_abs_diff"] = abs(float(l) - float(manual_loss(new, old, adv, mask, ref, eps, beta, mask.sum(-1))))
        l, st = grpo_policy_loss(new, old, adv, mask, ref, eps, beta, "dr_grpo", max_completion_length=L)
        res[f"dr_grpo_eps{eps}_beta{beta}_abs_diff"] = abs(float(l) - float(manual_loss(new, old, adv, mask, ref, eps, beta, float(L))))
    _, st = grpo_policy_loss(old, old, adv, mask, old, 0.2, 0.1)
    res["kl_zero_when_pi_eq_ref"] = float(st["sampled_kl"]) == 0.0
    _, st = grpo_policy_loss(new, old, adv, mask, ref, 0.2, 0.1)
    res["kl_nonnegative"] = float(st["sampled_kl"]) >= 0.0
    tm = mask_truncated_sequences(mask, [False, True, False, True])
    res["truncation_mask_ok"] = bool((tm[[1, 3]] == 0).all() and (tm[[0, 2]] == mask[[0, 2]]).all())

    res["all_checks_pass"] = (res["advantages_vs_manual_max_abs"] < 1e-5
                              and all(abs(m) < 1e-5 for m in res["advantages_group_means"])
                              and all(a == 0 for a in res["zero_variance_group_advantages"])
                              and res["interleaved_groups_max_abs_diff"] < 1e-9 and res["single_group_equals_starter"]
                              and all(v < 1e-9 for k, v in res.items() if k.endswith("_abs_diff"))
                              and res["kl_zero_when_pi_eq_ref"] and res["kl_nonnegative"] and res["truncation_mask_ok"])
    save_json("results/task3_grpo/objective_validation.json", res)
    print({k: v for k, v in res.items()})
    if not res["all_checks_pass"]:
        raise SystemExit("GRPO objective validation FAILED")


if __name__ == "__main__":
    main()
