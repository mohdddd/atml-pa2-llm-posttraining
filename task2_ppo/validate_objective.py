"""Validate the PPO objective helpers against the manual before any run (CPU, seconds).

Checks
 1. ppo_policy_loss equals -masked_mean(min(rho*A, clip(rho, 1-eps, 1+eps)*A)) written independently,
    for eps in {0.05, 0.20, 0.50}; the released starter (torch.maximum) does not.
 2. Trust-region behaviour on single tokens: the gradient w.r.t. log pi is zero when the clip binds
    (A>0, rho>1+eps or A<0, rho<1-eps) and equals -A*rho/N otherwise; the starter instead keeps
    pushing the ratio further out of the region (non-zero gradient in exactly those cases).
 3. At rho = 1 the loss equals -mean(A) and the clip fraction is 0.
 4. The clip fraction counts tokens outside [1-eps, 1+eps] before clipping (manual definition).
 5. compute_gae equals the direct sum A_t = sum_k (gamma*lambda)^k delta_{t+k} on random padded batches,
    and shaped_rewards puts the task reward on the last valid token with -beta*(log pi - log ref) on all.
Writes results/task2_ppo/objective_validation.json.
"""
from __future__ import annotations

import torch

from common.logging_utils import save_json
from common.metrics import masked_mean
from task2_ppo.ppo import clipping_diagnostics, compute_gae, ppo_policy_loss, shaped_rewards


def manual_loss(new, old, adv, mask, eps):
    rho = torch.exp(new - old)
    return -masked_mean(torch.minimum(rho * adv, torch.clamp(rho, 1 - eps, 1 + eps) * adv), mask)


def starter_loss(new, old, adv, mask, eps):
    rho = torch.exp(new - old)
    return -masked_mean(torch.maximum(rho * adv, torch.clamp(rho, 1 - eps, 1 + eps) * adv), mask)


def gae_direct(rewards, values, mask, gamma, lam):
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    for b in range(B):
        n = int(mask[b].sum())
        v = torch.cat([values[b, :n], torch.zeros(1, dtype=values.dtype)])
        delta = rewards[b, :n] + gamma * v[1:] - v[:-1]
        for t in range(n):
            adv[b, t] = sum((gamma * lam) ** k * delta[t + k] for k in range(n - t))
    return adv


def main():
    g = torch.Generator().manual_seed(0)
    res = {}
    B, T = 4, 12
    mask = torch.ones(B, T, dtype=torch.float64)
    mask[1, 8:] = 0; mask[3, 5:] = 0
    old = -3 * torch.rand(B, T, generator=g, dtype=torch.float64)
    new = old + 0.8 * (torch.rand(B, T, generator=g, dtype=torch.float64) - 0.5)
    adv = torch.randn(B, T, generator=g, dtype=torch.float64)
    for eps in (0.05, 0.20, 0.50):
        ours, ratio, cf = ppo_policy_loss(new, old, adv, mask, eps)
        res[f"eps_{eps}_matches_manual"] = abs(float(ours) - float(manual_loss(new, old, adv, mask, eps))) < 1e-12
        res[f"eps_{eps}_starter_minus_manual"] = float(starter_loss(new, old, adv, mask, eps) - manual_loss(new, old, adv, mask, eps))
        outside = ((ratio < 1 - eps) | (ratio > 1 + eps)).double()
        res[f"eps_{eps}_clip_fraction_definition_ok"] = abs(float(cf) - float((outside * mask).sum() / mask.sum())) < 1e-6
        res[f"eps_{eps}_diagnostics"] = clipping_diagnostics(ratio, adv, mask, eps)

    # single-token gradient cases, eps = 0.2
    eps, cases = 0.2, []
    for a, r in [(1.0, 1.5), (1.0, 0.5), (-1.0, 0.5), (-1.0, 1.5), (1.0, 1.1), (-1.0, 0.9)]:
        o = torch.zeros(1, 1, dtype=torch.float64)
        m = torch.ones(1, 1, dtype=torch.float64)
        A = torch.tensor([[a]], dtype=torch.float64)
        n1 = torch.log(torch.tensor([[r]], dtype=torch.float64)).requires_grad_()
        ppo_policy_loss(n1, o, A, m, eps)[0].backward()
        n2 = torch.log(torch.tensor([[r]], dtype=torch.float64)).requires_grad_()
        starter_loss(n2, o, A, m, eps).backward()
        binds = (a > 0 and r > 1 + eps) or (a < 0 and r < 1 - eps)
        expected = 0.0 if binds else -a * r
        cases.append({"A": a, "rho": r, "clip_binds": binds, "grad_fixed": float(n1.grad), "grad_expected": expected,
                      "grad_starter": float(n2.grad)})
    res["gradient_cases_eps_0.2"] = cases
    res["gradient_cases_all_match"] = all(abs(c["grad_fixed"] - c["grad_expected"]) < 1e-12 for c in cases)
    res["starter_grad_nonzero_where_clip_binds"] = all(abs(c["grad_starter"]) > 0 for c in cases if c["clip_binds"])

    # rho = 1
    l1, _, cf1 = ppo_policy_loss(old, old, adv, mask, 0.2)
    res["rho1_loss_equals_minus_mean_adv"] = abs(float(l1) + float(masked_mean(adv, mask))) < 1e-12
    res["rho1_clip_fraction_zero"] = float(cf1) == 0.0

    # GAE and reward shaping
    rewards = torch.randn(B, T, generator=g, dtype=torch.float64) * mask
    values = torch.randn(B, T, generator=g, dtype=torch.float64) * mask
    for gamma, lam in ((1.0, 0.95), (0.99, 0.9)):
        a1, ret = compute_gae(rewards, values, mask, gamma, lam)
        a2 = gae_direct(rewards, values, mask, gamma, lam)
        res[f"gae_gamma{gamma}_lam{lam}_max_abs_diff"] = float((a1 - a2).abs().max())
        res[f"gae_gamma{gamma}_lam{lam}_returns_ok"] = float((ret - (a1 + values)).abs().max()) < 1e-12
    pol, ref = -torch.rand(B, T, generator=g, dtype=torch.float64), -torch.rand(B, T, generator=g, dtype=torch.float64)
    task = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64)
    sr = shaped_rewards(task, pol, ref, mask, 0.1)
    exp = -0.1 * (pol - ref) * mask
    for b in range(B):
        exp[b, int(mask[b].sum()) - 1] += task[b]
    res["shaped_rewards_max_abs_diff"] = float((sr - exp).abs().max())

    res["all_checks_pass"] = (all(v for k, v in res.items() if k.endswith(("_matches_manual", "_definition_ok", "_returns_ok")))
                              and res["gradient_cases_all_match"] and res["rho1_loss_equals_minus_mean_adv"]
                              and res["rho1_clip_fraction_zero"] and res["shaped_rewards_max_abs_diff"] < 1e-12
                              and all(v < 1e-10 for k, v in res.items() if k.endswith("_max_abs_diff")))
    save_json("results/task2_ppo/objective_validation.json", res)
    print({k: v for k, v in res.items() if not isinstance(v, (list, dict))})
    if not res["all_checks_pass"]:
        raise SystemExit("PPO objective validation FAILED")


if __name__ == "__main__":
    main()
