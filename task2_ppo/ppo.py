from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions.

    rewards, values, mask: [batch, response_steps]. Padding positions must have mask=0.
    The final valid response position bootstraps with zero.
    """
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward."""
    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Return PPO clipped policy loss and diagnostics.

    Validate this starter implementation against the clipped surrogate in the assignment manual.
    """
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage

    # Defect fix: the starter took torch.maximum, which keeps the *optimistic* branch and removes the
    # trust region. The manual's clipped surrogate is the pessimistic bound min(rho*A, clip(rho)*A).
    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(affected, mask)
    return loss, ratio.detach(), clip_fraction.detach()


def clipping_diagnostics(ratio, advantage, mask, eps):
    """Clipping geometry of one batch (student addition), all over the same response mask.

    clip_fraction     : tokens whose ratio lies outside [1-eps, 1+eps] before clipping (manual definition)
    affected_fraction : tokens where the clipped branch is the active minimum and has zero gradient,
                        i.e. (A > 0 and rho > 1+eps) or (A < 0 and rho < 1-eps)
    surrogate_clipped / surrogate_unclipped : masked means of min(rho*A, clip(rho)*A) and rho*A
    """
    ratio, advantage = ratio.detach(), advantage.detach()
    outside = (ratio < 1.0 - eps) | (ratio > 1.0 + eps)
    affected = ((advantage > 0) & (ratio > 1.0 + eps)) | ((advantage < 0) & (ratio < 1.0 - eps))
    s1 = ratio * advantage
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage
    return {
        "clip_fraction": float(masked_mean(outside.float(), mask)),
        "affected_fraction": float(masked_mean(affected.float(), mask)),
        "surrogate_clipped": float(masked_mean(torch.minimum(s1, s2), mask)),
        "surrogate_unclipped": float(masked_mean(s1, mask)),
    }


def value_mse_loss(predicted_values, returns, mask):
    return masked_mean((predicted_values - returns) ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask
