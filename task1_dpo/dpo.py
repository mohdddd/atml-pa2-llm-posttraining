from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Manual objective:
        L = -E[ log sigmoid( beta * ( [log pi(y+|x) - log ref(y+|x)] - [log pi(y-|x) - log ref(y-|x)] ) ) ]
    which rearranges to beta * (policy_margin - ref_margin), where
        policy_margin = log pi(y+|x)  - log pi(y-|x)
        ref_margin    = log ref(y+|x) - log ref(y-|x).

    STUDENT FIX (checked by task1_dpo/validate_objective.py): the released starter computed
    beta * (policy_margin + ref_margin). Adding the reference margin instead of subtracting it is
    not the manual's objective: the loss is no longer log(2) when pi == ref, and the reference
    margin pushes the logits in the wrong direction.
    """

    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # The manual's DPO preference margin m_theta (sequence log-probs summed over response tokens).
    margin = policy_margin - ref_margin
    logits = beta * margin

    loss = -F.logsigmoid(logits).mean()

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "dpo_margin_mean": margin.detach().mean(),
        "chosen_reward_mean": (beta * (policy_chosen_logp - ref_chosen_logp)).detach().mean(),
        "rejected_reward_mean": (beta * (policy_rejected_logp - ref_rejected_logp)).detach().mean(),
        "preference_accuracy": (margin > 0).float().mean().detach(),
    }
