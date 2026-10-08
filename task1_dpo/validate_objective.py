"""Validate the DPO objective (and the log-prob plumbing it relies on) before training.

Checks
 1. dpo_loss matches the manual's equation computed independently, on random inputs.
 2. When policy == reference the loss is log 2 for any beta (the released starter fails this).
 3. Gradient signs: the loss pushes log pi(y+|x) up and log pi(y-|x) down.
 4. (--with-model) sequence log-probs from left-padded batches equal unpadded per-sequence
    computation, and the lean generation log-prob helper equals the starter's helper.
Writes results/task1_dpo/objective_validation.json.  Runtime: CPU, < 1 min (+2 min with model).
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from common.logging_utils import save_json
from task1_dpo.dpo import dpo_loss


def manual_loss(pc, pr, rc, rr, beta):
    """The manual's equation, written term by term."""
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()


def starter_loss(pc, pr, rc, rr, beta):
    """The released starter's (defective) logits, kept only to document the defect."""
    return -F.logsigmoid(beta * ((pc - pr) + (rc - rr))).mean()


def objective_checks():
    g = torch.Generator().manual_seed(0)
    res = {}
    pc, pr, rc, rr = (-200 * torch.rand(64, generator=g, dtype=torch.float64) for _ in range(4))
    for beta in (0.03, 0.10, 0.30):
        ours, _ = dpo_loss(pc, pr, rc, rr, beta)
        res[f"matches_manual_beta_{beta}"] = abs(ours.item() - manual_loss(pc, pr, rc, rr, beta).item()) < 1e-12
        res[f"starter_minus_manual_beta_{beta}"] = starter_loss(pc, pr, rc, rr, beta).item() - manual_loss(pc, pr, rc, rr, beta).item()
    # pi == ref  ->  loss must be log 2
    ours_eq, _ = dpo_loss(rc, rr, rc, rr, 0.1)
    res["loss_at_pi_eq_ref"] = ours_eq.item()
    res["starter_loss_at_pi_eq_ref"] = starter_loss(rc, rr, rc, rr, 0.1).item()
    res["log2"] = math.log(2)
    res["pi_eq_ref_gives_log2"] = abs(ours_eq.item() - math.log(2)) < 1e-12
    # gradient directions
    pc_ = pc.clone().requires_grad_(); pr_ = pr.clone().requires_grad_()
    loss, _ = dpo_loss(pc_, pr_, rc, rr, 0.1)
    loss.backward()
    res["grad_raises_chosen"] = bool((pc_.grad < 0).all())
    res["grad_lowers_rejected"] = bool((pr_.grad > 0).all())
    return res


def model_checks(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common.data import encode_prompt_response, pad_batch
    from common.generation import response_sequence_logprobs, response_token_logprobs, response_token_logprobs_lean

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    msgs = [[{"role": "user", "content": "Name a prime number."}],
            [{"role": "user", "content": "Explain in two sentences why the sky looks blue during the day."}]]
    resps = ["Seven is prime.", "Air molecules scatter short blue wavelengths more strongly than red ones. So scattered blue light reaches us from all directions."]
    enc = [encode_prompt_response(tok, m, r, 256) for m, r in zip(msgs, resps)]
    with torch.no_grad():
        batched, _, _ = response_sequence_logprobs(model, pad_batch(tok, enc))
        single = torch.cat([response_sequence_logprobs(model, pad_batch(tok, [e]))[0] for e in enc])
        # lean helper vs starter helper on a generated-style batch
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        e = tok(prompts, return_tensors="pt", padding=True)
        resp = tok(resps, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)["input_ids"]
        seq = torch.cat([e["input_ids"], resp], 1)
        attn = torch.cat([e["attention_mask"], torch.ones_like(resp)], 1)
        pw = e["input_ids"].shape[1]
        a, _ = response_token_logprobs(model, seq, attn, pw, resp)
        b, ent = response_token_logprobs_lean(model, seq, attn, pw, resp, with_entropy=True)
    return {
        "padded_vs_unpadded_max_abs_diff": float((batched - single).abs().max()),
        "lean_vs_starter_token_logp_max_abs_diff": float((a - b).abs().max()),
        "entropy_finite_nonneg": bool(torch.isfinite(ent).all() and (ent >= -1e-4).all()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-model", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="results/task1_dpo/objective_validation.json")
    args = ap.parse_args()
    res = {"objective": objective_checks()}
    if args.with_model:
        res["logprob_plumbing"] = model_checks(args.model)
    for sec, d in res.items():
        for k, v in d.items():
            print(f"{sec:18s} {k:45s} {v}")
    save_json(args.out, res)
    ok = all(v for k, v in res["objective"].items() if k.startswith(("matches", "pi_eq", "grad")))
    if "logprob_plumbing" in res:
        p = res["logprob_plumbing"]
        ok = ok and p["padded_vs_unpadded_max_abs_diff"] < 1e-3 and p["lean_vs_starter_token_logp_max_abs_diff"] < 1e-4
    print("ALL CHECKS PASSED" if ok else "CHECK FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
