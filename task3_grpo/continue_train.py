"""GRPO continuation from the supplied midpoint (Task 3, Steps 1 and 3).

One update = (1) take `prompts_per_update` prompts from the fixed schedule and sample K =
`num_generations` completions each; (2) score every completion with the frozen reward model;
(3) group-relative advantages inside each prompt group; (4) completions that hit
max_completion_length are masked out of the loss (their rewards still enter the group statistics);
(5) `policy_epochs` gradient steps on the clipped objective with the k3 KL penalty to the reference.
loss_type 'grpo' divides each completion's token sum by its own length T_k; 'dr_grpo' by the constant
max_completion_length.

Length-conditioned gradient statistic (every update): for each completion k the norm of the gradient of
its own policy term (its share of the batch loss, without the KL term) w.r.t. the LoRA parameters,
computed with torch.autograd.grad on the same forward pass; logged with its length and advantage.

    python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
    python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name fork_dr_grpo \
        --updates 8 --loss-type dr_grpo --output outputs/task3_grpo/forks/dr_grpo
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import torch
from torch.optim import AdamW

from common.data import repo_path
from common.experiment import peak_vram_gib, reset_peak_vram, run_metadata
from common.generation import batch_generate, response_token_logprobs_lean, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from common.rl import add_common_args, disable_dropout, load_config, update_schedule
from task3_grpo.grpo import group_relative_advantages, group_reward_stats, grpo_policy_loss, mask_truncated_sequences

GRAD_PROBE_SCALE = 1024.0      # keeps fp16 backward of a single completion's term away from underflow


def prepare_grpo_continuation(cfg: dict):
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    adapter = cfg["paths"]["grpo_midpoint_policy"]
    adapter = None if adapter in (None, "", "none", "None") else adapter
    policy = load_policy(cfg, adapter_path=adapter, trainable=True, fresh_lora=adapter is None)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    n_drop = disable_dropout(policy)
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {"cfg": cfg, "tokenizer": tokenizer, "policy": policy, "reward_model": reward_model,
            "reward_tokenizer": reward_tokenizer, "optimizer": optimizer, "dropout_disabled": n_drop}


def per_sequence_policy_terms(new_logp, old_logp, adv, mask, eps, loss_type, max_len):
    """The policy part of grpo_policy_loss split into one term per completion (sum of terms = policy_term)."""
    ratio = torch.exp(new_logp - old_logp)
    a = adv[:, None]
    obj = torch.minimum(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a)
    tok_sum = (obj * mask).sum(-1)
    denom = mask.sum(-1).clamp_min(1.0) if loss_type == "grpo" else torch.full_like(tok_sum, float(max_len))
    return -(tok_sum / denom) / tok_sum.shape[0]


def run_grpo(cfg: dict, run_name: str = "standard", output: str | None = None, updates: int | None = None,
             loss_type: str = "grpo", overwrite: bool = False):
    if updates is not None:
        cfg["updates"] = int(updates)
    cfg["loss_type"] = loss_type
    out = repo_path(output or cfg["output"])
    run_dir = repo_path(cfg["results_dir"]) / "runs" / run_name
    if (out / "train_summary.json").exists() and not overwrite:
        print(f"[{run_name}] finished run found at {out}; skipping (pass --overwrite to retrain).", flush=True)
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    b = prepare_grpo_continuation(cfg)
    policy, tok, rm, rm_tok, opt = b["policy"], b["tokenizer"], b["reward_model"], b["reward_tokenizer"], b["optimizer"]
    params = trainable_parameters(policy)
    U, ppu, K = int(cfg["updates"]), int(cfg["prompts_per_update"]), int(cfg["num_generations"])
    epochs, eps, beta = int(cfg["policy_epochs"]), float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_len = int(cfg["max_completion_length"])
    gen = cfg["generation"]
    seed = int(cfg["seed"])
    use_scaler = torch.cuda.is_available() and next(policy.parameters()).dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    schedule, sched_info = update_schedule(cfg, tok, U, ppu)
    save_json(run_dir / "schedule.json", {**sched_info, "updates": [[r["prompt_id"] for r in rows] for rows in schedule]})
    print(f"[{run_name}] loss_type={loss_type} updates={U} prompts/update={ppu} K={K} epochs={epochs} eps={eps} "
          f"beta={beta} max_completion={max_len} mask_truncated={cfg['mask_truncated_completions']} fp16_scaler={use_scaler}", flush=True)

    reset_peak_vram()
    t0 = time.perf_counter()
    gen_tokens = 0
    for u in range(U):
        tu = time.perf_counter()
        rows = schedule[u]
        prompts = [r["messages"] for r in rows for _ in range(K)]
        group_ids = torch.tensor([i for i in range(len(rows)) for _ in range(K)])
        torch.manual_seed(seed * 1000 + u)
        g = batch_generate(policy, tok, prompts, int(cfg["max_prompt_length"]), max_len,
                           temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
        # generate() runs under inference_mode; clone so the tensors can enter autograd
        seq, pw, rids, rmask = g["sequences"].clone(), g["prompt_width"], g["response_ids"].clone(), g["response_mask"].clone()
        attn = g["attention_mask"].clone()
        attn[:, pw:] = rmask.long()
        gen_tokens += int(rmask.sum())
        tmask = mask_truncated_sequences(rmask, g["truncated"]) if cfg.get("mask_truncated_completions") else rmask

        with torch.no_grad():
            rewards = score_reward_pairs(rm, rm_tok, prompts, g["responses"], max_length=int(cfg["reward_max_length"])).float()
            adv = group_relative_advantages(rewards.cpu(), group_ids).to(seq.device)
            gstats = group_reward_stats(rewards.cpu(), group_ids)
            old_logp, _ = response_token_logprobs_lean(policy, seq, attn, pw, rids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs_lean(policy, seq, attn, pw, rids)

        ep_logs, seq_grad = [], None
        for e in range(epochs):
            new_logp, ent = response_token_logprobs_lean(policy, seq, attn, pw, rids, with_entropy=True)
            ent = ent.detach()
            loss, st = grpo_policy_loss(new_logp, old_logp, adv, tmask, ref_logp, eps, beta,
                                        loss_type=loss_type, max_completion_length=max_len)
            if e == 0:
                # length-conditioned gradient statistic on the first epoch's forward pass
                terms = per_sequence_policy_terms(new_logp, old_logp, adv, tmask, eps, loss_type, max_len)
                if abs(float(terms.sum()) - float(st["policy_term"])) > 1e-4 * max(1.0, abs(float(st["policy_term"]))):
                    raise RuntimeError("per-completion terms do not add up to the GRPO policy term")
                seq_grad = []
                for k in range(terms.shape[0]):
                    if float(tmask[k].sum()) == 0 or float(adv[k]) == 0.0:
                        seq_grad.append(0.0)
                        continue
                    grads = torch.autograd.grad(terms[k] * GRAD_PROBE_SCALE, params, retain_graph=True, allow_unused=True)
                    seq_grad.append(float(torch.sqrt(sum((gg.float() ** 2).sum() for gg in grads if gg is not None))) / GRAD_PROBE_SCALE)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
            before = scaler.get_scale() if use_scaler else 1.0
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            ep_logs.append({"loss": float(loss), "policy_loss": float(st["policy_term"]), "kl_k3": float(st["sampled_kl"]),
                            "clip_fraction": float(st["clip_fraction"]), "grad_norm": gn,
                            "skipped": bool(use_scaler and scaler.get_scale() < before),
                            "entropy_valid": float(masked_mean(ent, rmask))})

        lens = rmask.sum(-1)
        stds = [s["std"] for s in gstats]
        rec = {
            "update": u + 1, "loss_type": loss_type,
            "prompt_ids": [r["prompt_id"] for r in rows],
            "reward_mean": float(rewards.mean()),
            "kl_token_mean": float(masked_mean(old_logp - ref_logp, rmask)),     # sampled-response estimator, all valid tokens
            "kl_k3_loss_mask": ep_logs[0]["kl_k3"],
            "group_reward_std_mean": float(np.mean(stds)),
            "uninformative_group_fraction": float(np.mean([s["uninformative"] for s in gstats])),
            "policy_loss": float(np.mean([x["policy_loss"] for x in ep_logs])),
            "loss": float(np.mean([x["loss"] for x in ep_logs])),
            "grad_norm": float(np.mean([x["grad_norm"] for x in ep_logs])),
            "clip_fraction": float(np.mean([x["clip_fraction"] for x in ep_logs])),
            "entropy": ep_logs[0]["entropy_valid"],
            "sample_entropy": float(-masked_mean(old_logp, rmask)),
            "response_length": float(lens.float().mean()),
            "truncation_rate": float(np.mean(g["truncated"])), "eos_rate": float(np.mean(g["terminated_with_eos"])),
            "masked_completion_fraction": float((tmask.sum(-1) == 0).float().mean()),
            "skipped_steps": int(sum(x["skipped"] for x in ep_logs)),
            "groups": gstats, "epochs": ep_logs,
            "generated_tokens_cum": gen_tokens,
            "update_s": round(time.perf_counter() - tu, 1), "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        append_jsonl(run_dir / "train_log.jsonl", rec)
        for k in range(len(prompts)):
            append_jsonl(run_dir / "completions.jsonl", {
                "update": u + 1, "group": int(group_ids[k]), "prompt_id": rows[int(group_ids[k])]["prompt_id"],
                "response": g["responses"][k], "response_tokens": int(lens[k]), "loss_tokens": int(tmask[k].sum()),
                "terminated_with_eos": bool(g["terminated_with_eos"][k]), "truncated": bool(g["truncated"][k]),
                "reward": float(rewards[k]), "advantage": float(adv[k]),
                "seq_policy_grad_norm": seq_grad[k] if seq_grad else None,
                "kl_seq_sum": float(((old_logp - ref_logp) * rmask)[k].sum())})
        print(f"[{run_name}] update {u + 1}/{U} R={rec['reward_mean']:.3f} groupSD={rec['group_reward_std_mean']:.3f} "
              f"uninf={rec['uninformative_group_fraction']:.2f} KL={rec['kl_token_mean']:.5f} H={rec['entropy']:.3f} "
              f"len={rec['response_length']:.0f} trunc={rec['truncation_rate']:.2f} L={rec['policy_loss']:.4f} "
              f"gn={rec['grad_norm']:.3f} t={rec['update_s']}s vram={peak_vram_gib()}GiB", flush=True)

    wall = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    summary = run_metadata(cfg, run_name=run_name, updates=U, loss_type=loss_type, generated_tokens=gen_tokens,
                           wall_clock_s=round(wall, 1), peak_vram_gib=peak_vram_gib(),
                           gpu_count_visible=torch.cuda.device_count(), output=str(out.relative_to(repo_path("."))),
                           schedule=sched_info)
    save_json(run_dir / "train_summary.json", summary)
    save_json(out / "train_summary.json", summary)
    print(f"[{run_name}] done in {wall / 60:.1f} min, peak VRAM {peak_vram_gib()} GiB -> {out}", flush=True)
    del policy, rm, b
    clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    cfg = load_config(args.config or "configs/grpo.yaml", args.set)
    run_grpo(cfg, args.run_name, args.output, args.updates, args.loss_type, args.overwrite)


if __name__ == "__main__":
    main()
