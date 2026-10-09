"""PPO continuation from the supplied midpoint (Task 2, Steps 1-3).

One update = (1) sample `prompts_per_update` prompts from the fixed schedule and generate one response
each with the current policy; (2) score with the frozen reward model (minus `missing_eos_penalty` when
the response did not end with EOS); (3) compute old-policy and reference log-probs and critic values;
(4) KL-shaped token rewards, GAE advantages/returns, advantage whitening; (5) `ppo_epochs` full-batch
gradient steps on the clipped surrogate (policy) and the value MSE (critic).

    python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
    python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name eps0.05_kl0.10 \
        --updates 8 --clip-epsilon 0.05 --output outputs/task2_ppo/forks/eps0.05_kl0.10

Outputs: <output>/ (policy LoRA adapter + train_summary.json completion marker),
results/task2_ppo/runs/<run>/{train_log.jsonl, rollouts.jsonl, schedule.json, train_summary.json}.
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
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    trainable_parameters,
    value_parameter_groups,
)
from common.rl import add_common_args, disable_dropout, load_config, update_schedule
from task2_ppo.ppo import (
    clipping_diagnostics,
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def _adapter_or_none(path):
    return None if path in (None, "", "none", "None") else path


def prepare_ppo_continuation(cfg: dict):
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    adapter = _adapter_or_none(cfg["paths"]["ppo_midpoint_policy"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=True, fresh_lora=adapter is None)
    value_model = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode=cfg.get("value_train_mode", "head_only"))
    # The released critic is fp16. Keep every *trainable* critic tensor in fp32 (master weights);
    # the head is then fed fp32 hidden states (see response_values).
    for p in value_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    reward_model, reward_tokenizer = load_reward_model(cfg)
    n_drop = disable_dropout(policy) + disable_dropout(value_model)

    policy_optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
    value_optimizer = AdamW(
        value_parameter_groups(value_model, lora_lr=float(cfg["value_lora_learning_rate"]),
                               head_lr=float(cfg["value_head_learning_rate"])),
        weight_decay=0.0,
    )
    return {"cfg": cfg, "tokenizer": tokenizer, "policy": policy, "value_model": value_model,
            "reward_model": reward_model, "reward_tokenizer": reward_tokenizer,
            "policy_optimizer": policy_optimizer, "value_optimizer": value_optimizer, "dropout_disabled": n_drop}


def response_values(value_model, sequences, attention_mask, prompt_width, n_response):
    """V(s_t) for every response position t: the critic head applied to the hidden state at the
    position that predicts response token t (prompt_width - 1 + t), i.e. the state before action a_t.
    Same alignment as the policy log-probs. Head runs in fp32."""
    inner = value_model.get_base_model() if hasattr(value_model, "get_base_model") else value_model
    backbone = getattr(inner, inner.base_model_prefix)           # the transformer body (LoRA layers injected)
    hidden = backbone(input_ids=sequences, attention_mask=attention_mask, use_cache=False, return_dict=True).last_hidden_state
    hidden = hidden[:, prompt_width - 1: prompt_width - 1 + n_response, :]
    head = inner.score if hasattr(inner, "score") else inner.classifier
    return head(hidden.float()).squeeze(-1)


def explained_variance(pred, target, mask):
    m = mask.bool()
    p, t = pred[m].float(), target[m].float()
    if t.numel() < 2 or float(t.var()) == 0.0:
        return float("nan")
    return float(1.0 - (t - p).var() / t.var())


def _step(scaler, opt, params, max_norm, loss):
    """Scaled backward + unscale + clip + step. Returns (pre-clip grad norm, skipped_nonfinite)."""
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gn = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
    before = scaler.get_scale() if scaler.is_enabled() else 1.0
    scaler.step(opt)
    scaler.update()
    skipped = scaler.is_enabled() and scaler.get_scale() < before
    opt.zero_grad(set_to_none=True)
    return gn, bool(skipped)


def run_ppo(cfg: dict, run_name: str = "standard", output: str | None = None, updates: int | None = None,
            clip_epsilon: float | None = None, kl_beta: float | None = None, overwrite: bool = False):
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    run_dir = repo_path(cfg["results_dir"]) / "runs" / run_name
    if (out / "train_summary.json").exists() and not overwrite:
        print(f"[{run_name}] finished run found at {out}; skipping (pass --overwrite to retrain).", flush=True)
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    b = prepare_ppo_continuation(cfg)
    policy, value_model, tok = b["policy"], b["value_model"], b["tokenizer"]
    rm, rm_tok = b["reward_model"], b["reward_tokenizer"]
    p_opt, v_opt = b["policy_optimizer"], b["value_optimizer"]
    p_params, v_params = trainable_parameters(policy), trainable_parameters(value_model)
    U, ppu, epochs = int(cfg["updates"]), int(cfg["prompts_per_update"]), int(cfg["ppo_epochs"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    gen = cfg["generation"]
    seed = int(cfg["seed"])
    use_scaler = torch.cuda.is_available() and next(policy.parameters()).dtype == torch.float16
    p_scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    v_scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    schedule, sched_info = update_schedule(cfg, tok, U, ppu)
    save_json(run_dir / "schedule.json", {**sched_info, "updates": [[r["prompt_id"] for r in rows] for rows in schedule]})
    print(f"[{run_name}] updates={U} prompts/update={ppu} ppo_epochs={epochs} eps={eps} kl_beta={beta} "
          f"max_response={cfg['max_response_length']} fp16_scaler={use_scaler} dropout_off={b['dropout_disabled']}", flush=True)

    reset_peak_vram()
    t0 = time.perf_counter()
    gen_tokens = 0
    for u in range(U):
        tu = time.perf_counter()
        rows = schedule[u]
        prompts = [r["messages"] for r in rows]
        torch.manual_seed(seed * 1000 + u)            # same sampling stream for every fork at update u
        g = batch_generate(policy, tok, prompts, int(cfg["max_prompt_length"]), int(cfg["max_response_length"]),
                           temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
        # generate() runs under inference_mode; clone so the tensors can enter autograd
        seq, pw, rids, rmask = g["sequences"].clone(), g["prompt_width"], g["response_ids"].clone(), g["response_mask"].clone()
        attn = g["attention_mask"].clone()
        attn[:, pw:] = rmask.long()
        T = rids.shape[1]
        gen_tokens += int(rmask.sum())

        with torch.no_grad():
            old_logp, _ = response_token_logprobs_lean(policy, seq, attn, pw, rids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs_lean(policy, seq, attn, pw, rids)
            values = response_values(value_model, seq, attn, pw, T).float()
            raw = score_reward_pairs(rm, rm_tok, prompts, g["responses"], max_length=int(cfg["reward_max_length"])).float().to(seq.device)
            no_eos = torch.tensor([not e for e in g["terminated_with_eos"]], device=seq.device, dtype=torch.float32)
            task_reward = raw - float(cfg["missing_eos_penalty"]) * no_eos
            rewards = shaped_rewards(task_reward, old_logp, ref_logp, rmask, beta)
            adv, ret = compute_gae(rewards, values * rmask, rmask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
            adv_n = normalize_advantages(adv, rmask)
            ev_before = explained_variance(values, ret, rmask)

        ep_logs = []
        for e in range(epochs):
            new_logp, ent = response_token_logprobs_lean(policy, seq, attn, pw, rids, with_entropy=True)
            ent = ent.detach()
            loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, adv_n, rmask, eps=eps)
            diag = clipping_diagnostics(ratio, adv_n, rmask, eps)
            log_r = (new_logp.detach() - old_logp)
            approx_kl_old = float(masked_mean(torch.exp(log_r) - 1.0 - log_r, rmask))     # k3 estimate of KL(old||new)
            p_gn, p_skip = _step(p_scaler, p_opt, p_params, float(cfg["max_grad_norm"]), loss)

            v_new = response_values(value_model, seq, attn, pw, T)
            v_loss = value_mse_loss(v_new, ret, rmask)
            v_gn, v_skip = _step(v_scaler, v_opt, v_params, float(cfg["max_grad_norm"]), float(cfg["value_coef"]) * v_loss)
            ep_logs.append({
                "policy_loss": float(loss), "value_loss": float(v_loss),
                "clip_fraction": float(clip_frac), "affected_fraction": diag["affected_fraction"],
                "approx_kl_old_new": approx_kl_old, "ratio_max": float((ratio * rmask).max()),
                "ratio_min": float(torch.where(rmask.bool(), ratio, torch.ones_like(ratio)).min()),
                "grad_norm": p_gn, "value_grad_norm": v_gn, "policy_skipped": p_skip, "value_skipped": v_skip,
                "entropy": float(masked_mean(ent, rmask)),
            })

        lens = rmask.sum(-1)
        rec = {
            "update": u + 1,
            "prompt_ids": [r["prompt_id"] for r in rows],
            "reward_raw": float(raw.mean()), "reward_effective": float(task_reward.mean()),
            "kl_token_mean": float(masked_mean(old_logp - ref_logp, rmask)),
            "kl_seq_sum_mean": float(((old_logp - ref_logp) * rmask).sum(-1).mean()),
            "entropy": ep_logs[0]["entropy"],                       # policy that generated the batch
            "sample_entropy": float(-masked_mean(old_logp, rmask)),
            "response_length": float(lens.float().mean()),
            "eos_rate": float(np.mean(g["terminated_with_eos"])), "truncation_rate": float(np.mean(g["truncated"])),
            "value_mean": float(masked_mean(values, rmask)), "return_mean": float(masked_mean(ret, rmask)),
            "value_first_token": float(values[:, 0].mean()),
            "advantage_raw_mean": float(masked_mean(adv, rmask)), "advantage_raw_std": float(adv[rmask.bool()].std()) if int(rmask.sum()) > 1 else 0.0,
            "explained_variance": ev_before,
            # per-update aggregates over PPO epochs (epoch-level values are kept in `epochs`)
            "policy_loss": float(np.mean([x["policy_loss"] for x in ep_logs])),
            "value_loss": float(np.mean([x["value_loss"] for x in ep_logs])),
            "clip_fraction": float(np.mean([x["clip_fraction"] for x in ep_logs])),
            "affected_fraction": float(np.mean([x["affected_fraction"] for x in ep_logs])),
            "grad_norm": float(np.mean([x["grad_norm"] for x in ep_logs])),
            "grad_norm_max": float(np.max([x["grad_norm"] for x in ep_logs])),
            "approx_kl_old_new_last": ep_logs[-1]["approx_kl_old_new"],
            "skipped_steps": int(sum(x["policy_skipped"] for x in ep_logs)),
            "epochs": ep_logs,
            "generated_tokens_cum": gen_tokens,
            "update_s": round(time.perf_counter() - tu, 1), "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        append_jsonl(run_dir / "train_log.jsonl", rec)
        for j, r in enumerate(rows):
            append_jsonl(run_dir / "rollouts.jsonl", {
                "update": u + 1, "prompt_id": r["prompt_id"], "source_index": r.get("source_index"),
                "response": g["responses"][j], "response_tokens": int(lens[j]),
                "terminated_with_eos": bool(g["terminated_with_eos"][j]), "truncated": bool(g["truncated"][j]),
                "reward_raw": float(raw[j]), "reward_effective": float(task_reward[j]),
                "kl_seq_sum": float(((old_logp - ref_logp) * rmask)[j].sum())})
        print(f"[{run_name}] update {u + 1}/{U} R={rec['reward_raw']:.3f} KL={rec['kl_token_mean']:.5f} "
              f"H={rec['entropy']:.3f} len={rec['response_length']:.0f} Lpi={rec['policy_loss']:.4f} "
              f"Lv={rec['value_loss']:.3f} EV={ev_before:.2f} clip={rec['clip_fraction']:.3f} gn={rec['grad_norm']:.3f} "
              f"t={rec['update_s']}s vram={peak_vram_gib()}GiB", flush=True)

    wall = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    summary = run_metadata(cfg, run_name=run_name, updates=U, clip_epsilon=eps, kl_beta=beta,
                           generated_tokens=gen_tokens, wall_clock_s=round(wall, 1), peak_vram_gib=peak_vram_gib(),
                           gpu_count_visible=torch.cuda.device_count(), output=str(out.relative_to(repo_path("."))),
                           schedule=sched_info)
    save_json(run_dir / "train_summary.json", summary)
    save_json(out / "train_summary.json", summary)
    print(f"[{run_name}] done in {wall / 60:.1f} min, peak VRAM {peak_vram_gib()} GiB -> {out}", flush=True)
    del policy, value_model, rm, b
    clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    cfg = load_config(args.config or "configs/ppo.yaml", args.set)
    run_ppo(cfg, args.run_name, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.overwrite)


if __name__ == "__main__":
    main()
