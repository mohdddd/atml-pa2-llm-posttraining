"""Task 2, Step 2 — clipping study.

Stage `cached` (one process, policy only; runtime: GPU, ~10 min):
  1. Reconstruct the supplied cached batch (32 rollouts): prompt = rendered eval prompt cut to
     max_prompt_length from the right exactly as the starter generator did when the cache was made,
     response = re-tokenised text (+EOS when it terminated; lengths are checked against the cache).
  2. Advantages/returns: KL-shaped rewards (cached old/ref log-probs, kl_beta from the config, cached
     effective terminal reward), cached critic values, GAE(gamma, lambda), whitening over the batch.
  3. Step-0 geometry: ratio = pi_midpoint / pi_old(cache) for every token; clip and affected-token
     fractions and the clipped/unclipped surrogate for each epsilon on that one ratio field.
  4. For each epsilon, from the identical midpoint: `ppo_epochs` passes over the batch in shuffled
     minibatches of `cached_minibatch` rollouts (same order for every epsilon), one optimizer step per
     minibatch; per step log clip/affected fraction, surrogates, grad norm. After the last step, the
     full-batch ratio field is re-measured and evaluated under all three epsilons.
  -> results/task2_ppo/clipping/cached_study.json, cached_steps.jsonl, cached_ratio_quantiles.json

Stage `forks` (GPU): matched 8-update continuations from the midpoint for each epsilon at kl_beta=0.10,
  then held-out evaluation of each fork. The epsilon=0.20 fork is also the kl_beta=0.10 condition of the
  KL study (identical settings, run once).

    python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --stage all
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.optim import AdamW

from common.data import read_jsonl, repo_path
from common.experiment import peak_vram_gib, reset_peak_vram, run_metadata
from common.generation import response_token_logprobs_lean
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer, trainable_parameters
from common.rl import add_common_args, disable_dropout, load_config, override_args, run_module
from task2_ppo.ppo import clipping_diagnostics, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def fork_name(eps: float, beta: float) -> str:
    return f"eps{eps:.2f}_kl{beta:.2f}"


# --------------------------------------------------------------------------------------
# Cached batch
# --------------------------------------------------------------------------------------
def build_cached_batch(cfg, tok, rows):
    """Token tensors for every cached rollout (kept per rollout, B=1 forwards: no padding at all)."""
    pool = {r["prompt_id"]: r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"]) + read_jsonl(cfg["paths"]["rl_prompt_train"])}
    max_prompt = int(cfg["max_prompt_length"])
    items, n_cut = [], 0
    for r in rows:
        msgs = pool[r["prompt_id"]]["messages"]
        p_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if len(p_ids) > max_prompt:
            p_ids, n_cut = p_ids[:max_prompt], n_cut + 1     # starter generator behaviour (right truncation)
        r_ids = tok(r["response"], add_special_tokens=False)["input_ids"]
        if r["terminated_with_eos"]:
            r_ids = r_ids + [tok.eos_token_id]
        T = len(r["old_logprobs"])
        if len(r_ids) != T:
            raise ValueError(f"re-tokenised response length {len(r_ids)} != cached {T} (prompt {r['prompt_id']})")
        items.append({"prompt_id": r["prompt_id"], "prompt_ids": torch.tensor([p_ids]), "response_ids": torch.tensor([r_ids]),
                      "old": r["old_logprobs"].float()[None], "ref": r["ref_logprobs"].float()[None],
                      "values": r["values"].float()[None], "reward": float(r["effective_terminal_reward"]),
                      "raw_reward": float(r.get("raw_terminal_reward", r["effective_terminal_reward"])),
                      "clipped_at_max": bool(r.get("clipped_at_max", False))})
    return items, n_cut


def cached_advantages(cfg, items):
    """GAE advantages/returns per rollout, then whitening over all tokens of the batch."""
    beta = float(cfg["kl_beta"])
    advs, rets = [], []
    for it in items:
        mask = torch.ones_like(it["old"])
        rew = shaped_rewards(torch.tensor([it["reward"]]), it["old"], it["ref"], mask, beta)
        a, ret = compute_gae(rew, it["values"], mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
        advs.append(a); rets.append(ret)
    flat = torch.cat([a[0] for a in advs])
    mean, std = flat.mean(), flat.std(unbiased=False).clamp_min(1e-6)   # == normalize_advantages over the batch
    for it, a, ret in zip(items, advs, rets):
        it["adv"] = (a - mean) / std
        it["ret"] = ret
    ev = 1 - float((torch.cat([r[0] for r in rets]) - torch.cat([it["values"][0] for it in items])).var()
                   / torch.cat([r[0] for r in rets]).var())
    return {"adv_raw_mean": float(mean), "adv_raw_std": float(std), "critic_explained_variance": ev}


def new_logps(policy, it, device, grad=False):
    seq = torch.cat([it["prompt_ids"], it["response_ids"]], 1).to(device)
    attn = torch.ones_like(seq)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        lp, _ = response_token_logprobs_lean(policy, seq, attn, it["prompt_ids"].shape[1], it["response_ids"].to(device))
    return lp


def full_batch_geometry(policy, items, device, eps_values):
    """Ratio field of the current policy vs the cached old policy, and its clipping geometry."""
    ratios, advs = [], []
    for it in items:
        lp = new_logps(policy, it, device).float().cpu()
        ratios.append(torch.exp(lp - it["old"])[0]); advs.append(it["adv"][0])
    ratio, adv = torch.cat(ratios)[None], torch.cat(advs)[None]
    mask = torch.ones_like(ratio)
    geo = {f"{e:.2f}": clipping_diagnostics(ratio, adv, mask, e) for e in eps_values}
    log_r = torch.log(ratio)
    q = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    stats = {"n_tokens": int(ratio.numel()), "ratio_quantiles": dict(zip(map(str, q), torch.quantile(ratio[0].double(), torch.tensor(q, dtype=torch.double)).tolist())),
             "abs_log_ratio_mean": float(log_r.abs().mean()), "kl_old_new_k3": float((ratio - 1 - log_r).mean())}
    return geo, stats


def stage_cached(cfg, overwrite=False):
    out_dir = repo_path(cfg["results_dir"]) / "clipping"
    if (out_dir / "cached_study.json").exists() and not overwrite:
        print("[clip cached] results exist; skipping (pass --overwrite).", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cached_steps.jsonl").unlink(missing_ok=True)
    seed = int(cfg["seed"])
    tok = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    if cfg.get("cached_max_rollouts"):
        rows = rows[: int(cfg["cached_max_rollouts"])]
    items, n_cut = build_cached_batch(cfg, tok, rows)
    adv_info = cached_advantages(cfg, items)
    eps_values = [float(e) for e in cfg["clip_values"]]

    set_seed(seed)
    adapter = cfg["paths"]["ppo_midpoint_policy"]
    adapter = None if adapter in (None, "none") else adapter
    policy = load_policy(cfg, adapter_path=adapter, trainable=True, fresh_lora=adapter is None)
    disable_dropout(policy)
    device = next(policy.parameters()).device
    params = trainable_parameters(policy)
    init_state = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}

    reset_peak_vram()
    geo0, stats0 = full_batch_geometry(policy, items, device, eps_values)
    per_rollout_step0 = []
    for it in items:
        lp = new_logps(policy, it, device).float().cpu()
        per_rollout_step0.append({"prompt_id": it["prompt_id"], "tokens": int(lp.shape[1]),
                                  "mean_abs_logp_diff_vs_cache": float((lp - it["old"]).abs().mean())})
    print(f"[clip cached] rollouts={len(items)} prompts_cut={n_cut} step0 |log ratio| mean={stats0['abs_log_ratio_mean']:.4f} "
          f"clip fractions={ {k: round(v['clip_fraction'], 4) for k, v in geo0.items()} }", flush=True)

    mb = int(cfg.get("cached_minibatch", 8))
    epochs = int(cfg["ppo_epochs"])
    order_rng = random.Random(seed)
    orders = [order_rng.sample(range(len(items)), len(items)) for _ in range(epochs)]   # same for every epsilon
    use_scaler = torch.cuda.is_available() and next(policy.parameters()).dtype == torch.float16
    per_eps = {}
    for eps in eps_values:
        with torch.no_grad():
            for n, p in policy.named_parameters():
                if n in init_state:
                    p.copy_(init_state[n])
        opt = AdamW(params, lr=float(cfg["policy_learning_rate"]))
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        step = 0
        for e in range(epochs):
            for s in range(0, len(items), mb):
                chunk = [items[i] for i in orders[e][s:s + mb]]
                n_tok = sum(it["old"].shape[1] for it in chunk)
                ratios, advs = [], []
                loss_total = 0.0
                for it in chunk:
                    lp = new_logps(policy, it, device, grad=True)
                    old, adv = it["old"].to(device), it["adv"].to(device)
                    mask = torch.ones_like(old)
                    loss_i, ratio, _ = ppo_policy_loss(lp.float(), old, adv, mask, eps=eps)
                    loss_i = loss_i * (old.shape[1] / n_tok)        # token-level masked mean over the minibatch
                    scaler.scale(loss_i).backward()
                    loss_total += float(loss_i)
                    ratios.append(ratio[0].cpu()); advs.append(adv[0].cpu())
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
                before = scaler.get_scale() if use_scaler else 1.0
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
                r, a = torch.cat(ratios)[None], torch.cat(advs)[None]
                d = clipping_diagnostics(r, a, torch.ones_like(r), eps)
                rec = {"epsilon": eps, "step": step, "epoch": e, "rollouts": len(chunk), "tokens": n_tok,
                       "policy_loss": loss_total, **d, "grad_norm": gn,
                       "skipped_nonfinite": bool(use_scaler and scaler.get_scale() < before)}
                append_jsonl(out_dir / "cached_steps.jsonl", rec)
                print(f"[clip cached] eps={eps} step {step} loss={loss_total:.4f} clip={d['clip_fraction']:.4f} "
                      f"affected={d['affected_fraction']:.4f} gn={gn:.3f}", flush=True)
        geo, stats = full_batch_geometry(policy, items, device, eps_values)
        per_eps[f"{eps:.2f}"] = {"after_update_geometry_all_eps": geo, "after_update_ratio_stats": stats, "steps": step}

    save_json(out_dir / "cached_study.json", {
        "n_rollouts": len(items), "n_prompts_right_truncated": n_cut, "n_tokens": stats0["n_tokens"],
        "kl_beta_for_shaping": float(cfg["kl_beta"]), "gamma": float(cfg["gamma"]), "gae_lambda": float(cfg["gae_lambda"]),
        "minibatch_rollouts": mb, "ppo_epochs": epochs, "learning_rate": float(cfg["policy_learning_rate"]),
        "advantages": adv_info, "step0_geometry": geo0, "step0_ratio_stats": stats0,
        "per_epsilon": per_eps, "step0_per_rollout": per_rollout_step0,
        "meta": run_metadata(cfg, peak_vram_gib=peak_vram_gib())})
    print(f"[clip cached] wrote {out_dir}/cached_study.json", flush=True)
    del policy
    clear_gpu()


# --------------------------------------------------------------------------------------
# Forks
# --------------------------------------------------------------------------------------
def stage_forks(cfg, eps_values=None, beta=None, overwrite=False):
    eps_values = [float(e) for e in (eps_values or cfg["clip_values"])]
    beta = float(cfg["kl_beta"] if beta is None else beta)
    root = cfg["output"].rsplit("/", 1)[0]
    extra = override_args(cfg) + (["--overwrite"] if overwrite else [])
    for eps in eps_values:
        name = fork_name(eps, beta)
        out = f"{root}/forks/{name}"
        run_module("task2_ppo.continue_train", "--config", cfg["_config_path"], "--run-name", f"fork_{name}",
                   "--updates", str(cfg["fork_updates"]), "--clip-epsilon", str(eps), "--kl-beta", str(beta),
                   "--output", out, *extra)
        run_module("task2_ppo.evaluate", "--config", cfg["_config_path"], "--adapter", out, "--name", f"fork_{name}", *extra)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--stage", choices=["cached", "forks", "all"], default="all")
    ap.add_argument("--eps", type=float, nargs="*", help="subset of clip_values for the forks stage")
    args = ap.parse_args()
    path = args.config or "configs/ppo.yaml"
    cfg = load_config(path, args.set)
    cfg["_config_path"] = path
    if args.stage in ("cached", "all"):
        stage_cached(cfg, args.overwrite)
    if args.stage in ("forks", "all"):
        stage_forks(cfg, args.eps, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
