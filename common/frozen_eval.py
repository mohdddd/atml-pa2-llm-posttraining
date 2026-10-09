"""Deterministic generation from a frozen policy, shared by Tasks 4 and 5 (student addition).

Uses the starter `batch_generate` with do_sample=False. Prompts are processed in the fixed file order
in fixed-size batches, so every policy sees identical batches. Prompts must fit `max_prompt_length`
rendered tokens: the starter generator truncates longer prompts from the right (removing the
assistant header), so an over-long prompt stops the run instead of being silently cut.
"""
from __future__ import annotations

import time

from common.experiment import peak_vram_gib, reset_peak_vram
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer


def check_prompt_fit(tokenizer, prompts: list[list[dict]], max_prompt_length: int) -> int:
    lens = [len(tokenizer.apply_chat_template(p, tokenize=True, add_generation_prompt=True)) for p in prompts]
    too_long = [i for i, n in enumerate(lens) if n > max_prompt_length]
    if too_long:
        raise SystemExit(f"{len(too_long)} prompts exceed max_prompt_length={max_prompt_length} (first: index {too_long[0]}, "
                         f"{lens[too_long[0]]} tokens); raise the limit in the config instead of truncating.")
    return max(lens) if lens else 0


def greedy_generate(cfg: dict, adapter: str | None, prompts: list[list[dict]], max_prompt_length: int,
                    max_new_tokens: int, batch_size: int, log_prefix: str = ""):
    """Returns (records, info). records[i] = {response, response_tokens, truncated, terminated_with_eos}."""
    tok = load_tokenizer(cfg["base_model"])
    longest = check_prompt_fit(tok, prompts, max_prompt_length)
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    gen_cfg = model.generation_config.to_dict()
    reset_peak_vram()
    t0 = time.perf_counter()
    out = []
    for start in range(0, len(prompts), batch_size):
        g = batch_generate(model, tok, prompts[start:start + batch_size], max_prompt_length=max_prompt_length,
                           max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, do_sample=False)
        for text, n, trunc, eos in zip(g["responses"], g["response_lengths"], g["truncated"], g["terminated_with_eos"]):
            out.append({"response": text, "response_tokens": int(n), "truncated": bool(trunc),
                        "terminated_with_eos": bool(eos)})
        print(f"{log_prefix} {min(start + batch_size, len(prompts))}/{len(prompts)} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    info = {
        "adapter": adapter,
        "n_prompts": len(prompts),
        "longest_prompt_tokens": longest,
        "max_prompt_length": max_prompt_length,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "decoding": "greedy (do_sample=False); model generation_config logits processors still apply",
        "effective_generation_config": {k: gen_cfg.get(k) for k in
                                        ("repetition_penalty", "do_sample", "eos_token_id", "pad_token_id")},
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_vram_gib": peak_vram_gib(),
    }
    del model
    clear_gpu()
    return out, info
