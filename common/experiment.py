"""Shared experiment plumbing for all tasks: run metadata, smoke-test overrides, GPU accounting.

Student addition (not part of the course starter).
"""
from __future__ import annotations

import copy
import platform
import subprocess
import time

import torch

from common.data import REPO_ROOT

# A smoke test checks the plumbing end to end on CPU in minutes. It swaps in the smallest
# Qwen2.5 model for every network, runs in float32, and shrinks every loop. Smoke outputs go to
# separate folders so they can never be mistaken for real results.
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def apply_smoke(cfg: dict, extra: dict | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["base_model"] = SMOKE_MODEL
    cfg["reward_model"] = SMOKE_MODEL      # random scalar head: plumbing only, scores are meaningless
    cfg["reward_tokenizer"] = SMOKE_MODEL
    cfg["dtype"] = "float32"
    cfg["smoke"] = True
    for key in ("results_dir", "standard_output", "length_output", "output"):
        if key in cfg and isinstance(cfg[key], str):
            parts = cfg[key].split("/", 1)
            cfg[key] = f"{parts[0]}/smoke/{parts[1]}" if len(parts) == 2 else f"{cfg[key]}/smoke"
    if extra:
        cfg.update(extra)
    return cfg


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def run_metadata(cfg: dict, **extra) -> dict:
    import peft
    import transformers

    meta = {
        "git_commit": git_commit(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": cfg,
    }
    meta.update(extra)
    return meta


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**30, 3)
