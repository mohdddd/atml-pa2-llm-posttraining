"""Task 2, Step 3 — KL-pressure (reward-overoptimisation) study.

Matched short continuations from the identical PPO midpoint for kl_beta in {0, 0.10, 0.20}
(fork_updates each, clip_epsilon = 0.20, same prompt schedule and sampling seeds), each followed by the
common held-out evaluation. kl_beta = 0.10 at epsilon = 0.20 is exactly the epsilon = 0.20 clipping fork,
so by default only the other two betas are trained here and the shared fork is reused
(`--betas` overrides this).

    python -m task2_ppo.ablate_kl --config configs/ppo.yaml               # betas 0.0 and 0.20
    python -m task2_ppo.ablate_kl --config configs/ppo.yaml --betas 0.10  # the shared fork, if run alone
"""
from __future__ import annotations

import argparse

from common.rl import add_common_args, load_config, override_args, run_module
from task2_ppo.analyze_clipping import fork_name


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--betas", type=float, nargs="*")
    args = ap.parse_args()
    path = args.config or "configs/ppo.yaml"
    cfg = load_config(path, args.set)
    eps = float(cfg["clip_epsilon"])
    shared = float(cfg["kl_beta"])
    betas = args.betas if args.betas else [b for b in map(float, cfg["kl_values"]) if abs(b - shared) > 1e-12]
    root = cfg["output"].rsplit("/", 1)[0]
    extra = override_args(cfg) + (["--overwrite"] if args.overwrite else [])
    print(f"KL forks: betas={betas} at eps={eps}; beta={shared} is the shared fork {fork_name(eps, shared)}", flush=True)
    for beta in betas:
        name = fork_name(eps, beta)
        out = f"{root}/forks/{name}"
        run_module("task2_ppo.continue_train", "--config", path, "--run-name", f"fork_{name}",
                   "--updates", str(cfg["fork_updates"]), "--clip-epsilon", str(eps), "--kl-beta", str(beta),
                   "--output", out, *extra)
        run_module("task2_ppo.evaluate", "--config", path, "--adapter", out, "--name", f"fork_{name}", *extra)


if __name__ == "__main__":
    main()
