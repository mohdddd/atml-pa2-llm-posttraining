"""Task 1 Step 2: short-run DPO forks for beta in cfg['betas'] from the original initialisation.

Every fork uses the same first `short_ablation_examples` eligible pairs of the standard training
file, the same seed (LoRA init + data order), optimiser, LoRA config and evaluation protocol;
only beta changes. Outputs: outputs/task1_dpo/beta_<b>/ and results/task1_dpo/{runs,eval}/beta_<b>/.
"""
from __future__ import annotations

import argparse

from task1_dpo.utils import add_common_args, load_task_config, run_module


def fork_name(beta: float) -> str:
    return f"beta_{beta:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--stage", choices=["train", "evaluate", "all"], default="all")
    ap.add_argument("--parts", default="heldout,generate")
    ap.add_argument("--overwrite", action="store_true")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_task_config(args.config, smoke=args.smoke, overrides=args.set)
    out_root = cfg["standard_output"].rsplit("/", 1)[0]
    n = int(cfg["short_ablation_examples"])
    print("betas:", cfg["betas"], "| examples per fork:", n, flush=True)
    for beta in cfg["betas"]:
        name, out = fork_name(beta), f"{out_root}/{fork_name(beta)}"
        ow = ["--overwrite"] if args.overwrite else []
        if args.stage in ("train", "all"):
            run_module("task1_dpo.train", "--config", args.config, "--run-name", name, "--output", out,
                       "--beta", str(beta), "--max-examples", str(n), *ow, smoke=args.smoke, overrides=args.set)
        if args.stage in ("evaluate", "all"):
            run_module("task1_dpo.evaluate", "--config", args.config, "--adapter", out, "--name", name,
                       "--beta", str(beta), "--parts", args.parts, *ow, smoke=args.smoke, overrides=args.set)


if __name__ == "__main__":
    main()
