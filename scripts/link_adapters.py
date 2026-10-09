"""Make the frozen Task 1-3 adapters visible at the paths configs/feedback.yaml expects (student addition).

Adapters are not in git; on Kaggle they live in the saved version outputs of the Task 1 and Task 2+3
run notebooks, attached to the Task 4 notebook via Add Input -> Notebook Output. This script finds

    outputs/task1_dpo/standard   outputs/task2_ppo/standard   outputs/task3_grpo/standard

anywhere under --search-root (default /kaggle/input), requires exactly one match each, symlinks it into
this repository's git-ignored outputs/ directory, and checks that each adapter is a trained LoRA
adapter for the course base model (non-zero LoRA-B weights). Fork adapters (outputs/*/forks/*) and
preflight adapters (outputs/preflight/*) never match. A provenance record (sha256 of every adapter file,
source path, training summary if present) is written to results/task4_safety/adapters.json.

    python -m scripts.link_adapters                      # Kaggle
    python -m scripts.link_adapters --search-root /some/dir --results-dir results/preflight
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

from common.data import load_yaml, repo_path

NAMES = ("dpo", "ppo", "grpo")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lora_b_norm(adapter_dir: Path) -> float:
    from safetensors.torch import load_file

    w = load_file(str(adapter_dir / "adapter_model.safetensors"))
    b = [t.float() for k, t in w.items() if "lora_B" in k]
    if not b:
        raise RuntimeError(f"{adapter_dir}: no lora_B tensors")
    return float(sum((t ** 2).sum() for t in b) ** 0.5)


def find_one(rel: str, root: str) -> Path:
    pattern = os.path.join(root, "**", rel, "adapter_config.json")
    hits = sorted({str(Path(p).resolve().parent) for p in glob.glob(pattern, recursive=True)})
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one '{rel}' under {root}, found {len(hits)}: {hits}\n"
                         "-> check that the right notebook VERSIONS are attached as inputs (see README Task 4).")
    return Path(hits[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--search-root", default="/kaggle/input")
    ap.add_argument("--results-dir", default=None, help="default: results_dir of the config")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    record = {}
    for name in NAMES:
        rel = cfg["policies"][name]                       # e.g. outputs/task1_dpo/standard
        dest = repo_path(rel)
        if (dest / "adapter_config.json").exists():
            src = dest.resolve()
            print(f"[{name}] already present: {dest} -> {src}")
        else:
            src = find_one(rel, args.search_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src, target_is_directory=True)
            print(f"[{name}] linked {dest} -> {src}")
        acfg = json.loads((dest / "adapter_config.json").read_text())
        base = acfg.get("base_model_name_or_path")
        if base != cfg["base_model"] and not cfg.get("smoke"):
            raise SystemExit(f"[{name}] adapter base model {base!r} != {cfg['base_model']!r}")
        norm = lora_b_norm(dest)
        if norm == 0.0:
            raise SystemExit(f"[{name}] all LoRA-B weights are zero: untrained adapter")
        files = {p.name: sha256(p) for p in sorted(dest.iterdir()) if p.is_file() and p.suffix in {".json", ".safetensors"}}
        summary = None
        for cand in src.parents:                          # .../<repo>/outputs/taskX/standard -> .../<repo>
            s = cand / "results" / rel.split("/")[1] / "runs" / "standard" / "train_summary.json"
            if s.exists():
                summary = json.loads(s.read_text())
                break
        record[name] = {"config_path": rel, "source": str(src), "base_model": base, "lora_r": acfg.get("r"),
                        "target_modules": acfg.get("target_modules"), "lora_B_frobenius_norm": round(norm, 6),
                        "sha256": files, "train_summary_found_next_to_adapter": summary is not None}
        print(f"[{name}] base={base} r={acfg.get('r')} |B|={norm:.4f} sha256(adapter)={files.get('adapter_model.safetensors', '')[:12]}")
    out = repo_path(args.results_dir or cfg["results_dir"]) / "task4_safety" / "adapters.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
