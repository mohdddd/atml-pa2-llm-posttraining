from __future__ import annotations

import argparse
import math
import shutil
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.experiment import peak_vram_gib, reset_peak_vram, run_metadata
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss
from task1_dpo.utils import add_common_args, eligible_pairs, load_task_config, row_id


def make_collate(tokenizer, max_length):
    """Chosen and rejected sequences of a microbatch are padded together and run in one forward:
    rows [0, n) are chosen, rows [n, 2n) are rejected."""
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        batch = pad_batch(tokenizer, chosen + rejected)
        batch["n_pairs"] = len(rows)
        return batch
    return collate


def select_rows(cfg, tokenizer, dataset_key: str, max_examples: int | None):
    """Eligibility filter first, then the first `max_examples` eligible rows (so every beta fork
    trains on exactly the same number and identity of pairs)."""
    rows = read_jsonl(cfg["paths"][dataset_key])
    kept, dropped = eligible_pairs(rows, tokenizer, int(cfg["max_prompt_tokens"]))
    if max_examples is not None:
        kept = kept[: int(max_examples)]
    return kept, dropped, len(rows)


def prepare_dpo_run(cfg: dict, dataset_key: str = "dpo_standard_train", beta: float | None = None, max_examples: int | None = None):
    seed = int(cfg["seed"])
    tokenizer = load_tokenizer(cfg["base_model"])
    rows, dropped, n_file = select_rows(cfg, tokenizer, dataset_key, max_examples)

    set_seed(seed)                                   # identical LoRA initialisation for every run
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    gen = torch.Generator().manual_seed(seed)        # identical data order for every run
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=gen,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg, "rows": rows, "dropped": dropped, "n_file": n_file,
        "tokenizer": tokenizer, "model": model, "loader": loader, "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(cfg: dict, run_name: str, dataset_key: str = "dpo_standard_train", output_path: str | None = None,
                 beta: float | None = None, max_examples: int | None = None, overwrite: bool = False):
    output = repo_path(output_path or cfg["standard_output"])
    run_dir = repo_path(cfg["results_dir"]) / "runs" / run_name
    if (output / "train_summary.json").exists() and not overwrite:
        print(f"[{run_name}] finished run found at {output}; skipping (pass --overwrite to retrain).")
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)                       # an unfinished earlier attempt: start the log clean
    run_dir.mkdir(parents=True, exist_ok=True)

    b = prepare_dpo_run(cfg, dataset_key, beta, max_examples)
    model, tok, loader, opt, beta = b["model"], b["tokenizer"], b["loader"], b["optimizer"], b["beta"]
    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg.get("epochs", 1))
    device = next(model.parameters()).device
    params = trainable_parameters(model)
    use_scaler = torch.cuda.is_available() and next(model.parameters()).dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    save_json(run_dir / "train_ids.json", {"dataset": cfg["paths"][dataset_key], "n_in_file": b["n_file"],
                                           "n_used": len(b["rows"]), "used_ids": [row_id(r) for r in b["rows"]],
                                           "dropped_by_prompt_rule": b["dropped"]})
    n_micro = len(loader)
    n_steps = epochs * math.ceil(n_micro / accum)
    print(f"[{run_name}] beta={beta} pairs={len(b['rows'])} (dropped {len(b['dropped'])}) "
          f"microbatches/epoch={n_micro} optimizer_steps={n_steps} fp16_scaler={use_scaler}", flush=True)

    reset_peak_vram()
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    step = 0
    acc_stats: list[dict] = []
    for epoch in range(epochs):
        for mb, batch in enumerate(loader):
            group_start = (mb // accum) * accum
            group_size = min(accum, n_micro - group_start)     # last group may be partial
            n = batch.pop("n_pairs")
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad(), reference_mode(model):
                ref_logp, _, _ = response_sequence_logprobs(model, batch)
            pol_logp, _, _ = response_sequence_logprobs(model, batch)
            loss, st = dpo_loss(pol_logp[:n], pol_logp[n:], ref_logp[:n], ref_logp[n:], beta)
            scaler.scale(loss / group_size).backward()
            acc_stats.append({"loss": loss.item(), **{k: float(v) for k, v in st.items()}})

            if (mb + 1) % accum == 0 or (mb + 1) == n_micro:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
                scale_before = scaler.get_scale() if use_scaler else 1.0
                scaler.step(opt)
                scaler.update()
                skipped = use_scaler and scaler.get_scale() < scale_before
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"step": step, "epoch": epoch, "microbatches": len(acc_stats),
                       **{k: sum(s[k] for s in acc_stats) / len(acc_stats) for k in acc_stats[0]},
                       "grad_norm": float(grad_norm), "skipped_nonfinite": bool(skipped),
                       "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.perf_counter() - t0, 1)}
                append_jsonl(run_dir / "train_log.jsonl", rec)
                acc_stats = []
                if step == 1 or step % 10 == 0 or step == n_steps:
                    print(f"[{run_name}] step {step}/{n_steps} loss={rec['loss']:.4f} acc={rec['preference_accuracy']:.3f} "
                          f"margin={rec['dpo_margin_mean']:.3f} gnorm={rec['grad_norm']:.3f} t={rec['elapsed_s']}s "
                          f"vram={peak_vram_gib()}GiB", flush=True)

    wall = time.perf_counter() - t0
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tok.save_pretrained(str(output))
    summary = run_metadata(cfg, run_name=run_name, dataset=cfg["paths"][dataset_key], beta=beta,
                           n_pairs=len(b["rows"]), optimizer_steps=step, epochs=epochs,
                           wall_clock_s=round(wall, 1), peak_vram_gib=peak_vram_gib(), output=str(output.relative_to(repo_path("."))))
    save_json(run_dir / "train_summary.json", summary)
    save_json(output / "train_summary.json", summary)     # completion marker next to the adapter
    print(f"[{run_name}] done in {wall/60:.1f} min, peak VRAM {peak_vram_gib()} GiB -> {output}", flush=True)
    del model, opt, loader, b
    clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset", default="dpo_standard_train", help="key under paths: in the config")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--overwrite", action="store_true")
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_task_config(args.config, smoke=args.smoke, overrides=args.set)
    max_examples = args.max_examples or (cfg.get("train_examples") if args.smoke else None)
    output = args.output or (cfg["standard_output"] if args.run_name == "standard" else f"{cfg['standard_output'].rsplit('/', 1)[0]}/{args.run_name}")
    run_training(cfg, args.run_name, args.dataset, output, args.beta, max_examples, args.overwrite)


if __name__ == "__main__":
    main()
