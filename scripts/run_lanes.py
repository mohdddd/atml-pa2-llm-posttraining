"""Run several command "lanes" in parallel, one GPU each (student addition, used by every task).

    python -m scripts.run_lanes --tag task1 "bash task1_dpo/run_all.sh lane0" "bash task1_dpo/run_all.sh lane1"

Lane i gets CUDA_VISIBLE_DEVICES=i (so each process sees exactly one GPU). With fewer GPUs than
lanes, the lanes run one after another on GPU 0. Every line is streamed with a [laneI] prefix and
also written to results/logs/<tag>_laneI.log. Exit code is non-zero if any lane failed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

from common.data import repo_path


def n_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return len([l for l in out.stdout.splitlines() if l.startswith("GPU ")])
    except FileNotFoundError:
        return 0


def run_lane(i: int, cmd: str, gpu: int | None, log_path, rcs: dict):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"### {time.strftime('%Y-%m-%d %H:%M:%S')} GPU={gpu} CMD: {cmd}\n")
        p = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); log.flush()
            sys.stdout.write(f"[lane{i}] {line}"); sys.stdout.flush()
        rc = p.wait()
        log.write(f"### exit={rc} after {(time.time() - t0) / 60:.1f} min\n")
    rcs[i] = rc
    print(f"[lane{i}] finished with exit code {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("lanes", nargs="+")
    args = ap.parse_args()
    logs = repo_path("results/logs"); logs.mkdir(parents=True, exist_ok=True)
    g = n_gpus()
    print(f"GPUs visible: {g}; lanes: {len(args.lanes)}", flush=True)
    rcs: dict[int, int] = {}
    if g >= len(args.lanes):
        threads = [threading.Thread(target=run_lane, args=(i, c, i, logs / f"{args.tag}_lane{i}.log", rcs)) for i, c in enumerate(args.lanes)]
        for t in threads: t.start()
        for t in threads: t.join()
    else:
        for i, c in enumerate(args.lanes):
            run_lane(i, c, 0 if g else None, logs / f"{args.tag}_lane{i}.log", rcs)
    print("lane exit codes:", rcs, flush=True)
    raise SystemExit(0 if all(rc == 0 for rc in rcs.values()) else 1)


if __name__ == "__main__":
    main()
