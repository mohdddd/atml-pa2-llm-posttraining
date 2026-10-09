"""Held-out evaluation of one frozen GRPO policy (same protocol as Task 2, common/rl.py).

    python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard

Outputs: results/task3_grpo/eval/<name>/{generations.jsonl, summary.json}.
"""
from common.rl import evaluate_cli

if __name__ == "__main__":
    evaluate_cli("configs/grpo.yaml")
