"""Held-out evaluation of one frozen PPO policy (common protocol in common/rl.py).

    python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
    python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint

Protocol: eligible prompts of data/rl_prompt_pool_eval.jsonl (rendered prompt <= max_prompt_length),
one sampled response per prompt (configs/base.yaml decoding, cap eval_max_response_length), same seed
and batch composition for every model; reward-model score, sampled-response KL to the reference
(LoRA disabled), token entropy, length, EOS/truncation.
Outputs: results/task2_ppo/eval/<name>/{generations.jsonl, summary.json}.
"""
from common.rl import evaluate_cli

if __name__ == "__main__":
    evaluate_cli("configs/ppo.yaml")
