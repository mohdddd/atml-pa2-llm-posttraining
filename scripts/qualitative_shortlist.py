"""Candidate qualitative examples for Tasks 1-3 (student addition; CPU, seconds; reads saved results only).

    python -m scripts.qualitative_shortlist

Writes results/qualitative/task{1,2,3}_shortlist.jsonl. These are CANDIDATES selected by fixed, stated rules from
already-saved per-item outputs; the student reads them, picks the examples, judges quality, and quotes the minimum
text. (Tasks 4 and 5 write their own shortlists: results/task4_safety/ and results/task5_feedback/.)
Responses are cut to 700 characters (head) + 200 characters (tail) to keep files small.

Task 1 (DPO)  (i) preference/reward vs quality: held-out pairs the standard model ranks most confidently against the
                  label; held-out pairs with the largest chosen-minus-rejected length gap; prompts where the RM score
                  of the standard model's generation differs most from SFT's (both texts shown).
              (ii) length / instruction compliance: word-limit responses of SFT / standard / length-balanced, the
                  non-compliant ones and the longest compliant ones; preferred-longer pairs that standard gets wrong
                  and length-balanced gets right (and vice versa).
Task 2 (PPO)  reward and quality move together / disagree: held-out prompts with the largest reward change
              standard vs midpoint; highest-reward truncated responses; lowest-reward complete responses;
              shortest training rollouts with the highest reward.
Task 3 (GRPO) normalisation / informativeness: training groups (canonical and Dr.GRPO forks) with the largest
              within-group length spread (all K completions with reward, advantage, length, per-completion grad
              norm); groups with the smallest reward SD; groups whose completions were all truncated.
"""
from __future__ import annotations

import json
from collections import defaultdict

from common.data import read_jsonl, repo_path, write_jsonl


def cut(t, head=700, tail=200):
    t = str(t)
    return t if len(t) <= head + tail + 20 else t[:head] + " [...] " + t[-tail:]


def user_text(messages):
    return next((m["content"] for m in messages if m.get("role") == "user"), "")


def load(p):
    p = repo_path(p)
    return read_jsonl(p) if p.exists() else []


def task1():
    out = []
    eval_rows = {f"{r['source_split']}:{r['source_index']}": r for r in load("data/dpo_standard_eval.jsonl")}
    by_pid = {r["prompt_id"]: r for r in eval_rows.values()}
    strat = {f"{r['source_split']}:{r['source_index']}": r for r in load("data/dpo_length_stratified_eval.jsonl")}
    base = "results/task1_dpo/eval"

    def pair_text(src, pid):
        r = by_pid.get(pid) or src
        if not r:
            return {}
        c, rj = r["chosen"], r["rejected"]
        c = c[-1]["content"] if isinstance(c, list) else c
        rj = rj[-1]["content"] if isinstance(rj, list) else rj
        return {"prompt": cut(r["prompt"], 500), "chosen": cut(c), "rejected": cut(rj)}

    hp = load(f"{base}/standard/heldout_pairs.jsonl")
    for r in sorted(hp, key=lambda x: x["margin"])[:6]:
        out.append({"kind": "i_heldout_confidently_against_label", "id": r["id"], "dpo_margin": round(r["margin"], 3),
                    "len_chosen": r["len_c"], "len_rejected": r["len_r"], "score_chosen": r.get("score_chosen"),
                    "score_rejected": r.get("score_rejected"), **pair_text(eval_rows.get(r["id"]), r.get("prompt_id"))})
    for r in sorted(hp, key=lambda x: -(x["len_c"] - x["len_r"]))[:4]:
        out.append({"kind": "i_heldout_largest_length_gap_chosen_longer", "id": r["id"], "dpo_margin": round(r["margin"], 3),
                    "len_chosen": r["len_c"], "len_rejected": r["len_r"], "score_chosen": r.get("score_chosen"),
                    "score_rejected": r.get("score_rejected"), **pair_text(eval_rows.get(r["id"]), r.get("prompt_id"))})
    sft = {r["id"]: r for r in load(f"{base}/sft/generations.jsonl")}
    std = load(f"{base}/standard/generations.jsonl")
    diffs = [(r, sft[r["id"]]) for r in std if r["id"] in sft and r["response"] != sft[r["id"]]["response"]
             and r.get("reward") is not None and sft[r["id"]].get("reward") is not None]
    for r, s in sorted(diffs, key=lambda x: -abs(x[0]["reward"] - x[1]["reward"]))[:6]:
        src = eval_rows.get(r["id"], {})
        out.append({"kind": "i_generation_rm_change_vs_sft", "id": r["id"], "prompt": cut(src.get("prompt", ""), 500),
                    "rm_standard": round(r["reward"], 3), "rm_sft": round(s["reward"], 3),
                    "len_standard": r["response_tokens"], "len_sft": s["response_tokens"],
                    "truncated_standard": r["truncated"], "truncated_sft": s["truncated"],
                    "response_standard": cut(r["response"]), "response_sft": cut(s["response"])})
    wl_prompts = {r["prompt_id"]: user_text(r["messages"]) for r in load("data/word_limit_prompts.jsonl")}
    for model in ("sft", "standard", "length_balanced"):
        wl = load(f"{base}/{model}/word_limit.jsonl")
        bad = [r for r in wl if not r["compliant"]]
        good = sorted([r for r in wl if r["compliant"]], key=lambda r: -r["words"] / r["word_limit"])
        for r in sorted(bad, key=lambda r: -(r["words"] - r["word_limit"]))[:3] + good[:1]:
            out.append({"kind": f"ii_word_limit_{'violation' if not r['compliant'] else 'closest_compliant'}", "model": model,
                        "id": r["id"], "prompt": wl_prompts.get(r["prompt_id"]), "word_limit": r["word_limit"],
                        "words": r["words"], "response": cut(r["response"])})
    s_std = {r["id"]: r for r in load(f"{base}/standard/stratified_pairs.jsonl")}
    s_bal = {r["id"]: r for r in load(f"{base}/length_balanced/stratified_pairs.jsonl")}
    for i, r in s_std.items():
        b = s_bal.get(i)
        if not b or r.get("length_stratum") != "preferred_longer":
            continue
        if (r["margin"] > 0) != (b["margin"] > 0):
            src = strat.get(i, {})
            c, rj = src.get("chosen", ""), src.get("rejected", "")
            c = c[-1]["content"] if isinstance(c, list) else c
            rj = rj[-1]["content"] if isinstance(rj, list) else rj
            out.append({"kind": "ii_preferred_longer_standard_vs_balanced_disagree", "id": i,
                        "standard_correct": r["margin"] > 0, "balanced_correct": b["margin"] > 0,
                        "margin_standard": round(r["margin"], 3), "margin_balanced": round(b["margin"], 3),
                        "length_difference": r.get("length_difference"), "prompt": cut(src.get("prompt", ""), 400),
                        "chosen": cut(c, 400, 100), "rejected": cut(rj, 400, 100)})
    return out


def rl_prompts():
    return {r["prompt_id"]: user_text(r["messages"]) for r in load("data/rl_prompt_pool_eval.jsonl") + load("data/rl_prompt_pool_train.jsonl")}


def task2():
    out, P = [], rl_prompts()
    mid = {r["id"]: r for r in load("results/task2_ppo/eval/midpoint/generations.jsonl")}
    std = load("results/task2_ppo/eval/standard/generations.jsonl")
    pairs = [(r, mid[r["id"]]) for r in std if r["id"] in mid and r["response"] != mid[r["id"]]["response"]]
    for r, m in sorted(pairs, key=lambda x: -abs(x[0]["reward"] - x[1]["reward"]))[:6]:
        out.append({"kind": "heldout_largest_reward_change_standard_vs_midpoint", "id": r["id"], "prompt": cut(P.get(r["id"], ""), 500),
                    "reward_standard": round(r["reward"], 3), "reward_midpoint": round(m["reward"], 3),
                    "len_standard": r["response_tokens"], "len_midpoint": m["response_tokens"],
                    "truncated_standard": r["truncated"], "truncated_midpoint": m["truncated"],
                    "response_standard": cut(r["response"]), "response_midpoint": cut(m["response"])})
    for kind, rows in (("heldout_highest_reward_truncated", sorted([r for r in std if r["truncated"]], key=lambda r: -r["reward"])[:4]),
                       ("heldout_lowest_reward_complete", sorted([r for r in std if not r["truncated"]], key=lambda r: r["reward"])[:4]),
                       ("heldout_highest_reward_complete", sorted([r for r in std if not r["truncated"]], key=lambda r: -r["reward"])[:3])):
        for r in rows:
            out.append({"kind": kind, "id": r["id"], "prompt": cut(P.get(r["id"], ""), 500), "reward": round(r["reward"], 3),
                        "len": r["response_tokens"], "response": cut(r["response"])})
    ro = load("results/task2_ppo/runs/standard/rollouts.jsonl")
    for r in sorted([r for r in ro if r["response_tokens"] <= 30], key=lambda r: -r["reward_raw"])[:4]:
        out.append({"kind": "train_rollout_short_high_reward", "update": r["update"], "prompt": cut(P.get(r["prompt_id"], ""), 500),
                    "reward_raw": round(r["reward_raw"], 3), "len": r["response_tokens"], "response": cut(r["response"])})
    return out


def task3():
    out, P = [], rl_prompts()
    for fork in ("fork_grpo", "fork_dr_grpo", "standard"):
        rows = load(f"results/task3_grpo/runs/{fork}/completions.jsonl")
        groups = defaultdict(list)
        for r in rows:
            groups[(r["update"], r["group"])].append(r)

        def show(kind, key, g):
            out.append({"kind": kind, "run": fork, "update": key[0], "group": key[1], "prompt": cut(P.get(g[0]["prompt_id"], ""), 400),
                        "completions": [{"reward": round(c["reward"], 3), "advantage": round(c["advantage"], 3),
                                         "len": c["response_tokens"], "loss_tokens": c.get("loss_tokens"), "truncated": c["truncated"],
                                         "seq_policy_grad_norm": None if c.get("seq_policy_grad_norm") is None else round(c["seq_policy_grad_norm"], 4),
                                         "response": cut(c["response"], 300, 100)} for c in sorted(g, key=lambda c: c["response_tokens"])]})

        def sd(g):
            x = [c["reward"] for c in g]
            m = sum(x) / len(x)
            return (sum((v - m) ** 2 for v in x) / len(x)) ** 0.5

        spread = sorted(groups.items(), key=lambda kv: -(max(c["response_tokens"] for c in kv[1]) - min(c["response_tokens"] for c in kv[1])))
        for key, g in spread[:2 if fork == "standard" else 3]:
            show("group_largest_length_spread", key, g)
        for key, g in sorted(groups.items(), key=lambda kv: sd(kv[1]))[:2]:
            show("group_smallest_reward_sd", key, g)
        for key, g in groups.items():
            if all(c["truncated"] for c in g):
                show("group_all_truncated_fully_masked", key, g)
    return out


def main():
    od = repo_path("results/qualitative")
    for name, fn in (("task1", task1), ("task2", task2), ("task3", task3)):
        rows = fn()
        write_jsonl(od / f"{name}_shortlist.jsonl", rows)
        kinds = defaultdict(int)
        for r in rows:
            kinds[r["kind"]] += 1
        print(f"{name}: {len(rows)} candidates -> {od / (name + '_shortlist.jsonl')}  {dict(kinds)}")


if __name__ == "__main__":
    main()
