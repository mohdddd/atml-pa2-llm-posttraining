"""Record the qualitative examples chosen by the student, with full text, numbers and source files (CPU, seconds).

    python -m scripts.selected_examples

Reads the fixed list of picks in results/qualitative/picks.json (chosen by the student with the picker page
built by `python -m scripts.example_picker`) and writes, for each pick, the prompt, the response(s), the numbers
that accompany it and the result file it comes from:
    results/qualitative/selected_examples.jsonl   full texts (machine-readable)
    results/qualitative/SELECTED.md               readable copy (long responses shown as head + tail)
No selection or interpretation happens here; this only traces each chosen example back to saved results.
"""
from __future__ import annotations

import csv
import json

from common.data import read_jsonl, repo_path, write_jsonl


def short(t, head=1500, tail=400):
    t = str(t)
    return t if len(t) <= head + tail + 50 else t[:head] + "\n[... middle omitted ...]\n" + t[-tail:]


def user_text(messages):
    return next((m["content"] for m in messages if m.get("role") == "user"), "")


def find(path, pred):
    rows = [r for r in read_jsonl(path) if pred(r)]
    if len(rows) != 1:
        raise SystemExit(f"{path}: expected one match, found {len(rows)}")
    return rows[0]


def rl_prompt(pid):
    for f in ("data/rl_prompt_pool_eval.jsonl", "data/rl_prompt_pool_train.jsonl"):
        for r in read_jsonl(f):
            if r["prompt_id"] == pid or r["prompt_id"].startswith(pid):
                return user_text(r["messages"])
    return ""


def t1_gen(pick):
    i = pick["key"]
    src = {f"{r['source_split']}:{r['source_index']}": r for r in read_jsonl("data/dpo_standard_eval.jsonl")}[i]
    s = find("results/task1_dpo/eval/sft/generations.jsonl", lambda r: r["id"] == i)
    d = find("results/task1_dpo/eval/standard/generations.jsonl", lambda r: r["id"] == i)
    return {"sources": ["results/task1_dpo/eval/sft/generations.jsonl", "results/task1_dpo/eval/standard/generations.jsonl"],
            "numbers": {"rm_score_sft": round(s["reward"], 3), "rm_score_standard_dpo": round(d["reward"], 3),
                        "tokens_sft": s["response_tokens"], "tokens_dpo": d["response_tokens"],
                        "truncated_at_256_sft": s["truncated"], "truncated_at_256_dpo": d["truncated"]},
            "texts": {"prompt": src["prompt"], "SFT response": s["response"], "Standard DPO response": d["response"]}}


def t1_wl(pick):
    r = find("results/task1_dpo/eval/standard/word_limit.jsonl", lambda r: r["id"] == pick["key"])
    p = {x["prompt_id"]: user_text(x["messages"]) for x in read_jsonl("data/word_limit_prompts.jsonl")}[r["prompt_id"]]
    return {"sources": ["results/task1_dpo/eval/standard/word_limit.jsonl"],
            "numbers": {"model": "standard DPO", "word_limit": r["word_limit"], "words": r["words"], "compliant": bool(r["compliant"]),
                        "sample": r["id"]},
            "texts": {"prompt": p, "response": r["response"]}}


def t2(pick):
    r = find("results/task2_ppo/eval/standard/generations.jsonl", lambda r: r["id"].startswith(pick["key"]))
    return {"sources": ["results/task2_ppo/eval/standard/generations.jsonl"],
            "numbers": {"model": "standard PPO (20 updates)", "reward_model_score": round(r["reward"], 3),
                        "tokens": r["response_tokens"], "truncated_at_768": r["truncated"], "prompt_id": r["id"]},
            "texts": {"prompt": rl_prompt(r["id"]), "response": r["response"]}}


def t3(pick):
    run, upd, grp = pick["key"]
    rows = [r for r in read_jsonl(f"results/task3_grpo/runs/{run}/completions.jsonl") if r["update"] == upd and r["group"] == grp]
    rows.sort(key=lambda r: r["response_tokens"])
    table = [{"tokens": r["response_tokens"], "loss_tokens": r.get("loss_tokens"), "reward": round(r["reward"], 3),
              "advantage": round(r["advantage"], 3), "per_completion_grad_norm": r.get("seq_policy_grad_norm"),
              "truncated": r["truncated"]} for r in rows]
    texts = {"prompt": rl_prompt(rows[0]["prompt_id"])}
    for k, r in enumerate(rows):
        texts[f"completion {k + 1} ({r['response_tokens']} tokens)"] = r["response"]
    return {"sources": [f"results/task3_grpo/runs/{run}/completions.jsonl"],
            "numbers": {"run": run, "update": upd, "group": grp, "completions_sorted_by_length": table}, "texts": texts}


def t4(pick):
    iid = pick["key"]
    sheet = {r["item_id"]: r for r in csv.DictReader(open(repo_path("results/task4_safety/manual_audit_sheet.csv"), encoding="utf-8-sig"))}
    key = {r["item_id"]: r["policies"] for r in csv.DictReader(open(repo_path("results/task4_safety/manual_audit_key.csv")))}
    r = sheet[iid]
    xid = int(r["xstest_id"])
    pols = key[iid].split(";")
    j = find(f"results/task4_safety/judged_{pols[0]}.jsonl", lambda x: x["xstest_id"] == xid)
    others = {key[k]: sheet[k]["manual_label"] for k in sheet if sheet[k]["xstest_id"] == r["xstest_id"] and k != iid}
    return {"sources": ["results/task4_safety/manual_audit_sheet.csv", "results/task4_safety/manual_audit_key.csv",
                        f"results/task4_safety/judged_{pols[0]}.jsonl"],
            "numbers": {"xstest_id": xid, "xstest_class": r["xstest_class"], "xstest_type": r["xstest_type"],
                        "policies_with_this_response": pols, "manual_label": r["manual_label"], "ai_label": j["label"],
                        "ai_confidence": j["confidence"], "ai_rationale_tag": j["rationale_tag"],
                        "manual_labels_of_other_policies_on_this_prompt": others},
            "texts": {"prompt": r["prompt"], "response": r["response"]}}


def t5(pick):
    pair, pid = pick["key"]
    pr = find("results/task5_feedback/diagnostics/pairs.jsonl", lambda r: r["pair"] == pair and str(r["problem_id"]) == pid)
    oc = find("results/task5_feedback/diagnostics/order_check.jsonl", lambda r: r["pair"] == pair and str(r["problem_id"]) == pid)
    vs = {r["variant"]: r for r in read_jsonl("results/task5_feedback/diagnostics/variant_scores.jsonl") if str(r["problem_id"]) == pid}
    d = {r["variant_type"]: r for r in read_jsonl("data/task5_controlled_reward_diagnostics.jsonl") if str(r["problem_id"]) == pid}
    a, b = pr["first"], pr["second"]
    return {"sources": ["data/task5_controlled_reward_diagnostics.jsonl", "results/task5_feedback/diagnostics/pairs.jsonl",
                        "results/task5_feedback/diagnostics/order_check.jsonl", "results/task5_feedback/diagnostics/variant_scores.jsonl"],
            "numbers": {"problem_id": pid, "gold_final": d[a]["gold_final"], "pair": f"{a} (A) vs {b} (B)",
                        "verifier_reward_A": vs[a]["rlvr_reward"], "verifier_reward_B": vs[b]["rlvr_reward"],
                        "verifier_preference": pr["verifier_pref"], "judge_preference": pr["judge_pref"],
                        "judge_outcome": pr["judge_outcome"], "judge_A_shown_first": oc["first_shown_first"],
                        "judge_B_shown_first (as A/B)": {"A": "B", "B": "A", "TIE": "TIE"}[oc["second_shown_first"]],
                        "judge_order_consistent": oc["consistent"],
                        "rlaif_group_reward_A": round(vs[a]["rlaif_group_reward"], 3), "rlaif_group_reward_B": round(vs[b]["rlaif_group_reward"], 3)},
            "texts": {"question": d[a]["question"], f"A = {a}": d[a]["response"], f"B = {b}": d[b]["response"]}}


HANDLERS = {"t1_gen": t1_gen, "t1_wl": t1_wl, "t2": t2, "t3": t3, "t4": t4, "t5": t5}


def main():
    picks = json.loads(repo_path("results/qualitative/picks.json").read_text())
    out, md = [], ["# Selected qualitative examples\n",
                   "Chosen by the student with the example picker (`python -m scripts.example_picker`); full texts, numbers and",
                   "source files written by `python -m scripts.selected_examples`. No interpretation here.\n"]
    for p in picks:
        rec = {"slot": p["slot"], "pick_id": p["pick_id"], **HANDLERS[p["handler"]](p)}
        out.append(rec)
        md.append(f"## {p['slot']} — `{p['pick_id']}`\n")
        md.append("Sources: " + ", ".join(f"`{s}`" for s in rec["sources"]) + "\n")
        md.append("```json\n" + json.dumps(rec["numbers"], indent=1, ensure_ascii=False) + "\n```\n")
        for k, v in rec["texts"].items():
            md.append(f"**{k}**\n\n```text\n{short(v)}\n```\n")
    write_jsonl("results/qualitative/selected_examples.jsonl", out)
    repo_path("results/qualitative/SELECTED.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(out)} examples -> results/qualitative/selected_examples.jsonl, results/qualitative/SELECTED.md")


if __name__ == "__main__":
    main()
