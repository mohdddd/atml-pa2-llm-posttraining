"""Build the offline page used to choose the qualitative examples (student addition; CPU, seconds).

    python -m scripts.example_picker          # -> results/qualitative/example_picker.html

One section per example the manual requires; candidates come from the shortlists (fixed rules, see
scripts/qualitative_shortlist.py and the Task 4/5 shortlists). The student ticks the examples; the picks are
recorded in results/qualitative/picks.json and expanded by `python -m scripts.selected_examples`.
"""
import collections
import html
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
OUT = R / "results/qualitative/example_picker.html"


def jl(p):
    return [json.loads(l) for l in open(R / p, encoding="utf-8") if l.strip()]


def cut(t, n=600):
    t = str(t or "")
    return t if len(t) <= n else t[:n] + " […]"


def tail(t, n=350):
    t = str(t or "")
    return t if len(t) <= n else "[…] " + t[-n:]


slots = []  # (task, slot_id, title, manual_requirement, how_to_judge, [cards])


def card(cid, facts, blocks):
    return {"id": cid, "facts": facts, "blocks": blocks}


# ---------------- Task 1 ----------------
t1 = jl("results/qualitative/task1_shortlist.jsonl")
c = []
for r in [x for x in t1 if x["kind"] == "i_heldout_confidently_against_label"][:3]:
    c.append(card(f"T1i-{r['id']}", f"Held-out pair {r['id']}: dataset labels CHOSEN as better (dataset scores chosen {r["score_chosen"]} vs rejected {r["score_rejected"]}), "
                  f"but standard DPO prefers REJECTED (margin {r['dpo_margin']}). Lengths: chosen {int(r['len_chosen'])}, rejected {int(r['len_rejected'])} tokens.",
                  [("Prompt", cut(r["prompt"], 300)), ("Chosen (label: better)", cut(r["chosen"], 450)), ("Rejected", cut(r["rejected"], 450))]))
for r in [x for x in t1 if x["kind"] == "i_generation_rm_change_vs_sft"][:3]:
    c.append(card(f"T1i-gen-{r['id']}", f"Same prompt, SFT vs standard DPO generation. RM score SFT {r['rm_sft']} → DPO {r['rm_standard']}. "
                  f"Lengths {r['len_sft']} → {r['len_standard']} tokens; truncated at 256: SFT {r['truncated_sft']}, DPO {r['truncated_standard']}.",
                  [("Prompt", cut(r["prompt"], 300)), ("SFT response", cut(r["response_sft"], 450)), ("DPO response", cut(r["response_standard"], 450))]))
slots.append(("Task 1 — DPO", "T1-i", "(i) Preference / reward vs quality disagreement",
              "An example where the preference model or reward-model score disagrees with what you judge as better quality.",
              "Read both texts. Pick ONE where you can say in a sentence why the higher-scored (or DPO-preferred) text is NOT the better one — or why the label looks wrong.", c))
c = []
for r in [x for x in t1 if x["kind"] == "ii_word_limit_violation" and x["model"] in ("standard", "length_balanced")][:4]:
    c.append(card(f"T1ii-{r['model']}-{r['id']}", f"{r['model']} model; word limit {r['word_limit']}, response has {r['words']} words → NOT compliant.",
                  [("Prompt", r["prompt"]), ("Response", cut(r["response"], 500))]))
for r in [x for x in t1 if x["kind"] == "ii_preferred_longer_standard_vs_balanced_disagree"][:2]:
    c.append(card(f"T1ii-strat-{r['id']}", f"Preferred-longer pair (chosen is {r['length_difference']} tokens longer). Standard DPO correct: {r['standard_correct']} "
                  f"(margin {r['margin_standard']}); length-balanced correct: {r['balanced_correct']} (margin {r['margin_balanced']}).",
                  [("Prompt", cut(r["prompt"], 300)), ("Chosen", cut(r["chosen"], 400)), ("Rejected", cut(r["rejected"], 400))]))
slots.append(("Task 1 — DPO", "T1-ii", "(ii) Length bias or instruction compliance",
              "An example showing behaviour related to length bias or to following an explicit instruction (e.g. the word limit).",
              "Pick ONE. A word-limit violation is the simplest (the numbers speak for themselves).", c))

# ---------------- Task 2 ----------------
t2 = jl("results/qualitative/task2_shortlist.jsonl")
c = []
for r in [x for x in t2 if x["kind"] == "heldout_largest_reward_change_standard_vs_midpoint"][:4]:
    c.append(card(f"T2-chg-{r['id'][:10]}", f"Held-out prompt: reward midpoint {r['reward_midpoint']} → standard PPO {r['reward_standard']}. "
                  f"Lengths {r['len_midpoint']} → {r['len_standard']} tokens (cap 768); truncated: midpoint {r['truncated_midpoint']}, standard {r['truncated_standard']}.",
                  [("Prompt", cut(r["prompt"], 300)), ("Midpoint response", cut(r["response_midpoint"], 450)), ("Standard PPO response", cut(r["response_standard"], 450))]))
for k, lab in (("heldout_highest_reward_truncated", "High reward although the response was CUT OFF at 768 tokens"),
               ("train_rollout_short_high_reward", "Very short training rollout with high reward"),
               ("heldout_lowest_reward_complete", "Lowest reward among complete responses")):
    for r in [x for x in t2 if x["kind"] == k][:2]:
        c.append(card(f"T2-{k[:12]}-{str(r.get('id', r.get('update')))[:10]}", f"{lab}: reward {r.get('reward', r.get('reward_raw'))}, {r['len']} tokens.",
                      [("Prompt", cut(r["prompt"], 300)), ("Response", cut(r["response"], 500))]))
slots.append(("Task 2 — PPO", "T2", "Reward and quality move together (≥1) AND disagree (≥1)",
              "At least one case where the reward-model score and the response quality agree, and at least one where they disagree.",
              "Pick TWO: one where a higher reward really is a better answer, one where it is not (e.g. a cut-off or very short answer that still scores high). Write which is which in the note box.", c))

# ---------------- Task 3 ----------------
t3 = jl("results/qualitative/task3_shortlist.jsonl")
c = []
for r in [x for x in t3 if x["run"] in ("fork_grpo", "fork_dr_grpo") and x["kind"] in ("group_largest_length_spread", "group_smallest_reward_sd", "group_all_truncated_fully_masked")][:6]:
    rows = "\n".join(f"len {x['len']:>4} | reward {x['reward']:+.2f} | advantage {x['advantage']:+.2f} | grad-norm {x['seq_policy_grad_norm']} | truncated {x['truncated']}"
                     for x in r["completions"])
    lab = {"group_largest_length_spread": "Group with the largest length spread",
           "group_smallest_reward_sd": "Group with almost identical rewards (low information)",
           "group_all_truncated_fully_masked": "Group where ALL completions were truncated → fully masked, no gradient"}[r["kind"]]
    c.append(card(f"T3-{r['run']}-u{r['update']}g{r['group']}", f"{lab}. Run: {r['run']} (update {r['update']}, group {r['group']}).",
                  [("Prompt", cut(r["prompt"], 250)), ("The K=4 completions (sorted by length)", rows),
                   ("Shortest completion", cut(r["completions"][0]["response"], 300)), ("Longest completion", cut(r["completions"][-1]["response"], 300))]))
slots.append(("Task 3 — GRPO", "T3", "Interpreting normalisation or group informativeness",
              "Brief examples that help interpret the normalisation (canonical vs Dr.GRPO) or group-informativeness results.",
              "Pick ONE or TWO. Compare grad-norm vs length inside a group: in canonical GRPO short completions get larger per-token gradients; in Dr.GRPO long ones do.", c))

# ---------------- Task 4 ----------------
sheet = {r["item_id"]: r for r in __import__("csv").DictReader(open(R / "results/task4_safety/manual_audit_sheet.csv", encoding="utf-8-sig"))}
key = {r["item_id"]: r["policies"] for r in __import__("csv").DictReader(open(R / "results/task4_safety/manual_audit_key.csv"))}
ai = {}
for p in ("sft", "dpo", "ppo", "grpo"):
    for r in jl(f"results/task4_safety/judged_{p}.jsonl"):
        ai[(p, r["xstest_id"])] = r["label"]
by_x = collections.defaultdict(list)
for iid, r in sheet.items():
    by_x[r["xstest_id"]].append(iid)


def t4card(iid):
    r = sheet[iid]
    pols = key[iid].split(";")
    a = ai[(pols[0], int(r["xstest_id"]))]
    others = [(key[j], sheet[j]["manual_label"]) for j in by_x[r["xstest_id"]] if j != iid]
    pol_diff = any(lbl != r["manual_label"] for _, lbl in others)
    judge_err = a != r["manual_label"]
    verdict = ("judge error + policy difference" if judge_err and pol_diff else "judge error" if judge_err
               else "policy difference" if pol_diff else "no disagreement (clean example)")
    other_txt = "; ".join(f"{p}: {l}" for p, l in others) or "all 4 policies gave this exact response"
    return card(f"T4-{iid}", f"XSTest {r['xstest_class']} / {r['xstest_type']}. Policies with this response: {key[iid]}. "
                f"YOUR label: {r['manual_label']}  ·  AI judge: {a}  ·  other policies on this prompt (your labels): {other_txt}.  → {verdict}",
                [("Prompt", r["prompt"]), ("Response", cut(r["response"], 500))])


for lab, title, req in (("UNSAFE_COMPLIANCE", "Harmful compliance", "an example of harmful compliance"),
                        ("JUSTIFIED_REFUSAL", "Justified refusal", "an example of a justified refusal"),
                        ("OVER_REFUSAL", "Exaggerated refusal / over-refusal", "an example of over-refusal on a safe prompt")):
    ids = [i for i, r in sheet.items() if r["manual_label"] == lab]
    ids.sort(key=lambda i: (ai[(key[i].split(";")[0], int(sheet[i]["xstest_id"]))] == sheet[i]["manual_label"], len(key[i])))
    if lab == "JUSTIFIED_REFUSAL":
        ids = [i for i in ids if sheet[i]["xstest_class"] == "UNSAFE"]
    slots.append(("Task 4 — Safety", f"T4-{lab}", title,
                  f"Brief {req}, and say whether the disagreement reflects a policy difference, a judge error, or both.",
                  "The verdict line is computed from YOUR labels: 'judge error' = AI label differs from yours; 'policy difference' = other policies "
                  "got a different label from you on the same prompt. Pick ONE and copy that verdict.", [t4card(i) for i in ids[:5]]))

# ---------------- Task 5 ----------------
diag = {(str(r["problem_id"]), r["variant_type"]): r for r in jl("data/task5_controlled_reward_diagnostics.jsonl")}
pairs = jl("results/task5_feedback/diagnostics/pairs.jsonl")
for pair, title, req in (("reasoning", "Reasoning-only corruption", "reasoning-only corruption (final answer still correct)"),
                         ("filler", "Persuasive filler", "persuasive filler"),
                         ("distractor", "Wrong final / gold distractor", "wrong-final / gold-distractor behaviour")):
    rows = [r for r in pairs if r["pair"] == pair]
    rows.sort(key=lambda r: {"wrong": 0, "better": 1, "tie": 2}[r["judge_outcome"]])
    c = []
    for r in rows[:4]:
        a, b = diag[(r["problem_id"], r["first"])], diag[(r["problem_id"], r["second"])]
        c.append(card(f"T5-{pair}-{r['problem_id']}",
                      f"Problem {r['problem_id']} (gold answer {a['gold_final']}). Exact verifier: {r['verifier_outcome'].upper()} · "
                      f"AI judge: {r['judge_outcome'].upper()}  ('better' = prefers the clean response, 'wrong' = prefers the {r['second'].replace('_', ' ')}).",
                      [("Question", cut(a["question"], 300)), (f"A = {r['first']} (end)", tail(a["response"])), (f"B = {r['second']} (end)", tail(b["response"]))]))
    slots.append(("Task 5 — RLVR vs RLAIF", f"T5-{pair}", title, f"A brief example of {req} that explains verifier–judge disagreement.",
                  "Pick ONE. Cases where the judge says WRONG or the verifier and judge differ are the most informative.", c))

# ---------------- HTML ----------------
data = json.dumps(slots, ensure_ascii=False).replace("</", "<\\/")
page = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qualitative Example Picker</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--mute:#666;--line:#ddd;--acc:#2f6f4f;--pick:#e8f3ec}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--card:#202020;--ink:#eee;--mute:#aaa;--line:#3a3a3a;--acc:#7fc79f;--pick:#1f3328}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:920px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0}h2{font-size:18px;margin:28px 0 4px;border-bottom:2px solid var(--line);padding-bottom:4px}
.req{background:var(--card);border-left:4px solid var(--acc);padding:8px 12px;margin:8px 0;font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}
.card.on{border-color:var(--acc);background:var(--pick)}
.facts{font-weight:600;font-size:14px;margin:4px 0 8px}
.lbl{font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.04em;margin-top:6px}
pre{white-space:pre-wrap;word-break:break-word;margin:2px 0;font:13px/1.45 ui-monospace,monospace;background:rgba(127,127,127,.08);padding:6px 8px;border-radius:6px}
label.pick{display:inline-flex;gap:6px;align-items:center;font-weight:600;cursor:pointer}
input[type=text]{width:100%;box-sizing:border-box;margin-top:6px;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
#bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{padding:6px 12px;border-radius:6px;border:1px solid var(--acc);background:var(--acc);color:#fff;cursor:pointer}
textarea{width:100%;box-sizing:border-box;height:220px;margin-top:8px;font:12px ui-monospace,monospace;background:var(--card);color:var(--ink)}
</style></head><body><main>
<h1>Qualitative example picker — PA2</h1>
<p>Each section is one example the manual requires. Read the candidates, tick the one(s) you want, optionally type a short note.
Then press <b>Copy my picks</b> and keep the text for writing the report.</p>
<div id="bar"><span id="count"></span><button onclick="exportPicks()">Copy my picks</button></div>
<textarea id="out" readonly placeholder="Your picks appear here after you press the button."></textarea>
<div id="root"></div></main>
<script>
const SLOTS = __DATA__;
let st = {}; try { st = JSON.parse(localStorage.getItem("pa2_picker") || "{}"); } catch (e) {}
function save(){ try { localStorage.setItem("pa2_picker", JSON.stringify(st)); } catch (e) {} count(); }
function esc(s){ const d=document.createElement("div"); d.textContent=s; return d.innerHTML; }
let h = "", lastTask = "";
SLOTS.forEach(([task, sid, title, req, how, cards]) => {
  if (task !== lastTask) { h += `<h2>${esc(task)}</h2>`; lastTask = task; }
  h += `<h3>${esc(title)}</h3><div class="req"><b>Manual asks for:</b> ${esc(req)}<br><b>How to choose:</b> ${esc(how)}</div>`;
  cards.forEach(c => {
    const on = st[c.id] && st[c.id].on;
    h += `<div class="card ${on?"on":""}" id="c_${c.id}"><label class="pick"><input type="checkbox" ${on?"checked":""}
      onchange="st['${c.id}']=Object.assign(st['${c.id}']||{}, {on:this.checked, slot:'${sid}'}); document.getElementById('c_${c.id}').classList.toggle('on', this.checked); save()"> Use this (${esc(c.id)})</label>
      <div class="facts">${esc(c.facts)}</div>
      ${c.blocks.map(([l, t]) => `<div class="lbl">${esc(l)}</div><pre>${esc(t)}</pre>`).join("")}
      <input type="text" placeholder="optional note (e.g. why you picked it)" value="${esc((st[c.id]&&st[c.id].note)||"")}"
       onchange="st['${c.id}']=Object.assign(st['${c.id}']||{}, {note:this.value, slot:'${sid}'}); save()"></div>`;
  });
});
document.getElementById("root").innerHTML = h;
function count(){ const n = Object.values(st).filter(v => v.on).length; document.getElementById("count").textContent = n + " picked (aim for about 11: T1×2, T2×2, T3×1, T4×3, T5×3)"; }
count();
function exportPicks(){
  const lines = [];
  SLOTS.forEach(([task, sid, title, req, how, cards]) => cards.forEach(c => { const s = st[c.id];
    if (s && s.on) lines.push(`[${title}] ${c.id}\\n  facts: ${c.facts}${s.note ? "\\n  note: " + s.note : ""}`); }));
  const t = lines.join("\\n\\n") || "(nothing picked yet)";
  const o = document.getElementById("out"); o.value = t;
  try { navigator.clipboard.writeText(t); } catch (e) {}
}
</script></body></html>"""
OUT.write_text(page.replace("__DATA__", data), encoding="utf-8")
print(f"{len(slots)} sections, {sum(len(s[5]) for s in slots)} candidates -> {OUT}")
