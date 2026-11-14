"""Step 6: base vs Adaption-fine-tuned on held-out TEST prompts (domains never used in training).

Variants (all temperature 0, same vLLM server on the GX10):
  base-short  : base model, the training prompt format (what AutoScientist compares against)
  base-full   : base model with Kirill's full system prompt + request (the fairest strong baseline)
  finetuned   : base + Adaption LoRA, training prompt format
Metrics per variant: JSON parse rate, TaskSpecDraft schema-valid rate, Kirill-validator-clean rate,
status agreement with gold, exact class-set agreement (status=ok), target_event agreement, median latency.
Usage: python3 evaluate.py [--limit N] [--variants base-short,base-full,finetuned]
"""
import argparse, json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from experiments.adaption.client import _json, generate          # noqa: E402
from experiments.adaption.common import BASE_MODEL, D, make_prompt  # noqa: E402
from backend.task_engine.prompts import SYSTEM_PROMPT, user_message  # noqa: E402
from backend.task_engine.schema import TaskSpecDraft, consistency_errors  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--variants", default="base-short,base-full,finetuned"); a = ap.parse_args()
test = [json.loads(l) for l in (D / "test.jsonl").read_text().splitlines() if l.strip()]
if a.limit: test = test[: a.limit]


def run(variant, prompt):
    if variant == "finetuned":
        return generate(make_prompt(prompt), model="zalu-taskspec")
    if variant == "base-short":
        return generate(make_prompt(prompt), model=BASE_MODEL)
    return generate(SYSTEM_PROMPT + "\n\n" + user_message(prompt) + "\n\nReply with the TaskSpec JSON only.", model=BASE_MODEL)


results = {}
for v in a.variants.split(","):
    rows = []
    for t in test:
        t0 = time.time(); r = {"prompt": t["prompt"]}
        try:
            out = run(v, t["prompt"]); r["latency"] = time.time() - t0
            obj = _json(out); r["parsed"] = True
            d = TaskSpecDraft.model_validate(obj); r["schema"] = True
            r["clean"] = not consistency_errors(d)
            g = t["gold"]
            r["status_ok"] = d.status == g["status"]
            if g["status"] == "ok":
                r["classes_ok"] = sorted(c.name for c in d.classes) == sorted(c["name"] for c in g["classes"])
                r["event_ok"] = d.target_event == g["target_event"]
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {str(e)[:120]}"; r.setdefault("latency", time.time() - t0)
        rows.append(r)
    n = len(rows); okg = [r for r, t in zip(rows, test) if t["gold"]["status"] == "ok"]
    rate = lambda k, rs=rows: round(sum(bool(r.get(k)) for r in rs) / max(len(rs), 1), 3)
    results[v] = {"n": n, "parse": rate("parsed"), "schema_valid": rate("schema"), "validator_clean": rate("clean"),
                  "status_agree": rate("status_ok"), "class_set_agree(ok)": rate("classes_ok", okg),
                  "target_event_agree(ok)": rate("event_ok", okg),
                  "median_latency_s": round(statistics.median(r["latency"] for r in rows), 2)}
    (D / f"eval_{v}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(v, results[v], flush=True)
(D / "eval_results.json").write_text(json.dumps(results, indent=1))
