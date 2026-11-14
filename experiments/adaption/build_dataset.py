"""Step 2b: gold.jsonl -> train.jsonl (prompt/completion) + test.jsonl. Test = held-out DOMAINS only."""
import json
from common import D, make_prompt

rows = [json.loads(l) for l in (D / "gold.jsonl").read_text().splitlines() if l.strip()]
ok = [r for r in rows if r.get("ok")]
train = [{"prompt": make_prompt(r["prompt"]), "completion": r["completion"]} for r in ok if r["split"] == "train"]
test = [{"prompt": r["prompt"], "domain": r["domain"], "gold": json.loads(r["completion"])} for r in ok if r["split"] == "test"]
(D / "train.jsonl").write_text("\n".join(json.dumps(x) for x in train) + "\n")
(D / "test.jsonl").write_text("\n".join(json.dumps(x) for x in test) + "\n")
print({"gold_rows": len(rows), "validator_ok": len(ok), "train": len(train), "test": len(test)})
