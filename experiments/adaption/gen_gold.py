"""Step 2: gold TaskSpecs via Kirill's compile_task on the Claude subscription; keep only validator-clean specs.
Resumable: skips prompts already in gold.jsonl."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from backend.infrastructure.claude_cli import claude_cli_backend          # noqa: E402
from backend.task_engine.compiler import compile_task, TaskCompilationError  # noqa: E402
from backend.task_engine.prompts import SYSTEM_PROMPT, user_message        # noqa: E402
from backend.task_engine.schema import consistency_errors, TaskSpecDraft   # noqa: E402

D = Path(__file__).parent / "data"
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
llm = claude_cli_backend(model="sonnet")


def one(row):
    t0 = time.time()
    try:
        spec = compile_task(row["prompt"], llm=llm)
    except TaskCompilationError as e:
        return {**row, "ok": False, "error": str(e)[:300]}
    except Exception as e:
        return {**row, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    draft = TaskSpecDraft(**{k: v for k, v in spec.model_dump().items() if k in TaskSpecDraft.model_fields})
    errs = consistency_errors(draft)
    return {**row, "ok": not errs, "errors": errs, "seconds": round(time.time() - t0, 1),
            "system": SYSTEM_PROMPT, "user": user_message(row["prompt"]), "completion": draft.model_dump_json()}


if __name__ == "__main__":
    rows = [json.loads(l) for l in (D / "prompts.jsonl").read_text().splitlines() if l.strip()]
    gold_path = D / "gold.jsonl"
    done = {json.loads(l)["prompt"] for l in gold_path.read_text().splitlines()} if gold_path.exists() else set()
    todo = [r for r in rows if r["prompt"] not in done]
    print(f"{len(done)} done, {len(todo)} to go, {WORKERS} workers", flush=True)
    n = 0
    with ThreadPoolExecutor(WORKERS) as ex, gold_path.open("a") as fh:
        for f in as_completed([ex.submit(one, r) for r in todo]):
            fh.write(json.dumps(f.result()) + "\n"); fh.flush(); n += 1
            if n % 25 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    print("done")
