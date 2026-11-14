"""CLI: python -m backend.task_engine "Detect workers without safety vests" [-o spec.json]"""
import argparse
import json
import sys
from pathlib import Path

from . import TaskSpec, compile_task


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.task_engine")
    ap.add_argument("prompt", nargs="?", help="Natural-language task")
    ap.add_argument("-o", "--out", type=Path, help="Write TaskSpec JSON here")
    ap.add_argument("--schema", action="store_true", help="Print the TaskSpec JSON Schema and exit")
    args = ap.parse_args()

    if args.schema:
        print(json.dumps(TaskSpec.model_json_schema(), indent=2))
        return 0
    if not args.prompt:
        ap.error("prompt is required")
    spec = compile_task(args.prompt)
    text = spec.model_dump_json(indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out} (status={spec.status})", file=sys.stderr)
    else:
        print(text)
    return 0 if spec.status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
