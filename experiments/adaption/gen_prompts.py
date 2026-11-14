"""Step 1: generate diverse camera-task prompts with the Claude CLI (owner's subscription), split by domain."""
import json, random, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).parent / "data"; OUT.mkdir(exist_ok=True)
CLAUDE = "/home/pi/.local/bin/claude"
DOMAINS = ["construction site safety", "warehouse and logistics", "retail store", "restaurant and cafe",
           "university campus and lecture halls", "hospital and care home", "factory quality control",
           "farm and livestock", "parking and traffic", "home and pets", "sports and gyms", "public transport stations",
           "offices and coworking", "wildlife and nature cameras", "airports", "kitchens and food hygiene"]
TEST_DOMAINS = {"airports", "kitchens and food hygiene"}          # never used for training
SCHEMA = {"type": "object", "properties": {"prompts": {"type": "array", "items": {"type": "string"}}}, "required": ["prompts"]}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 90


def batch(domain: str) -> list[str]:
    ask = (f"Write {N} different requests that a non-technical user might type into a no-code tool that trains a camera "
           f"object detector, for the domain: {domain}. Vary length (3 to 40 words), tone, wording and spelling mistakes. "
           f"About 80% clear detection tasks (often 'X without Y' or 'X in zone'), 10% vague requests, 10% tasks a detector "
           f"cannot do (e.g. reading minds, identifying a named person, predicting the future). One request per item.")
    p = subprocess.run([CLAUDE, "-p", "--model", "sonnet", "--output-format", "json", "--tools", "",
                        "--json-schema", json.dumps(SCHEMA), ask], capture_output=True, text=True, timeout=300)
    out = json.loads(p.stdout)
    return [s.strip() for s in out["structured_output"]["prompts"] if s.strip()]


if __name__ == "__main__":
    with ThreadPoolExecutor(4) as ex:
        res = dict(zip(DOMAINS, ex.map(batch, DOMAINS)))
    seen, rows = set(), []
    for d, ps in res.items():
        for p in ps:
            k = p.lower()
            if k not in seen:
                seen.add(k); rows.append({"domain": d, "prompt": p, "split": "test" if d in TEST_DOMAINS else "train"})
    random.Random(0).shuffle(rows)
    (OUT / "prompts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print({s: sum(r["split"] == s for r in rows) for s in ("train", "test")})
