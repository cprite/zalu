"""Shared prompt format for training, evaluation and inference (must be identical everywhere)."""
import json
import os
from pathlib import Path

D = Path(__file__).parent / "data"
D.mkdir(exist_ok=True)
STATE = D / "state.json"
BASE_MODEL = os.environ.get("ZALU_ADAPTION_BASE", "Qwen/Qwen3.5-4B")
INSTRUCTION = ("You are ZALU's task compiler. Convert the camera task request below into a ZALU TaskSpec as one JSON "
               "object (fields: status, issues, clarification_questions, task_type, summary, classes, target_event, "
               "event_rules, visual_conditions, requires_tracking, primary_metric, secondary_metrics, test_scenarios, "
               "synthetic_targets, labeling). Reply with the JSON only.\n\nRequest: ")


def make_prompt(user_prompt: str) -> str:
    return INSTRUCTION + user_prompt.strip()


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kw) -> dict:
    s = load_state(); s.update(kw); STATE.write_text(json.dumps(s, indent=1)); return s
