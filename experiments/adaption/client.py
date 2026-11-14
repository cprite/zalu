"""Step 7: optional drop-in LLMBackend for Kirill's compile_task, backed by the Adaption-fine-tuned model served on
the GX10 (vLLM, OpenAI-compatible, reached from nikopi5 through the ssh tunnel).

    from experiments.adaption.client import adaption_backend
    spec = compile_task(prompt, llm=adaption_backend())

The fine-tuned model was trained on make_prompt(prompt) -> TaskSpecDraft JSON, so the backend ignores Kirill's long
system prompt and uses the training format; compile_task's validator and repair round still apply.
"""
import json
import os
import re
from typing import List

import requests

from experiments.adaption.common import make_prompt

URL = os.environ.get("ZALU_TASK_LLM_URL", "http://127.0.0.1:9300/v1")
MODEL = os.environ.get("ZALU_TASK_LLM_MODEL", "zalu-taskspec")


def _json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


def generate(prompt_text: str, model: str = MODEL, url: str = URL, max_tokens: int = 2048) -> str:
    r = requests.post(f"{url}/chat/completions", timeout=120, json={
        "model": model, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": prompt_text}],
        "chat_template_kwargs": {"enable_thinking": False}})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def adaption_backend(model: str = MODEL):
    from backend.task_engine.schema import TaskSpecDraft

    def call(system: str, messages: List[dict]):
        first = str(messages[0]["content"])
        m = re.search(r"<request>\s*(.*?)\s*</request>", first, flags=re.S)  # Kirill's user_message() format
        req = m.group(1) if m else first
        return TaskSpecDraft.model_validate(_json(generate(make_prompt(req), model=model)))

    call.name = f"adaption:{model}"  # type: ignore[attr-defined]
    return call
