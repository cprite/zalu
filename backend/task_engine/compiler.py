"""compile_task(prompt) -> TaskSpec, using Claude structured outputs."""
from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional

from .prompts import SYSTEM_PROMPT, user_message
from .schema import LabelingPlan, TaskSpec, TaskSpecDraft, consistency_errors

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("VF_TASK_MODEL", "claude-opus-5-5")
MAX_PROMPT_CHARS = 2000

# An LLM backend takes (system, messages) and returns a TaskSpecDraft.
LLMBackend = Callable[[str, List[dict]], TaskSpecDraft]


class TaskCompilationError(RuntimeError):
    """Claude did not return a usable spec (refusal, truncation, repeated invalid output)."""


def claude_backend(model: str = DEFAULT_MODEL, client=None) -> LLMBackend:
    import anthropic

    client = client or anthropic.Anthropic()

    def call(system: str, messages: List[dict]) -> TaskSpecDraft:
        # Server-side refusal fallback is enabled so a classifier false positive
        # on e.g. a security-camera prompt is retried on a fallback model.
        resp = client.beta.messages.parse(
            model=model,
            max_tokens=16000,
            system=system,
            messages=messages,
            output_format=TaskSpecDraft,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            raise TaskCompilationError(f"Claude declined the request: {resp.stop_details}")
        if resp.stop_reason == "max_tokens":
            raise TaskCompilationError("Claude output was truncated (max_tokens)")
        if resp.parsed_output is None:
            raise TaskCompilationError("Claude returned no parsable TaskSpec")
        return resp.parsed_output

    call.name = f"claude:{model}"  # type: ignore[attr-defined]
    return call


def _placeholder(prompt: str, status: str, issues: List[str], questions: List[str], compiler: str) -> TaskSpec:
    return TaskSpec(
        prompt=prompt,
        compiler=compiler,
        status=status,  # type: ignore[arg-type]
        issues=issues,
        clarification_questions=questions,
        task_type="object_detection",
        summary="",
        classes=[],
        target_event=None,
        event_rules=[],
        visual_conditions=[],
        requires_tracking=False,
        primary_metric="map50",
        secondary_metrics=[],
        test_scenarios=[],
        synthetic_targets=[],
        labeling=LabelingPlan(detector_prompts=[], rules=[]),
    )


def compile_task(
    prompt: str,
    llm: Optional[LLMBackend] = None,
    max_repairs: int = 1,
) -> TaskSpec:
    """Convert a natural-language request into a validated TaskSpec.

    Returns a spec with status "ok", "needs_clarification" or "unsupported".
    Raises TaskCompilationError if the LLM keeps returning inconsistent output.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return _placeholder("", "needs_clarification", ["empty prompt"],
                            ["What should the camera detect?"], "rules")
    if len(prompt) > MAX_PROMPT_CHARS:
        return _placeholder(prompt[:MAX_PROMPT_CHARS], "needs_clarification",
                            [f"prompt longer than {MAX_PROMPT_CHARS} characters"],
                            ["Please describe the task in one or two sentences."], "rules")

    llm = llm or claude_backend()
    compiler_name = getattr(llm, "name", "custom")
    messages: List[dict] = [{"role": "user", "content": user_message(prompt)}]

    for attempt in range(max_repairs + 1):
        draft = llm(SYSTEM_PROMPT, messages)
        errors = consistency_errors(draft)
        if not errors:
            return TaskSpec(prompt=prompt, compiler=compiler_name, **draft.model_dump())
        log.warning("TaskSpec attempt %d inconsistent: %s", attempt + 1, errors)
        messages = messages + [
            {"role": "assistant", "content": draft.model_dump_json()},
            {"role": "user", "content": "That TaskSpec failed validation:\n- " + "\n- ".join(errors)
             + "\nReturn a corrected TaskSpec."},
        ]
    raise TaskCompilationError(f"TaskSpec still invalid after {max_repairs + 1} attempts: {errors}")


def create_task_spec(prompt: str) -> TaskSpec:
    """Backend hook (POST /api/tasks). Never raises on bad LLM output: if Claude refuses
    or keeps returning an inconsistent spec, the user gets a needs_clarification spec
    explaining why. Network/auth errors from the Anthropic SDK still raise."""
    try:
        return compile_task(prompt)
    except TaskCompilationError as e:
        return _placeholder(prompt.strip()[:MAX_PROMPT_CHARS], "needs_clarification",
                            [f"Task Engine could not build a valid spec: {e}"],
                            ["Please rephrase the task, naming the objects to detect."], "claude:error")
