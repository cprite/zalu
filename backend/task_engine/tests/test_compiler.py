import json
import os
from pathlib import Path

import pytest

from backend.task_engine import TaskCompilationError, TaskSpec, TaskSpecDraft, compile_task

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _draft(**overrides) -> TaskSpecDraft:
    base = json.loads((EXAMPLES / "vest.json").read_text())
    for k in ("spec_version", "task_id", "prompt", "compiler"):
        base.pop(k)
    base.update(overrides)
    return TaskSpecDraft.model_validate(base)


class FakeLLM:
    def __init__(self, *drafts):
        self.drafts = list(drafts)
        self.calls = []
        self.name = "fake"

    def __call__(self, system, messages):
        self.calls.append(messages)
        return self.drafts.pop(0)


def test_valid_spec_passes_through():
    llm = FakeLLM(_draft())
    spec = compile_task("Detect workers without safety vests.", llm=llm)
    assert spec.status == "ok"
    assert spec.class_names == ["person_with_vest", "person_without_vest"]
    assert spec.class_id("person_without_vest") == 1
    assert spec.compiler == "fake" and spec.task_id.startswith("task_")
    assert len(llm.calls) == 1


def test_inconsistent_spec_is_repaired_once():
    bad = _draft(target_event="no_such_class")
    llm = FakeLLM(bad, _draft())
    spec = compile_task("vests", llm=llm)
    assert spec.status == "ok"
    assert "failed validation" in llm.calls[1][-1]["content"]


def test_inconsistent_spec_twice_raises():
    bad = _draft(target_event="no_such_class")
    with pytest.raises(TaskCompilationError):
        compile_task("vests", llm=FakeLLM(bad, bad))


def test_negated_detector_prompt_rejected():
    d = _draft()
    d.labeling.detector_prompts[0].text = "person without vest"
    with pytest.raises(TaskCompilationError, match="negation"):
        compile_task("vests", llm=FakeLLM(d, d))


def test_rule_producing_undeclared_class_rejected():
    d = _draft()
    d.labeling.rules[0].class_if_absent = "naked_person"
    with pytest.raises(TaskCompilationError, match="undeclared"):
        compile_task("vests", llm=FakeLLM(d, d))


def test_empty_prompt_needs_clarification_without_llm_call():
    llm = FakeLLM()
    spec = compile_task("   ", llm=llm)
    assert spec.status == "needs_clarification" and not llm.calls


def test_unsupported_spec_may_be_partial():
    llm = FakeLLM(_draft(status="unsupported", classes=[], target_event=None))
    spec = compile_task("recognise faces", llm=llm)
    assert spec.status == "unsupported"


@pytest.mark.parametrize("name", ["vest", "tables", "vague", "face"])
def test_example_files_validate(name):
    spec = TaskSpec.model_validate_json((EXAMPLES / f"{name}.json").read_text())
    assert spec.spec_version == "1.0"


@pytest.mark.skipif(not os.environ.get("VF_LIVE_TESTS"), reason="set VF_LIVE_TESTS=1 to call Claude")
def test_live_claude_vest():
    spec = compile_task("Detect workers without safety vests.")
    assert spec.status == "ok"
    assert spec.target_event in spec.class_names
    assert any(r.kind == "attribute" for r in spec.labeling.rules)
