import pytest

from backend.task_engine import TaskSpec

from backend.data_engine.tests.helpers import VEST_SPEC


@pytest.fixture
def vest_task() -> TaskSpec:
    return TaskSpec.model_validate_json(VEST_SPEC.read_text())
