"""VisionForge Task Engine: natural-language prompt -> validated TaskSpec."""
from .compiler import DEFAULT_MODEL, create_task_spec, TaskCompilationError, claude_backend, compile_task
from .schema import SPEC_VERSION, ClassDef, DerivationRule, DetectorPrompt, LabelingPlan, TaskSpec, TaskSpecDraft

__all__ = [
    "compile_task", "create_task_spec", "claude_backend", "TaskCompilationError", "DEFAULT_MODEL",
    "TaskSpec", "TaskSpecDraft", "ClassDef", "DetectorPrompt", "DerivationRule", "LabelingPlan", "SPEC_VERSION",
]
