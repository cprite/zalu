"""Evaluation Engine (owner: Damir).

The shared API discovers `evaluate` here (see backend/main.py `_hook`). Pure logic lives in core/matching/metrics
and has no GPU dependency; engine.py wires it to the Deployment Engine and the dataset on disk.
"""
from backend.evaluation_engine.core import Criteria, compare, evaluate_model, select_threshold  # noqa: F401
from backend.evaluation_engine.engine import evaluate  # noqa: F401
