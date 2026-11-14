"""Data-leakage guards for the final test set (owner: Damir). Uses Kirill's DatasetManifest (per-frame session_id
and synthetic flags).

Rules:
1. Final test data comes from recording sessions never used for training or validation.
2. Synthetic samples never appear in the final test set.
3. Validation picks models/thresholds; the test set only reports. (Enforced by process, recorded in notes.)

Errors make the report invalid (verdict "invalid_test_set"); warnings are shown but do not block.
"""
from __future__ import annotations

from backend.evaluation_engine.dataset_io import TestFrame, train_sessions

SYNTHETIC_MARKERS = ("synth", "fal_", "generated")


def check_test_set(test, train_sets: list, frames: list[TestFrame], split: str) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if test.variant == "real_plus_synthetic":
        errors.append(f"Test dataset {test.dataset_id} is 'real_plus_synthetic': the final test set must be real footage only.")
    elif test.variant != "test":
        warnings.append(f"Dataset {test.dataset_id} was not created as an independent test set (variant '{test.variant}'). "
                        "Upload test footage with kind=test.")
    if split != "test":
        warnings.append(f"Dataset has no 'test' split; using '{split}'. Make sure it was not used to tune the model.")

    syn = [f.image for f in frames if f.synthetic or any(k in f.image.lower() for k in SYNTHETIC_MARKERS)]
    if syn:
        errors.append(f"{len(syn)} synthetic images found in the test set (e.g. {syn[0]}).")

    sessions = {f.session_id for f in frames if f.session_id}
    if not sessions:
        warnings.append("Test frames have no recording-session ids, so independence from training data cannot be verified.")
    for tr in train_sets:
        if tr is None:
            continue
        if tr.dataset_id == test.dataset_id or tr.root == test.root:
            errors.append(f"Test dataset is the same as training dataset {tr.dataset_id}.")
            continue
        shared = sessions & train_sessions(tr)
        if shared:
            errors.append(f"Recording session(s) {sorted(shared)} appear in both training dataset {tr.dataset_id} and the test set.")
    return errors, warnings
