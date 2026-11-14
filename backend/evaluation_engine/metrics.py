"""Metrics, confidence intervals and sample-size warnings (owner: Damir).

Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = harmonic mean. All counts are box-level
(micro-averaged over images), plus an image-level view ("did the model flag this frame at all?").

Confidence intervals are 95% Wilson score intervals. Boxes in the same video are correlated, so the real
uncertainty is larger than the interval says; the report states this.
"""
from __future__ import annotations

import math
from typing import Optional

Z95 = 1.959963984540054

# Below these counts we warn that the numbers are too noisy to trust (the UI shows the warnings verbatim).
MIN_GT_PER_CLASS = 30
MIN_IMAGES = 20
MIN_SESSIONS = 2
MAX_CI_WIDTH = 0.20


def safe_div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


def f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None:
        return None
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def wilson(successes: int, n: int, z: float = Z95) -> Optional[tuple[float, float]]:
    """95% Wilson score interval for a proportion; None when n == 0."""
    if n == 0:
        return None
    ph = successes / n
    denom = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / denom
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def prf(tp: int, fp: int, fn: int) -> dict:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _r(p),
        "recall": _r(r),
        "f1": _r(f1(p, r)),
        "precision_ci95": _rci(wilson(tp, tp + fp)),
        "recall_ci95": _rci(wilson(tp, tp + fn)),
    }


def sample_warnings(
    n_images: int, gt_per_class: dict[str, int], n_sessions: Optional[int], m: dict
) -> list[str]:
    w = []
    if n_images < MIN_IMAGES:
        w.append(f"Only {n_images} test images (recommended >= {MIN_IMAGES}). Treat the metrics as indicative.")
    for c, n in sorted(gt_per_class.items()):
        if n == 0:
            w.append(f"No ground-truth '{c}' objects in the test set: recall for '{c}' cannot be measured.")
        elif n < MIN_GT_PER_CLASS:
            w.append(f"Only {n} '{c}' objects in the test set (recommended >= {MIN_GT_PER_CLASS}).")
    if n_sessions is not None and n_sessions < MIN_SESSIONS:
        w.append(
            f"Test data comes from {n_sessions} recording session(s); frames from one session are highly "
            f"correlated, so the true uncertainty is larger than the intervals shown."
        )
    for name in ("precision", "recall"):
        ci = m.get(f"{name}_ci95")
        if ci and ci[1] - ci[0] > MAX_CI_WIDTH:
            w.append(f"{name.capitalize()} 95% interval is wide ({ci[0]:.2f}-{ci[1]:.2f}): more test data needed.")
    return w


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 4)


def _rci(ci: Optional[tuple[float, float]]) -> Optional[list[float]]:
    return None if ci is None else [round(ci[0], 4), round(ci[1], 4)]
