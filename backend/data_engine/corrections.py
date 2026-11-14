"""Apply frontend label corrections (POST /api/datasets/{id}/labels) to an AnnotationSet."""
from __future__ import annotations

from typing import Iterable, List

from .annotate import clip_box
from .schema import AnnotationSet, BoxLabel, LabelCorrection


class CorrectionError(ValueError):
    pass


def apply_corrections(aset: AnnotationSet, corrections: Iterable[LabelCorrection]) -> AnnotationSet:
    """Return a new AnnotationSet with corrections applied. Raises CorrectionError on bad input."""
    out = aset.model_copy(deep=True)
    by_id = {a.frame_id: a for a in out.annotations}
    errors: List[str] = []
    for c in corrections:
        c = LabelCorrection.model_validate(c)
        ann = by_id.get(c.frame_id)
        if ann is None:
            errors.append(f"unknown frame_id {c.frame_id}")
            continue
        if c.boxes is not None:
            new_boxes: List[BoxLabel] = []
            for i, b in enumerate(c.boxes):
                if b.class_name not in aset.class_names:
                    errors.append(f"{c.frame_id} box {i}: unknown class '{b.class_name}'")
                    continue
                if len(b.xyxy) != 4:
                    errors.append(f"{c.frame_id} box {i}: xyxy must have 4 numbers")
                    continue
                clipped = clip_box(b.xyxy, ann.width, ann.height)
                if clipped is None:
                    errors.append(f"{c.frame_id} box {i}: degenerate or outside image")
                    continue
                new_boxes.append(BoxLabel(class_name=b.class_name, xyxy=clipped, confidence=None, source="manual"))
            ann.boxes = new_boxes
        ann.status = c.status
        ann.needs_review = False
        ann.review_reasons = []
    if errors:
        raise CorrectionError("; ".join(errors))
    return out
