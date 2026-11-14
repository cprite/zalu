"""Box matching for object detection (owner: Damir).

Pure Python, no ML dependencies: given ground truth and predictions for one image, decide which predictions
are true positives, which are false positives and which ground-truth boxes were missed, and say WHY.

Rules (fixed, documented so the numbers are reproducible):
- Only predictions with score >= conf_threshold count. Lower-score predictions are kept only to explain misses.
- Matching is per class and greedy by descending score (the COCO/VOC convention): each prediction takes the
  unmatched ground-truth box of the same class with the highest IoU, if that IoU >= iou_threshold.
- Each ground-truth box can be matched at most once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

# Failure reasons, written for people without ML background (the UI shows these strings' explanations).
FP_WRONG_CLASS = "wrong_class"           # box is on a real object, but of another class
FP_DUPLICATE = "duplicate"               # second box on an object already detected
FP_POOR_LOCALIZATION = "poor_localization"  # right class, box too far off the object
FP_BACKGROUND = "background"             # nothing there at all (hallucination)
FN_LOW_CONFIDENCE = "low_confidence"     # model saw it, but below the confidence threshold
FN_POOR_LOCALIZATION = "poor_localization"  # model's box overlapped, but not enough
FN_WRONG_CLASS = "wrong_class"           # model called it something else
FN_MISSED = "missed"                     # model did not react at all

LOCALIZATION_MIN_IOU = 0.1               # below this an overlap is treated as "no overlap"


@dataclass(frozen=True)
class Box:
    image: str
    cls: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: Optional[float] = None           # None for ground truth

    @staticmethod
    def of(image: str, cls: str, box: Iterable[float], score: Optional[float] = None) -> "Box":
        b = tuple(float(v) for v in box)
        if len(b) != 4:
            raise ValueError(f"box must be [x1, y1, x2, y2], got {list(b)}")
        x1, y1, x2, y2 = b
        return Box(image, cls, (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), score)


@dataclass
class Outcome:
    """Result of matching one image."""
    tp: list[tuple[Box, Box, float]] = field(default_factory=list)      # (pred, gt, iou)
    fp: list[tuple[Box, str, float]] = field(default_factory=list)      # (pred, reason, best_iou)
    fn: list[tuple[Box, str, Optional[Box]]] = field(default_factory=list)  # (gt, reason, closest pred)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def match_image(
    gts: list[Box],
    preds: list[Box],
    iou_threshold: float,
    conf_threshold: float,
) -> Outcome:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
    out = Outcome()
    kept = sorted((p for p in preds if (p.score or 0.0) >= conf_threshold), key=lambda p: -(p.score or 0.0))
    below = [p for p in preds if (p.score or 0.0) < conf_threshold]
    matched: dict[int, Box] = {}  # gt index -> pred

    for p in kept:
        best_i, best_iou = -1, 0.0
        for i, g in enumerate(gts):
            if i in matched or g.cls != p.cls:
                continue
            v = iou(p.box, g.box)
            if v > best_iou:
                best_i, best_iou = i, v
        if best_i >= 0 and best_iou >= iou_threshold:
            matched[best_i] = p
            out.tp.append((p, gts[best_i], best_iou))
            continue
        # explain the false positive
        same_matched = max((iou(p.box, gts[i].box) for i in matched if gts[i].cls == p.cls), default=0.0)
        other_cls = max((iou(p.box, g.box) for g in gts if g.cls != p.cls), default=0.0)
        if same_matched >= iou_threshold:
            out.fp.append((p, FP_DUPLICATE, same_matched))
        elif other_cls >= iou_threshold:
            out.fp.append((p, FP_WRONG_CLASS, other_cls))
        elif best_iou >= LOCALIZATION_MIN_IOU:
            out.fp.append((p, FP_POOR_LOCALIZATION, best_iou))
        else:
            out.fp.append((p, FP_BACKGROUND, max(best_iou, other_cls)))

    # explain every missed ground-truth box
    for i, g in enumerate(gts):
        if i in matched:
            continue
        low = _closest(g, [p for p in below if p.cls == g.cls])
        same = _closest(g, [p for p in kept if p.cls == g.cls])
        other = _closest(g, [p for p in kept if p.cls != g.cls])
        if low and iou(low.box, g.box) >= iou_threshold:
            out.fn.append((g, FN_LOW_CONFIDENCE, low))
        elif other and iou(other.box, g.box) >= iou_threshold:
            out.fn.append((g, FN_WRONG_CLASS, other))
        elif same and iou(same.box, g.box) >= LOCALIZATION_MIN_IOU:
            out.fn.append((g, FN_POOR_LOCALIZATION, same))
        else:
            out.fn.append((g, FN_MISSED, None))
    return out


def _closest(g: Box, cands: list[Box]) -> Optional[Box]:
    best, best_v = None, 0.0
    for p in cands:
        v = iou(p.box, g.box)
        if v > best_v:
            best, best_v = p, v
    return best
