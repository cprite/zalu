"""Render annotated preview images for quick visual QA."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from .schema import AnnotationSet

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72)]


def render_previews(aset: AnnotationSet, out_dir, limit: Optional[int] = None) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for a in aset.annotations[:limit]:
        img = cv2.imread(a.image_path)
        if img is None:
            continue
        for b in a.boxes:
            color = PALETTE[aset.class_names.index(b.class_name) % len(PALETTE)]
            x1, y1, x2, y2 = map(int, b.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            tag = b.class_name + (f" {b.confidence:.2f}" if b.confidence is not None else " (manual)")
            cv2.putText(img, tag, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if a.needs_review:
            cv2.putText(img, "REVIEW: " + "; ".join(a.review_reasons)[:90], (8, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(str(out / f"{a.frame_id}.jpg"), img)
        n += 1
    return n
