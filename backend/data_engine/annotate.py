"""auto_annotate(frames, task) -> AnnotationSet.

1. An open-vocabulary detector (YOLO-World by default) finds every
   `task.labeling.detector_prompts` item in each frame.
2. `task.labeling.rules` turn those raw detections into training classes
   (e.g. person + safety vest -> person_with_vest / person_without_vest).
3. Boxes are validated and uncertain frames are flagged for human review.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from .schema import Annotation, AnnotationSet, BoxLabel, Frame, FrameCollection

log = logging.getLogger(__name__)

Det = Tuple[List[float], float]  # (xyxy, confidence)
RawDetections = Dict[str, List[Det]]  # detector prompt key -> detections

WEIGHTS_DIR = Path(os.environ.get("VF_WEIGHTS_DIR", Path.home() / ".cache" / "visionforge" / "weights"))
MIN_BOX_PX = 4.0
LOW_CONF_MARGIN = 0.15
NEAR_MISS_FACTOR = 0.5  # detectors return down to this fraction of min_confidence; used only for review flags
EDGE_PX = 3.0


class Detector(Protocol):
    name: str

    def detect(self, image_paths: Sequence[str], prompts) -> List[RawDetections]: ...


class YoloWorldDetector:
    """Ultralytics YOLO-World (or YOLOE) open-vocabulary detector."""

    def __init__(self, weights: str = "yolov8s-worldv2.pt", device: Optional[str] = None,
                 imgsz: int = 640, batch: int = 8):
        from ultralytics import YOLO

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        path = Path(weights)
        if not path.is_absolute() and not path.exists():
            path = WEIGHTS_DIR / weights  # ultralytics downloads release assets to this path
        self.model = YOLO(str(path))
        self.device, self.imgsz, self.batch = device, imgsz, batch
        self.name = f"yolo-world:{Path(weights).name}"
        self._classes: Optional[List[str]] = None

    def detect(self, image_paths, prompts) -> List[RawDetections]:
        texts = [p.text for p in prompts]
        if texts != self._classes:
            self.model.set_classes(texts)
            self._classes = texts
        floor = NEAR_MISS_FACTOR * min(p.min_confidence for p in prompts)
        out: List[RawDetections] = []
        for i in range(0, len(image_paths), self.batch):
            chunk = list(image_paths[i:i + self.batch])
            results = self.model.predict(chunk, conf=floor, imgsz=self.imgsz, device=self.device, verbose=False)
            for r in results:
                dets: RawDetections = {p.key: [] for p in prompts}
                for xyxy, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                    p = prompts[int(cls)]
                    if conf >= NEAR_MISS_FACTOR * p.min_confidence:
                        dets[p.key].append(([round(v, 1) for v in xyxy], round(float(conf), 4)))
                out.append(dets)
        return out


# ---------------------------------------------------------------- geometry
def _pad(b: List[float], f: float) -> List[float]:
    w, h = b[2] - b[0], b[3] - b[1]
    return [b[0] - w * f, b[1] - h * f, b[2] + w * f, b[3] + h * f]


def ioa(subject: List[float], attr: List[float]) -> float:
    """Fraction of `attr` area that lies inside `subject`."""
    ix = max(0.0, min(subject[2], attr[2]) - max(subject[0], attr[0]))
    iy = max(0.0, min(subject[3], attr[3]) - max(subject[1], attr[1]))
    area = max(1e-9, (attr[2] - attr[0]) * (attr[3] - attr[1]))
    return ix * iy / area


def clip_box(b: List[float], w: int, h: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = (max(0.0, min(float(b[0]), w)), max(0.0, min(float(b[1]), h)),
                      max(0.0, min(float(b[2]), w)), max(0.0, min(float(b[3]), h)))
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        return None
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


# ---------------------------------------------------------------- rules
def _touches_edge(b: List[float], w: int, h: int) -> bool:
    return b[0] <= EDGE_PX or b[1] <= EDGE_PX or b[2] >= w - EDGE_PX or b[3] >= h - EDGE_PX


def apply_rules(raw: RawDetections, task, width: int, height: int) -> Tuple[List[BoxLabel], List[str]]:
    """Derive training-class boxes from raw detections. Returns (boxes, review_reasons).

    Detections below a prompt's min_confidence never become labels; detections
    between NEAR_MISS_FACTOR*min_confidence and min_confidence only raise review
    flags (likely missed objects / missed attributes).
    """
    prompts = {p.key: p for p in task.labeling.detector_prompts}
    accepted = {k: [(b, c) for b, c in v if c >= prompts[k].min_confidence] for k, v in raw.items() if k in prompts}
    near = {k: [(b, c) for b, c in v if c < prompts[k].min_confidence] for k, v in raw.items() if k in prompts}
    boxes: List[BoxLabel] = []
    reasons: List[str] = []

    for rule in task.labeling.rules:
        subj_p = prompts[rule.subject]
        for sbox, sconf in accepted.get(rule.subject, []):
            clipped = clip_box(sbox, width, height)
            if clipped is None:
                continue
            ev = {"subject": rule.subject, "subject_conf": sconf}
            if sconf < subj_p.min_confidence + LOW_CONF_MARGIN:
                reasons.append(f"low-confidence {rule.subject} ({sconf:.2f})")
            if rule.kind == "direct":
                cls = rule.class_if_present
            else:
                padded = _pad(sbox, rule.subject_padding)
                best, best_conf = 0.0, None
                for abox, aconf in accepted.get(rule.attribute, []):
                    o = ioa(padded, abox)
                    if o > best:
                        best, best_conf = o, aconf
                present = best >= rule.min_overlap
                cls = rule.class_if_present if present else rule.class_if_absent
                ev.update(attribute=rule.attribute, attribute_overlap=round(best, 3), attribute_conf=best_conf)
                if present:
                    attr_p = prompts[rule.attribute]
                    if best_conf is not None and best_conf < attr_p.min_confidence + LOW_CONF_MARGIN:
                        reasons.append(f"low-confidence {rule.attribute} ({best_conf:.2f}) decided {cls}")
                else:
                    if best >= 0.5 * rule.min_overlap:
                        reasons.append(f"borderline {rule.attribute} overlap {best:.2f} -> {cls}")
                    weak = [c for b, c in near.get(rule.attribute, []) if ioa(padded, b) >= rule.min_overlap]
                    if weak:
                        reasons.append(f"possible {rule.attribute} below threshold ({max(weak):.2f}) -> {cls}")
                    if _touches_edge(sbox, width, height):
                        reasons.append(f"{rule.subject} cut off at image edge; {rule.attribute} may be out of frame -> {cls}")
            boxes.append(BoxLabel(class_name=cls, xyxy=clipped, confidence=sconf, source="auto", evidence=ev))

        # Near-threshold subjects that overlap no accepted subject are probably missed objects.
        acc_boxes = [b for b, _ in accepted.get(rule.subject, [])]
        missed = [c for b, c in near.get(rule.subject, [])
                  if not any(ioa(a, b) >= 0.5 for a in acc_boxes)]
        if missed:
            reasons.append(f"possible unlabelled {rule.subject} ({len(missed)} below threshold, max {max(missed):.2f})")

    # An attribute (e.g. a vest) inside no subject usually means the subject was missed.
    for rule in task.labeling.rules:
        if rule.kind != "attribute":
            continue
        subjects = [_pad(s, rule.subject_padding) for s, _ in accepted.get(rule.subject, [])]
        n_orphan = sum(1 for a, _ in accepted.get(rule.attribute, [])
                       if not any(ioa(s, a) >= rule.min_overlap for s in subjects))
        if n_orphan:
            reasons.append(f"{n_orphan} {rule.attribute} detection(s) without a {rule.subject} (missed {rule.subject}?)")
    return boxes, sorted(set(reasons))


# ---------------------------------------------------------------- prompt calibration
def calibrate_prompts(detector: Detector, image_paths: Sequence[str], task, sample_size: int = 16):
    """Pick, per detector prompt, the phrasing the detector responds to best on these frames.

    Detector recall is very sensitive to wording (measured: "vest" 23/26 frames vs
    "high-visibility safety vest" 3/26). Score = mean over sampled frames of the best
    detection confidence; for attribute prompts only detections inside a subject box
    count, so phrases that fire on background are not rewarded.
    Returns (task_with_chosen_texts, report). No ground truth is used, so the result
    is a heuristic and should be confirmed in the review UI.
    """
    from backend.task_engine.schema import DetectorPrompt

    prompts = task.labeling.detector_prompts
    if not any(p.alternatives for p in prompts) or not image_paths:
        return task, {}
    step = max(1, len(image_paths) // sample_size)
    sample = list(image_paths)[::step][:sample_size]
    attr_rules = {r.attribute: r for r in task.labeling.rules if r.kind == "attribute"}
    chosen: Dict[str, str] = {}
    report: Dict[str, dict] = {}
    subject_boxes: Dict[str, List[List[List[float]]]] = {}

    # Subjects / direct prompts first, then attributes constrained to subject boxes.
    for p in sorted(prompts, key=lambda q: q.key in attr_rules):
        rule = attr_rules.get(p.key)
        scores: Dict[str, float] = {}
        dets_by_text: Dict[str, list] = {}
        for text in dict.fromkeys([p.text, *p.alternatives]):
            probe = DetectorPrompt(key=p.key, text=text, min_confidence=0.02)
            dets = [d[p.key] for d in detector.detect(sample, [probe])]
            per_frame = []
            for i, frame_dets in enumerate(dets):
                if rule is not None:
                    subs = [_pad(s, rule.subject_padding) for s in subject_boxes.get(rule.subject, [[]] * len(sample))[i]]
                    frame_dets = [(b, c) for b, c in frame_dets if any(ioa(s, b) >= rule.min_overlap for s in subs)]
                per_frame.append(max([c for _, c in frame_dets] or [0.0]))
            scores[text] = round(sum(per_frame) / len(per_frame), 4)
            dets_by_text[text] = dets
        best = max(scores, key=scores.get)
        if scores[best] <= 0.0:
            best = p.text
        chosen[p.key] = best
        subject_boxes[p.key] = [[b for b, c in d if c >= p.min_confidence] for d in dets_by_text[best]]
        report[p.key] = {"chosen": best, "original": p.text, "scores": scores, "sample_frames": len(sample)}
        log.info("calibrated prompt %s: %r (scores %s)", p.key, best, scores)

    new_prompts = [p.model_copy(update={"text": chosen[p.key]}) for p in prompts]
    new_task = task.model_copy(update={"labeling": task.labeling.model_copy(update={"detector_prompts": new_prompts})})
    return new_task, report


# ---------------------------------------------------------------- entry
def auto_annotate(frames, task, detector: Optional[Detector] = None,
                  out_path: Optional[str] = None, calibrate: bool = True) -> AnnotationSet:
    """Label every frame according to the TaskSpec labeling plan."""
    if task.status != "ok":
        raise ValueError(f"TaskSpec status is '{task.status}', not 'ok': {task.issues}")
    if task.task_type != "object_detection":
        raise NotImplementedError("auto_annotate currently supports object_detection tasks only")
    frame_list: List[Frame] = frames.frames if isinstance(frames, FrameCollection) else list(frames)
    detector = detector or YoloWorldDetector()

    calibration = {}
    if calibrate:
        task, calibration = calibrate_prompts(detector, [f.path for f in frame_list if not f.synthetic], task)
    raws = detector.detect([f.path for f in frame_list], task.labeling.detector_prompts)
    anns: List[Annotation] = []
    for fr, raw in zip(frame_list, raws):
        boxes, reasons = apply_rules(raw, task, fr.width, fr.height)
        anns.append(Annotation(
            frame_id=fr.frame_id, image_path=fr.path, width=fr.width, height=fr.height,
            session_id=fr.session_id, synthetic=fr.synthetic, boxes=boxes,
            needs_review=bool(reasons), review_reasons=reasons,
        ))

    per_class = {c: 0 for c in task.class_names}
    for a in anns:
        for b in a.boxes:
            per_class[b.class_name] = per_class.get(b.class_name, 0) + 1
    stats = dict(frames=len(anns), boxes=sum(per_class.values()), per_class=per_class,
                 empty_frames=sum(1 for a in anns if not a.boxes),
                 needs_review=sum(1 for a in anns if a.needs_review),
                 detector_prompts={p.key: p.text for p in task.labeling.detector_prompts},
                 prompt_calibration=calibration)
    aset = AnnotationSet(task_id=task.task_id, class_names=task.class_names,
                         annotator=getattr(detector, "name", "custom"), annotations=anns, stats=stats)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(aset.model_dump_json(indent=2))
    log.info("auto_annotate: %s", stats)
    return aset
