"""P1: fal synthetic data. generate_synthetic(...) -> SyntheticDataset.

Pipeline: pick real TRAIN frames (rare class first) -> fal image edit for a target
condition (lighting, weather, distance...) -> sanity checks -> re-label with the same
detector -> reject edits whose labels disagree with the parent frame.

Rules enforced:
- frames from excluded (val / independent test) sessions are never used as parents;
- synthetic frames inherit the parent's session_id, so export_dataset() puts them in
  train only, and only when the parent session is in train;
- every image records provider, model, prompt, seed and parent id.
We do not assume synthetic data helps: export with and without it and let
evaluation on independent real footage decide.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

import cv2
import numpy as np
from pydantic import BaseModel

from .annotate import auto_annotate
from .ingest import dhash, hamming, sharpness
from .schema import Annotation, AnnotationSet, Frame, now_iso

log = logging.getLogger(__name__)

DEFAULT_FAL_MODEL = os.environ.get("VF_FAL_MODEL", "fal-ai/flux-pro/kontext")
PRESERVE = ("Keep the camera position, framing, scene layout and every person, their pose, clothing "
            "and safety equipment exactly the same. Photorealistic CCTV frame.")
DEFAULT_CONDITIONS = ["dim evening lighting", "harsh direct sunlight with strong shadows", "overcast flat light"]


class ImageEditor(Protocol):
    name: str

    def edit(self, image_path: str, prompt: str, seed: int) -> bytes: ...


class FalImageEditor:
    """fal image-editing backend (needs FAL_KEY in the environment)."""

    def __init__(self, model: str = DEFAULT_FAL_MODEL, timeout: float = 120.0):
        if not os.environ.get("FAL_KEY"):
            raise RuntimeError("FAL_KEY is not set; export it to use fal synthetic generation")
        import fal_client
        import httpx

        self._fal, self._http = fal_client, httpx
        self.model, self.timeout = model, timeout
        self.name = f"fal:{model}"

    def edit(self, image_path: str, prompt: str, seed: int) -> bytes:
        url = self._fal.upload_file(image_path)
        res = self._fal.subscribe(self.model, arguments={
            "prompt": prompt, "image_url": url, "seed": seed, "num_images": 1, "output_format": "jpeg",
        }, client_timeout=self.timeout)
        if any(res.get("has_nsfw_concepts") or []):
            raise RuntimeError("fal flagged the output as unsafe")
        return self._http.get(res["images"][0]["url"], timeout=60).content


class SyntheticRejection(BaseModel):
    parent_frame_id: str
    condition: str
    reason: str


class SyntheticDataset(BaseModel):
    frames: List[Frame]
    annotations: AnnotationSet
    rejected: List[SyntheticRejection]
    editor: str
    created_at: str


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def labels_consistent(parent: Annotation, child: Annotation, min_iou: float = 0.5) -> Optional[str]:
    """None if child labels match the parent's (same classes, boxes in place); else the reason."""
    sx, sy = child.width / parent.width, child.height / parent.height
    pboxes = [(b.class_name, [b.xyxy[0] * sx, b.xyxy[1] * sy, b.xyxy[2] * sx, b.xyxy[3] * sy]) for b in parent.boxes]
    if len(pboxes) != len(child.boxes):
        return f"box count changed {len(pboxes)} -> {len(child.boxes)}"
    unmatched = list(child.boxes)
    for cls, pb in pboxes:
        match = next((c for c in unmatched if c.class_name == cls and _iou(pb, c.xyxy) >= min_iou), None)
        if match is None:
            return f"no matching {cls} box after edit"
        unmatched.remove(match)
    return None


def _pick_parents(aset: AnnotationSet, frames: Dict[str, Frame], exclude: set, target: Optional[str],
                  max_parents: int) -> List[Annotation]:
    usable = [a for a in aset.annotations
              if not a.synthetic and a.status != "rejected" and a.session_id not in exclude
              and a.frame_id in frames and a.boxes and not a.needs_review]
    counts: Dict[str, int] = {}
    for a in usable:
        for b in a.boxes:
            counts[b.class_name] = counts.get(b.class_name, 0) + 1

    def rarity(a: Annotation) -> float:
        # Frames with the rarest classes (and the target event) first.
        s = min(counts.get(b.class_name, 1) for b in a.boxes)
        return s - (1000 if target and any(b.class_name == target for b in a.boxes) else 0)

    usable.sort(key=lambda a: (rarity(a), a.frame_id))
    return usable[:max_parents]


def generate_synthetic(
    annotations: AnnotationSet,
    frames: Sequence[Frame],
    task,
    out_dir,
    editor: Optional[ImageEditor] = None,
    detector=None,
    conditions: Optional[List[str]] = None,
    exclude_sessions: Sequence[str] = (),
    max_images: int = 20,
    seed: int = 0,
    min_iou: float = 0.5,
) -> SyntheticDataset:
    """Generate condition variants of real train frames, re-label and filter them.

    exclude_sessions: validation + independent-test sessions; never used as parents.
    detector: pass the same detector used for the real frames.
    """
    out = Path(out_dir).resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)
    editor = editor or FalImageEditor()
    conditions = conditions or list(task.synthetic_targets) or DEFAULT_CONDITIONS
    fidx = {f.frame_id: f for f in frames}
    exclude = set(exclude_sessions)
    # Re-label with exactly the detector phrasing chosen for the real frames.
    chosen = annotations.stats.get("detector_prompts") or {}
    prompts = [p.model_copy(update={"text": chosen.get(p.key, p.text)}) for p in task.labeling.detector_prompts]
    relabel_task = task.model_copy(update={"labeling": task.labeling.model_copy(update={"detector_prompts": prompts})})

    parents = _pick_parents(annotations, fidx, exclude, task.target_event, max_images)
    rejected: List[SyntheticRejection] = []
    candidates: List[Frame] = []
    for i, parent in enumerate(parents):
        pf = fidx[parent.frame_id]
        cond = conditions[i % len(conditions)]
        prompt = f"Change the scene to {cond}. {PRESERVE}"
        s = seed + i
        try:
            data = editor.edit(pf.path, prompt, s)
        except Exception as e:  # noqa: BLE001 - provider errors must not kill the batch
            rejected.append(SyntheticRejection(parent_frame_id=pf.frame_id, condition=cond, reason=f"editor error: {e}"))
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            rejected.append(SyntheticRejection(parent_frame_id=pf.frame_id, condition=cond, reason="undecodable image"))
            continue
        ph = cv2.imread(pf.path)
        h = dhash(img)
        if ph is not None and hamming(h, dhash(cv2.resize(ph, (img.shape[1], img.shape[0])))) <= 1:
            rejected.append(SyntheticRejection(parent_frame_id=pf.frame_id, condition=cond, reason="edit made no visible change"))
            continue
        fid = f"syn_{pf.frame_id}_{i}"
        path = out / "images" / f"{fid}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        candidates.append(Frame(
            frame_id=fid, path=str(path), session_id=pf.session_id, source_file=pf.source_file,
            source_frame_index=pf.source_frame_index, timestamp_s=pf.timestamp_s,
            width=img.shape[1], height=img.shape[0], sharpness=round(sharpness(img), 1), dhash=f"{h:016x}",
            synthetic=True, parent_frame_id=pf.frame_id,
            generation={"provider": getattr(editor, "name", "custom"), "prompt": prompt, "condition": cond,
                        "seed": s, "created_at": now_iso()},
        ))

    kept_frames: List[Frame] = []
    kept_anns: List[Annotation] = []
    if candidates:
        relabel = auto_annotate(candidates, relabel_task, detector=detector, calibrate=False)
        parent_ann = {a.frame_id: a for a in annotations.annotations}
        for fr, ann in zip(candidates, relabel.annotations):
            why = labels_consistent(parent_ann[fr.parent_frame_id], ann, min_iou)
            if why:
                rejected.append(SyntheticRejection(parent_frame_id=fr.parent_frame_id,
                                                   condition=fr.generation["condition"], reason=why))
                Path(fr.path).unlink(missing_ok=True)
                continue
            kept_frames.append(fr)
            kept_anns.append(ann)

    aset = AnnotationSet(task_id=task.task_id, class_names=task.class_names,
                         annotator=annotations.annotator, annotations=kept_anns,
                         stats={"generated": len(candidates), "kept": len(kept_anns), "rejected": len(rejected)})
    ds = SyntheticDataset(frames=kept_frames, annotations=aset, rejected=rejected,
                          editor=getattr(editor, "name", "custom"), created_at=now_iso())
    (out / "synthetic.json").write_text(ds.model_dump_json(indent=2))
    log.info("generate_synthetic: %d parents, %d kept, %d rejected", len(parents), len(kept_anns), len(rejected))
    return ds


def merge(real: AnnotationSet, synth: SyntheticDataset) -> AnnotationSet:
    """Real + synthetic annotations, ready for export_dataset(frames=real_frames + synth.frames)."""
    out = real.model_copy(deep=True)
    out.annotations += synth.annotations.annotations
    return out
