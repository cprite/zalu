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
from typing import Dict, List, Literal, Optional, Protocol, Sequence

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .annotate import auto_annotate
from .ingest import dhash, sharpness
from .schema import Annotation, AnnotationSet, Frame, now_iso

log = logging.getLogger(__name__)

DEFAULT_FAL_MODEL = os.environ.get("VF_FAL_MODEL", "fal-ai/flux-pro/kontext")
PRESERVE = ("Keep the camera position, framing, scene layout and every person, their pose, clothing "
            "and safety equipment exactly the same. Do not add any text, timestamps, logos or overlays.")
# Aspect ratios accepted by fal-ai/flux-pro/kontext.
ASPECTS = {"21:9": 21 / 9, "16:9": 16 / 9, "4:3": 4 / 3, "3:2": 3 / 2, "1:1": 1.0,
           "2:3": 2 / 3, "3:4": 3 / 4, "9:16": 9 / 16, "9:21": 9 / 21}


def nearest_aspect(width: int, height: int) -> str:
    r = width / height
    return min(ASPECTS, key=lambda k: abs(ASPECTS[k] - r))
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
        img = cv2.imread(image_path)
        url = self._fal.upload_file(image_path)
        res = self._fal.subscribe(self.model, arguments={
            "prompt": prompt, "image_url": url, "seed": seed, "num_images": 1, "output_format": "jpeg",
            "aspect_ratio": nearest_aspect(img.shape[1], img.shape[0]),
        }, client_timeout=self.timeout)
        if any(res.get("has_nsfw_concepts") or []):
            raise RuntimeError("fal flagged the output as unsafe")
        return self._http.get(res["images"][0]["url"], timeout=60).content


class SyntheticRejection(BaseModel):
    parent_frame_id: str
    condition: str
    reason: str
    image: Optional[str] = Field(default=None, description="Rejected edit kept for audit (never exported)")


def _park(path: str, out: Path) -> Path:
    """Move a rejected edit to <out>/rejected/ so it can be inspected but never reaches a dataset."""
    dst = out / "rejected" / Path(path).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    Path(path).replace(dst)
    return dst


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


def labels_consistent(parent: Annotation, child: Annotation, min_iou: float = 0.5,
                      class_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """None if child labels match the parent's after applying `class_map`
    (expected label changes, e.g. {"person_with_vest": "person_without_vest"}); else the reason."""
    class_map = class_map or {}
    sx, sy = child.width / parent.width, child.height / parent.height
    pboxes = [(class_map.get(b.class_name, b.class_name),
               [b.xyxy[0] * sx, b.xyxy[1] * sy, b.xyxy[2] * sx, b.xyxy[3] * sy]) for b in parent.boxes]
    if len(pboxes) != len(child.boxes):
        return f"box count changed {len(pboxes)} -> {len(child.boxes)}"
    unmatched = list(child.boxes)
    for cls, pb in pboxes:
        match = next((c for c in unmatched if c.class_name == cls and _iou(pb, c.xyxy) >= min_iou), None)
        if match is None:
            got = [c.class_name for c in unmatched if _iou(pb, c.xyxy) >= min_iou]
            return f"expected {cls} after edit, got {got or 'no box'}"
        unmatched.remove(match)
    return None


# ---------------------------------------------------------------- edit specs
class EditSpec(BaseModel):
    """One kind of edit. appearance: labels must stay the same. counterfactual: labels must
    change exactly as `class_map` says (verified by re-detecting), e.g. vest removed."""

    kind: Literal["appearance", "counterfactual"]
    condition: str
    prompt: str
    class_map: Dict[str, str] = {}
    source_class: Optional[str] = Field(default=None, description="Parent frame must contain this class")


def appearance_edit(condition: str) -> EditSpec:
    return EditSpec(kind="appearance", condition=condition, prompt=f"Change the scene to {condition}. {PRESERVE}")


def counterfactual_edits(task, detector_texts: Optional[Dict[str, str]] = None) -> List[EditSpec]:
    """Derive label-flipping edits from the TaskSpec's attribute rules.

    For rule (subject=person, attribute=vest): "remove the vest" turns class_if_present into
    class_if_absent, "add a vest" does the reverse. Edits that produce the target_event come first.
    """
    texts = {p.key: p.text for p in task.labeling.detector_prompts}
    texts.update(detector_texts or {})
    keep = ("Keep the camera position, framing, background, lighting and every person's position and pose "
            "exactly the same. Do not add any text, timestamps, logos or overlays.")
    out: List[EditSpec] = []
    for r in task.labeling.rules:
        if r.kind != "attribute":
            continue
        subj, attr = texts.get(r.subject, r.subject), texts.get(r.attribute, r.attribute)
        out.append(EditSpec(
            kind="counterfactual", condition=f"remove {attr}", source_class=r.class_if_present,
            class_map={r.class_if_present: r.class_if_absent},
            prompt=f"Remove the {attr} from every {subj} and show ordinary everyday clothing in a plain, "
                   f"non-reflective colour instead, so that no {attr} is visible anywhere. {keep}"))
        out.append(EditSpec(
            kind="counterfactual", condition=f"add {attr}", source_class=r.class_if_absent,
            class_map={r.class_if_absent: r.class_if_present},
            prompt=f"Give every {subj} a clearly visible {attr}. {keep}"))
    out.sort(key=lambda e: task.target_event not in e.class_map.values())
    return out


# ---------------------------------------------------------------- checks
def edit_strength(parent_img: np.ndarray, edit_img: np.ndarray, boxes) -> Dict[str, float]:
    """Mean absolute LAB difference globally and inside the most-changed labelled box (0-255 scale).

    Measured on real fal-ai/flux-pro/kontext output: a JPEG re-encode (no-op) scores ~0.5,
    any real edit scores >10 because the model re-renders the whole frame. So this only
    catches literal no-ops; whether the requested condition was met needs a verifier.
    """
    W, H = 320, max(1, round(320 * parent_img.shape[0] / parent_img.shape[1]))
    a = cv2.cvtColor(cv2.resize(parent_img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(edit_img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.abs(a - b)
    sx, sy = W / parent_img.shape[1], H / parent_img.shape[0]
    box = 0.0
    for bx in boxes:
        x1, y1, x2, y2 = int(bx.xyxy[0] * sx), int(bx.xyxy[1] * sy), int(bx.xyxy[2] * sx), int(bx.xyxy[3] * sy)
        if x2 > x1 and y2 > y1:
            box = max(box, float(d[y1:y2, x1:x2].mean()))
    return {"global": round(float(d.mean()), 2), "max_box": round(box, 2)}


class Verdict(BaseModel):
    ok: bool
    reason: str = ""


class EditVerifier(Protocol):
    name: str

    def verify(self, parent_path: str, edit_path: str, spec: EditSpec) -> Verdict: ...


class _VisionJudgement(BaseModel):
    condition_met: bool = Field(description="The requested change is clearly visible in the EDITED image")
    scene_preserved: bool = Field(description="Same camera view, layout, number and positions of people")
    unwanted_additions: List[str] = Field(description="Text, timestamps, logos, extra people/objects added; empty if none")
    explanation: str


class ClaudeEditVerifier:
    """Claude vision check: was the requested edit really made, and nothing else? Needs ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-opus-5-5", client=None):
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.model = model
        self.name = f"claude:{model}"

    @staticmethod
    def _b64(path: str) -> str:
        import base64

        img = cv2.imread(path)
        if img.shape[1] > 1024:
            img = cv2.resize(img, (1024, round(1024 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()

    def verify(self, parent_path: str, edit_path: str, spec: EditSpec) -> Verdict:
        img = lambda p: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": self._b64(p)}}  # noqa: E731
        resp = self.client.beta.messages.parse(
            model=self.model, max_tokens=4000,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_format=_VisionJudgement,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "ORIGINAL camera frame:"}, img(parent_path),
                {"type": "text", "text": "EDITED version:"}, img(edit_path),
                {"type": "text", "text":
                    f"The edit instruction was: \"{spec.prompt}\"\n"
                    "These images become training data for an object detector, so be strict. Judge whether the "
                    "requested change is clearly visible (a barely noticeable change does not count), whether the "
                    "scene is otherwise preserved, and list anything unwanted the edit added."},
            ]}],
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return Verdict(ok=False, reason=f"verifier gave no judgement ({resp.stop_reason})")
        j = resp.parsed_output
        problems = ([] if j.condition_met else ["requested change not visible"]) + \
                   ([] if j.scene_preserved else ["scene not preserved"]) + \
                   [f"added: {u}" for u in j.unwanted_additions]
        return Verdict(ok=not problems, reason="; ".join(problems) or j.explanation)


# ---------------------------------------------------------------- generation
MIN_EDIT_STRENGTH = 2.0  # see edit_strength(): no-op ~0.5, real edits >10


def _pick_pool(aset: AnnotationSet, frames: Dict[str, Frame], exclude: set, target: Optional[str]) -> List[Annotation]:
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
    return usable


def _plan(pool: List[Annotation], edits: List[EditSpec], max_images: int):
    """Round-robin over edits; each edit takes the next eligible parent it hasn't used."""
    plan, used = [], {i: set() for i in range(len(edits))}
    starved = set()
    i = 0
    while len(plan) < max_images and len(starved) < len(edits):
        k = i % len(edits)
        i += 1
        if k in starved:
            continue
        spec = edits[k]
        parent = next((a for a in pool if a.frame_id not in used[k]
                       and (spec.source_class is None or any(b.class_name == spec.source_class for b in a.boxes))), None)
        if parent is None:
            starved.add(k)
            continue
        used[k].add(parent.frame_id)
        plan.append((parent, spec))
    return plan, [edits[k] for k in sorted(starved) if not used[k]]


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
    counterfactual: bool = False,
    edits: Optional[List[EditSpec]] = None,
    verifier: Optional[EditVerifier] = None,
) -> SyntheticDataset:
    """Generate variants of real train frames, re-label and filter them.

    conditions:      appearance edits (lighting, weather...); labels must not change.
    counterfactual:  also add label-flipping edits derived from the TaskSpec (e.g. remove the
                     vest -> person_without_vest); kept only if re-detection confirms the flip.
                     With counterfactual=True and no explicit conditions, only counterfactuals run.
    edits:           explicit EditSpec list (overrides conditions/counterfactual).
    verifier:        optional vision check that the requested change is really visible.
    exclude_sessions: validation + independent-test sessions; never used as parents.
    detector:        pass the same detector used for the real frames.
    """
    out = Path(out_dir).resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)
    editor = editor or FalImageEditor()
    # Re-label with exactly the detector phrasing chosen for the real frames.
    chosen = annotations.stats.get("detector_prompts") or {}
    prompts = [p.model_copy(update={"text": chosen.get(p.key, p.text)}) for p in task.labeling.detector_prompts]
    relabel_task = task.model_copy(update={"labeling": task.labeling.model_copy(update={"detector_prompts": prompts})})

    if edits is None:
        edits = []
        if conditions or not counterfactual:
            edits += [appearance_edit(c) for c in (conditions or list(task.synthetic_targets) or DEFAULT_CONDITIONS)]
        if counterfactual:
            edits += counterfactual_edits(task, chosen)
    if not edits:
        raise ValueError("no edits requested")

    fidx = {f.frame_id: f for f in frames}
    pool = _pick_pool(annotations, fidx, set(exclude_sessions), task.target_event)
    plan, starved = _plan(pool, edits, max_images)
    rejected: List[SyntheticRejection] = [
        SyntheticRejection(parent_frame_id="-", condition=e.condition,
                           reason=f"no eligible parent frame containing {e.source_class}") for e in starved]

    candidates: List[tuple] = []
    for i, (parent, spec) in enumerate(plan):
        pf = fidx[parent.frame_id]
        s = seed + i

        def reject(reason: str):
            rejected.append(SyntheticRejection(parent_frame_id=pf.frame_id, condition=spec.condition, reason=reason))

        try:
            data = editor.edit(pf.path, spec.prompt, s)
        except Exception as e:  # noqa: BLE001 - provider errors must not kill the batch
            reject(f"editor error: {e}")
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            reject("undecodable image")
            continue
        if (img.shape[1], img.shape[0]) != (pf.width, pf.height):
            # Editors return their own resolution; keep the parent's so frames are interchangeable.
            img = cv2.resize(img, (pf.width, pf.height), interpolation=cv2.INTER_AREA)
        ph = cv2.imread(pf.path)
        strength = edit_strength(ph, img, parent.boxes) if ph is not None else {}
        if strength and max(strength.values()) < MIN_EDIT_STRENGTH:
            reject(f"edit made no visible change (strength {strength})")
            continue
        fid = f"syn_{pf.frame_id}_{i}"
        path = out / "images" / f"{fid}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        candidates.append((spec, Frame(
            frame_id=fid, path=str(path), session_id=pf.session_id, source_file=pf.source_file,
            source_frame_index=pf.source_frame_index, timestamp_s=pf.timestamp_s,
            width=img.shape[1], height=img.shape[0], sharpness=round(sharpness(img), 1), dhash=f"{dhash(img):016x}",
            synthetic=True, parent_frame_id=pf.frame_id,
            generation={"provider": getattr(editor, "name", "custom"), "kind": spec.kind, "prompt": spec.prompt,
                        "condition": spec.condition, "class_map": spec.class_map, "seed": s,
                        "edit_strength": strength, "created_at": now_iso()},
        )))

    kept_frames: List[Frame] = []
    kept_anns: List[Annotation] = []
    if candidates:
        relabel = auto_annotate([f for _, f in candidates], relabel_task, detector=detector, calibrate=False)
        parent_ann = {a.frame_id: a for a in annotations.annotations}
        for (spec, fr), ann in zip(candidates, relabel.annotations):
            why = labels_consistent(parent_ann[fr.parent_frame_id], ann, min_iou, spec.class_map)
            if why is None and ann.needs_review:
                why = "re-labelled edit is uncertain: " + "; ".join(ann.review_reasons)
            if why is None and verifier is not None:
                try:
                    v = verifier.verify(fidx[fr.parent_frame_id].path, fr.path, spec)
                except Exception as e:  # noqa: BLE001
                    v = Verdict(ok=False, reason=f"verifier error: {e}")
                fr.generation["verifier"] = {"name": getattr(verifier, "name", "custom"), **v.model_dump()}
                if not v.ok:
                    why = f"verifier: {v.reason}"
            if why:
                rejected.append(SyntheticRejection(parent_frame_id=fr.parent_frame_id,
                                                   condition=spec.condition, reason=why, image=str(_park(fr.path, out))))
                continue
            kept_frames.append(fr)
            kept_anns.append(ann)

    per_class: Dict[str, int] = {}
    for a in kept_anns:
        for b in a.boxes:
            per_class[b.class_name] = per_class.get(b.class_name, 0) + 1
    aset = AnnotationSet(task_id=task.task_id, class_names=task.class_names,
                         annotator=annotations.annotator, annotations=kept_anns,
                         stats={"generated": len(candidates), "kept": len(kept_anns), "rejected": len(rejected),
                                "kept_per_class": per_class, "verifier": getattr(verifier, "name", None)})
    ds = SyntheticDataset(frames=kept_frames, annotations=aset, rejected=rejected,
                          editor=getattr(editor, "name", "custom"), created_at=now_iso())
    (out / "synthetic.json").write_text(ds.model_dump_json(indent=2))
    log.info("generate_synthetic: %d planned, %d kept, %d rejected", len(plan), len(kept_anns), len(rejected))
    return ds


def merge(real: AnnotationSet, synth: SyntheticDataset) -> AnnotationSet:
    """Real + synthetic annotations, ready for export_dataset(frames=real_frames + synth.frames)."""
    out = real.model_copy(deep=True)
    out.annotations += synth.annotations.annotations
    return out
