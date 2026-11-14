from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.data_engine import auto_annotate, export_dataset, validate_dataset
from backend.data_engine.schema import Frame
from backend.data_engine.synthetic import generate_synthetic, merge


class ColourBoxDetector:
    """'person' = bounding box of green pixels; 'vest' = box of the orange patch."""

    name = "colour"

    def detect(self, paths, prompts):
        out = []
        for p in paths:
            img = cv2.imread(p)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            d = {}
            for pr in prompts:
                lo, hi = ((45, 80, 40), (75, 255, 255)) if pr.key == "person" else ((5, 120, 60), (20, 255, 255))
                ys, xs = np.nonzero(cv2.inRange(hsv, lo, hi))
                d[pr.key] = [([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], 0.9)] if len(xs) > 50 else []
            out.append(d)
        return out


class FakeEditor:
    name = "fake-editor"

    def edit(self, path, prompt, seed):
        img = cv2.imread(path)
        if "mirror" in prompt:
            img = cv2.flip(img, 1)
        elif "dim" in prompt:
            img = cv2.convertScaleAbs(img, alpha=0.6, beta=0)
        elif "boom" in prompt:
            raise RuntimeError("provider down")
        elif "Remove the vest" in prompt:
            if "fail" not in path:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img[cv2.inRange(hsv, (5, 120, 60), (20, 255, 255)) > 0] = (0, 200, 0)  # vest painted as shirt
            img = cv2.convertScaleAbs(img, alpha=0.95, beta=8)  # re-render noise
        return cv2.imencode(".jpg", img)[1].tobytes()


def _frames(tmp_path):
    rng = np.random.default_rng(0)
    frames = []
    for s in ("s_a", "s_b", "s_val"):
        for k in range(4):
            img = cv2.cvtColor((rng.random((240, 320)) * 60).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img, (40, 40), (110, 200), (0, 200, 0), -1)  # person (green)
            if k % 2 == 0:
                cv2.rectangle(img, (55, 80), (95, 130), (0, 128, 255), -1)  # vest (orange)
            p = tmp_path / "frames" / f"{s}_{k}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), img)
            frames.append(Frame(frame_id=f"{s}_{k}", path=str(p), session_id=s, source_file=f"{s}.mp4",
                                timestamp_s=float(k), width=320, height=240, sharpness=100, dhash="0" * 16))
    return frames


def test_generate_filter_and_export(tmp_path, vest_task):
    frames = _frames(tmp_path)
    det = ColourBoxDetector()
    real = auto_annotate(frames, vest_task, detector=det, calibrate=False)
    syn = generate_synthetic(real, frames, vest_task, tmp_path / "syn", editor=FakeEditor(), detector=det,
                             conditions=["dim light", "mirror world", "boom", "identical"],
                             exclude_sessions=["s_val"], max_images=8)

    reasons = {r.condition: r.reason for r in syn.rejected}
    assert "editor error" in reasons["boom"]
    assert "expected" in reasons["mirror world"] or "count" in reasons["mirror world"]
    assert reasons["identical"].startswith("edit made no visible change")
    assert syn.frames and all(f.generation["condition"] == "dim light" for f in syn.frames)
    assert all(f.session_id != "s_val" and f.synthetic and f.parent_frame_id for f in syn.frames)
    assert all(f.generation["seed"] is not None and "Keep the camera" in f.generation["prompt"] for f in syn.frames)

    merged = merge(real, syn)
    m = export_dataset(merged, vest_task, tmp_path / "ds", frames=frames + syn.frames,
                       val_sessions=["s_val"])
    assert m.variant == "real_plus_synthetic"
    assert all(i.split == "train" for i in m.images if i.synthetic)
    assert validate_dataset(m.root).ok

    m0 = export_dataset(merged, vest_task, tmp_path / "ds0", frames=frames + syn.frames,
                        val_sessions=["s_val"], include_synthetic=False)
    assert m0.variant == "real_only"


def test_rare_target_class_prioritised(tmp_path, vest_task):
    frames = _frames(tmp_path)
    det = ColourBoxDetector()
    real = auto_annotate(frames, vest_task, detector=det, calibrate=False)
    syn = generate_synthetic(real, frames, vest_task, tmp_path / "syn", editor=FakeEditor(), detector=det,
                             conditions=["dim light"], max_images=2)
    parents = {f.parent_frame_id for f in syn.frames}
    parent_classes = {b.class_name for a in real.annotations if a.frame_id in parents for b in a.boxes}
    assert parent_classes == {vest_task.target_event}


def test_fal_editor_requires_key(monkeypatch):
    from backend.data_engine.synthetic import FalImageEditor
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FAL_KEY"):
        FalImageEditor()


def test_nearest_aspect():
    from backend.data_engine.synthetic import nearest_aspect
    assert nearest_aspect(1280, 720) == "16:9"
    assert nearest_aspect(768, 432) == "16:9"
    assert nearest_aspect(640, 480) == "4:3"
    assert nearest_aspect(720, 1280) == "9:16"


def test_output_resized_to_parent(tmp_path, vest_task):
    frames = _frames(tmp_path)
    det = ColourBoxDetector()
    real = auto_annotate(frames, vest_task, detector=det, calibrate=False)

    class BigEditor(FakeEditor):
        def edit(self, path, prompt, seed):
            img = cv2.convertScaleAbs(cv2.imread(path), alpha=0.6)
            return cv2.imencode(".jpg", cv2.resize(img, (348, 188)))[1].tobytes()

    syn = generate_synthetic(real, frames, vest_task, tmp_path / "syn", editor=BigEditor(), detector=det,
                             conditions=["dim"], max_images=2)
    assert syn.frames and all((f.width, f.height) == (320, 240) for f in syn.frames)


def test_counterfactual_edits_from_rules(vest_task):
    from backend.data_engine.synthetic import counterfactual_edits
    edits = counterfactual_edits(vest_task)
    assert [e.condition for e in edits] == ["remove vest", "add vest"]  # target-event producer first
    assert edits[0].class_map == {"person_with_vest": "person_without_vest"}
    assert edits[0].source_class == "person_with_vest"
    assert "without" not in edits[0].prompt.split("so that")[0]


def test_counterfactual_kept_only_when_flip_confirmed(tmp_path, vest_task):
    from backend.data_engine.synthetic import counterfactual_edits
    frames = _frames(tmp_path)
    # make one with-vest parent "fail": the fake editor leaves its vest in place
    bad = next(f for f in frames if f.frame_id == "s_b_0")
    newp = Path(bad.path).with_name("fail_s_b_0.jpg"); Path(bad.path).rename(newp); bad.path = str(newp)
    det = ColourBoxDetector()
    real = auto_annotate(frames, vest_task, detector=det, calibrate=False)
    remove = counterfactual_edits(vest_task)[:1]
    syn = generate_synthetic(real, frames, vest_task, tmp_path / "syn", editor=FakeEditor(), detector=det,
                             edits=remove, max_images=6)
    assert syn.frames, syn.rejected
    for f, a in zip(syn.frames, syn.annotations.annotations):
        assert [b.class_name for b in a.boxes] == ["person_without_vest"]
        assert f.generation["kind"] == "counterfactual"
    fail = [r for r in syn.rejected if r.parent_frame_id == "s_b_0"]
    assert fail and "expected person_without_vest" in fail[0].reason


def test_verifier_can_reject(tmp_path, vest_task):
    from backend.data_engine.synthetic import Verdict
    frames = _frames(tmp_path)
    det = ColourBoxDetector()
    real = auto_annotate(frames, vest_task, detector=det, calibrate=False)

    class NoVerifier:
        name = "strict"
        def verify(self, parent, edit, spec):
            return Verdict(ok=False, reason="requested change not visible")

    syn = generate_synthetic(real, frames, vest_task, tmp_path / "syn", editor=FakeEditor(), detector=det,
                             conditions=["dim light"], max_images=2, verifier=NoVerifier())
    assert not syn.frames
    assert all(r.reason.startswith("verifier:") for r in syn.rejected)


def test_plan_reports_edits_without_parents(tmp_path, vest_task):
    from backend.data_engine.synthetic import counterfactual_edits
    frames = _frames(tmp_path)
    det = ColourBoxDetector()
    real = auto_annotate(frames, vest_task, detector=det, calibrate=False)
    for a in real.annotations:  # no frames without a vest -> "add vest" has no parents... and vice versa
        a.boxes = [b for b in a.boxes if b.class_name == "person_with_vest"]
    syn = generate_synthetic(real, frames, vest_task, tmp_path / "syn", editor=FakeEditor(), detector=det,
                             edits=counterfactual_edits(vest_task), max_images=2)
    assert any("no eligible parent frame containing person_without_vest" in r.reason for r in syn.rejected)
