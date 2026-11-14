import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from backend.data_engine import (BoxLabel, CorrectionError, ExportError, LabelCorrection, apply_corrections,
                                 apply_rules, auto_annotate, export_dataset, export_test_set, validate_dataset,
                                 check_leakage)
from backend.data_engine.export import TEST_TRAIN_SENTINEL, yolo_line
from backend.data_engine.schema import Frame

from backend.data_engine.tests.helpers import FakeDetector

PERSON = [100.0, 50.0, 200.0, 250.0]
VEST_IN = [120.0, 100.0, 180.0, 170.0]


# ------------------------------------------------------------------ rules
def test_attribute_rule_with_and_without(vest_task):
    boxes, reasons = apply_rules({"person": [(PERSON, 0.9)], "vest": [(VEST_IN, 0.8)]}, vest_task, 640, 480)
    assert [b.class_name for b in boxes] == ["person_with_vest"]
    assert not reasons

    boxes, reasons = apply_rules({"person": [(PERSON, 0.9)], "vest": []}, vest_task, 640, 480)
    assert [b.class_name for b in boxes] == ["person_without_vest"]


def test_flags_low_conf_borderline_and_orphans(vest_task):
    half_out = [170.0, 100.0, 230.0, 170.0]  # 50% inside -> 0.5 boundary; shift a bit more
    mostly_out = [180.0, 100.0, 240.0, 170.0]  # 33% inside
    boxes, reasons = apply_rules({"person": [(PERSON, 0.35)], "vest": [(mostly_out, 0.8)]},
                                 vest_task, 640, 480)
    assert boxes[0].class_name == "person_without_vest"
    assert any("borderline" in r for r in reasons)
    assert any("low-confidence person" in r for r in reasons)

    _, reasons = apply_rules({"person": [], "vest": [(VEST_IN, 0.8)]}, vest_task, 640, 480)
    assert any("without a person" in r for r in reasons)


def test_boxes_are_clipped_and_degenerate_dropped(vest_task):
    boxes, _ = apply_rules({"person": [([-20, -10, 50, 600], 0.9), ([10, 10, 12, 12], 0.9)],
                            "vest": []}, vest_task, 320, 240)
    assert len(boxes) == 1
    assert boxes[0].xyxy == [0.0, 0.0, 50.0, 240.0]


def test_rejects_non_ok_task(vest_task):
    t = vest_task.model_copy(update={"status": "needs_clarification"})
    with pytest.raises(ValueError):
        auto_annotate([], t, detector=FakeDetector(lambda i, p: {}))


# ------------------------------------------------------------------ fixtures
def _frames(tmp_path, sessions, per_session=6, size=(320, 240)):
    rng = np.random.default_rng(0)
    frames = []
    for s in sessions:
        for k in range(per_session):
            p = tmp_path / "frames" / s / f"{s}_{k}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8))
            frames.append(Frame(frame_id=f"{s}_{k}", path=str(p), session_id=s, source_file=f"{s}.mp4",
                                source_frame_index=k * 10, timestamp_s=float(k), width=size[0], height=size[1],
                                sharpness=100.0, dhash="0" * 16))
    return frames


def _annotate(frames, task):
    def fn(i, path):
        # alternate vest / no vest; every 4th frame empty
        if i % 4 == 3:
            return {"person": [], "vest": []}
        vest = [(VEST_IN, 0.8)] if i % 2 == 0 else []
        return {"person": [(PERSON, 0.9)], "vest": vest}
    return auto_annotate(frames, task, detector=FakeDetector(fn))


# ------------------------------------------------------------------ corrections
def test_corrections(tmp_path, vest_task):
    frames = _frames(tmp_path, ["s1"], 3)
    aset = _annotate(frames, vest_task)
    fixed = apply_corrections(aset, [
        LabelCorrection(frame_id="s1_0", boxes=[BoxLabel(class_name="person_without_vest", xyxy=[1, 1, 50, 90])]),
        LabelCorrection(frame_id="s1_1", status="rejected"),
    ])
    a0 = fixed.annotations[0]
    assert a0.status == "reviewed" and a0.boxes[0].source == "manual" and a0.boxes[0].confidence is None
    assert fixed.annotations[1].status == "rejected"
    assert aset.annotations[0].status == "auto"  # original untouched
    with pytest.raises(CorrectionError):
        apply_corrections(aset, [LabelCorrection(frame_id="s1_0", boxes=[BoxLabel(class_name="cat", xyxy=[1, 1, 9, 9])])])
    with pytest.raises(CorrectionError):
        apply_corrections(aset, [LabelCorrection(frame_id="nope")])


# ------------------------------------------------------------------ export
def test_export_splits_by_session_and_excludes_held_out(tmp_path, vest_task):
    frames = _frames(tmp_path, ["s_a", "s_b", "s_c", "s_test"])
    aset = _annotate(frames, vest_task)
    m = export_dataset(aset, vest_task, tmp_path / "ds", frames=frames, held_out_sessions=["s_test"])

    tr, va = set(m.splits["train"].sessions), set(m.splits["val"].sessions)
    assert tr and va and not (tr & va)
    assert "s_test" not in tr | va
    assert m.excluded_sessions == ["s_test"]
    assert m.split_strategy == "by_session(auto)"
    data = yaml.safe_load(Path(m.data_yaml).read_text())
    assert data["names"] == {0: "person_with_vest", 1: "person_without_vest"}

    rep = validate_dataset(m.root)
    assert rep.ok, rep.errors
    assert rep.stats["train"]["images"] + rep.stats["val"]["images"] == 18

    t = export_test_set(aset, vest_task, tmp_path / "test", ["s_test"], frames=frames)
    assert yaml.safe_load(Path(t.data_yaml).read_text())["train"] == TEST_TRAIN_SENTINEL
    assert validate_dataset(t.root).ok
    assert check_leakage(m.root, t.root) == []


def test_single_session_uses_temporal_block_with_warning(tmp_path, vest_task):
    frames = _frames(tmp_path, ["only"], 10)
    m = export_dataset(_annotate(frames, vest_task), vest_task, tmp_path / "ds", frames=frames)
    assert m.split_strategy.startswith("temporal_block")
    assert any("only one recording session" in w for w in m.warnings)
    tr = [i for i in m.images if i.split == "train"]
    va = [i for i in m.images if i.split == "val"]
    assert len(va) == 2 and len(tr) == 7  # one boundary frame dropped
    last_train = max(int(i.frame_id.split("_")[1]) for i in tr)
    first_val = min(int(i.frame_id.split("_")[1]) for i in va)
    assert first_val - last_train == 2


def test_synthetic_only_in_train_and_only_from_train_parents(tmp_path, vest_task):
    frames = _frames(tmp_path, ["s_a", "s_b", "s_c"])
    aset = _annotate(frames, vest_task)
    synth = []
    for parent in (frames[0], frames[-1]):  # s_a (smallest-first -> val) and s_c (train)
        a = aset.annotations[frames.index(parent)].model_copy(deep=True)
        a.frame_id, a.synthetic = f"syn_{parent.frame_id}", True
        synth.append(a)
    aset.annotations += synth
    m = export_dataset(aset, vest_task, tmp_path / "ds", frames=frames, val_sessions=["s_a"])
    syn = [i for i in m.images if i.synthetic]
    assert [i.frame_id for i in syn] == ["syn_s_c_5"]
    assert all(i.split == "train" for i in syn)
    assert m.variant == "real_plus_synthetic"
    assert any("synthetic image(s) dropped" in w for w in m.warnings)

    m2 = export_dataset(aset, vest_task, tmp_path / "ds2", frames=frames, val_sessions=["s_a"], include_synthetic=False)
    assert m2.variant == "real_only" and not any(i.synthetic for i in m2.images)


def test_export_guards(tmp_path, vest_task):
    frames = _frames(tmp_path, ["s_a", "s_b"])
    aset = _annotate(frames, vest_task)
    with pytest.raises(ExportError):
        export_dataset(aset, vest_task, tmp_path / "x", val_sessions=["s_a"], held_out_sessions=["s_a"])
    with pytest.raises(ExportError):
        export_dataset(aset, vest_task, tmp_path / "y", val_sessions=["s_a", "s_b"])
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "f").write_text("x")
    with pytest.raises(ExportError):
        export_dataset(aset, vest_task, tmp_path / "busy")


def test_yolo_line():
    assert yolo_line(1, [0, 0, 160, 120], 320, 240) == "1 0.250000 0.250000 0.500000 0.500000"


# ------------------------------------------------------------------ validator
def test_validator_catches_bad_labels_and_synthetic_in_val(tmp_path, vest_task):
    frames = _frames(tmp_path, ["s_a", "s_b"])
    m = export_dataset(_annotate(frames, vest_task), vest_task, tmp_path / "ds", frames=frames)
    root = Path(m.root)
    lbl = next((root / "labels" / "train").glob("*.txt"))
    lbl.write_text("7 0.5 0.5 0.2 0.2\n0 1.4 0.5 0.2 0.2\n0 0.5\n")
    man = json.loads((root / "manifest.json").read_text())
    vi = next(i for i in man["images"] if i["split"] == "val")
    vi["synthetic"] = True
    (root / "manifest.json").write_text(json.dumps(man))

    rep = validate_dataset(root)
    assert not rep.ok
    joined = " | ".join(rep.errors)
    assert "class id 7" in joined and "not normalised" in joined and "expected 5 fields" in joined
    assert "synthetic image" in joined


def test_ultralytics_accepts_dataset(tmp_path, vest_task):
    """The export must load with stock Ultralytics, no VisionForge code involved."""
    from ultralytics.data.utils import check_det_dataset

    frames = _frames(tmp_path, ["s_a", "s_b"])
    m = export_dataset(_annotate(frames, vest_task), vest_task, tmp_path / "ds", frames=frames)
    info = check_det_dataset(m.data_yaml)
    assert info["nc"] == 2 and Path(info["train"]).is_dir() and Path(info["val"]).is_dir()


# ------------------------------------------------------------------ review flags & calibration
def test_near_threshold_detections_only_flag(vest_task):
    # person min_conf 0.3: a 0.2 person is not labelled but flagged; a 0.1 vest (min 0.15) inside the person is flagged
    raw = {"person": [(PERSON, 0.9), ([400, 50, 480, 250], 0.2)], "vest": [(VEST_IN, 0.1)]}
    boxes, reasons = apply_rules(raw, vest_task, 640, 480)
    assert [b.class_name for b in boxes] == ["person_without_vest"]
    assert any("possible unlabelled person" in r for r in reasons)
    assert any("possible vest below threshold" in r for r in reasons)


def test_edge_truncated_subject_flagged(vest_task):
    boxes, reasons = apply_rules({"person": [([500, 300, 640, 480], 0.9)], "vest": []}, vest_task, 640, 480)
    assert boxes[0].class_name == "person_without_vest"
    assert any("cut off at image edge" in r for r in reasons)


def test_calibration_picks_phrase_detector_responds_to(tmp_path, vest_task):
    from backend.data_engine.annotate import calibrate_prompts

    class PhraseDetector:
        name = "phrase"

        def detect(self, paths, prompts):
            out = []
            for _ in paths:
                d = {}
                for p in prompts:
                    if p.text == "person":
                        d[p.key] = [(PERSON, 0.9)]
                    elif p.text == "hi-vis vest":
                        d[p.key] = [(VEST_IN, 0.7)]
                    elif p.text == "vest":
                        d[p.key] = [([500, 400, 560, 470], 0.95)]  # confident but on background -> must not win
                    else:
                        d[p.key] = []
                out.append(d)
            return out

    task, report = calibrate_prompts(PhraseDetector(), ["a.jpg", "b.jpg"], vest_task)
    texts = {p.key: p.text for p in task.labeling.detector_prompts}
    assert texts == {"person": "person", "vest": "hi-vis vest"}
    assert report["vest"]["scores"]["vest"] == 0.0
