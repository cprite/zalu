"""Deterministic tests for the Evaluation Engine (owner: Damir). Run: python -m pytest tests/test_evaluation.py -q

Every number asserted here is computed by hand from the fixtures, including deliberately wrong predictions.
"""
import io

import pytest

from backend import schemas
from backend.evaluation_engine import core, leakage
from backend.evaluation_engine import metrics as M
from backend.evaluation_engine.matching import Box, iou, match_image


def G(img, cls, box):
    return Box.of(img, cls, box)


def P(img, cls, box, score):
    return Box.of(img, cls, box, score)


# ---------------- IoU and matching ----------------
def test_iou_known_values():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0  # touching edges


def test_box_normalises_corners():
    assert Box.of("a", "x", [10, 10, 0, 0]).box == (0, 0, 10, 10)
    with pytest.raises(ValueError):
        Box.of("a", "x", [1, 2, 3])


def test_iou_threshold_is_explicit():
    gt = [G("a", "person", (0, 0, 10, 10))]
    pr = [P("a", "person", (2, 0, 12, 10), 0.9)]  # IoU = 80/120 = 0.667
    assert len(match_image(gt, pr, 0.5, 0.25).tp) == 1
    o = match_image(gt, pr, 0.75, 0.25)
    assert (len(o.tp), len(o.fp), len(o.fn)) == (0, 1, 1)
    assert o.fp[0][1] == "poor_localization" and o.fn[0][1] == "poor_localization"
    with pytest.raises(ValueError):
        match_image(gt, pr, 0.0, 0.25)


def test_greedy_by_score_one_match_per_object():
    gt = [G("a", "person", (0, 0, 10, 10))]
    pr = [P("a", "person", (1, 0, 11, 10), 0.6), P("a", "person", (0, 0, 10, 10), 0.9)]
    o = match_image(gt, pr, 0.5, 0.25)
    assert len(o.tp) == 1 and o.tp[0][0].score == 0.9
    assert o.fp[0][1] == "duplicate" and o.fp[0][0].score == 0.6


def test_failure_reasons_for_deliberately_wrong_predictions():
    gt = [
        G("a", "person", (0, 0, 10, 10)),     # found correctly
        G("a", "person", (50, 50, 60, 60)),   # seen only at low confidence
        G("a", "vest", (100, 0, 110, 10)),    # called "person" -> wrong class
        G("a", "vest", (200, 200, 210, 210)), # nothing near it -> missed
    ]
    pr = [
        P("a", "person", (0, 0, 10, 10), 0.95),       # TP
        P("a", "person", (50, 50, 60, 60), 0.10),     # below threshold
        P("a", "person", (100, 0, 110, 10), 0.80),    # wrong class
        P("a", "vest", (300, 300, 320, 320), 0.70),   # hallucination on background
    ]
    o = match_image(gt, pr, 0.5, 0.25)
    assert len(o.tp) == 1
    assert sorted(r for _, r, _ in o.fp) == ["background", "wrong_class"]
    assert {g.box: r for g, r, _ in o.fn} == {
        (50, 50, 60, 60): "low_confidence",
        (100, 0, 110, 10): "wrong_class",
        (200, 200, 210, 210): "missed",
    }


# ---------------- metrics ----------------
def test_wilson_interval_reference_values():
    lo, hi = M.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert M.wilson(0, 0) is None
    assert M.wilson(10, 10)[1] == pytest.approx(1.0)


def test_prf_handles_zero_division():
    m = M.prf(0, 0, 0)
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    m = M.prf(0, 3, 0)
    assert m["precision"] == 0.0 and m["recall"] is None


FIXTURE_IMAGES = ["f1.jpg", "f2.jpg", "f3.jpg", "empty.jpg"]
FIXTURE_GT = [
    G("f1.jpg", "person", (0, 0, 10, 10)), G("f1.jpg", "person", (20, 0, 30, 10)),
    G("f2.jpg", "person", (0, 0, 10, 10)),
    G("f3.jpg", "person", (0, 0, 10, 10)),
]
FIXTURE_PRED = [
    P("f1.jpg", "person", (0, 0, 10, 10), 0.9), P("f1.jpg", "person", (20, 0, 30, 10), 0.8),  # 2 TP
    P("f2.jpg", "person", (0, 0, 10, 10), 0.7),                                            # 1 TP
    # f3: object missed entirely                                                           # 1 FN
    P("empty.jpg", "person", (5, 5, 15, 15), 0.6),                                         # 1 FP on empty frame
    P("f2.jpg", "dog", (0, 0, 5, 5), 0.99),                                                # class not evaluated
]


def test_evaluate_model_counts_and_metrics():
    e = core.evaluate_model(FIXTURE_GT, FIXTURE_PRED, FIXTURE_IMAGES, ["person"], 0.5, 0.25,
                            latencies_ms=[10, 20, 30])
    o = e["overall"]
    assert (o["tp"], o["fp"], o["fn"]) == (3, 1, 1)
    assert o["precision"] == 0.75 and o["recall"] == 0.75 and o["f1"] == 0.75
    assert e["precision"] == 0.75 and e["fp"] == 1 and e["fn"] == 1  # flat copies
    assert e["test"] == {"n_images": 4, "n_images_without_objects": 1, "n_sessions": None,
                         "gt_per_class": {"person": 4}, "classes": ["person"]}
    il = e["per_class"]["person"]["image_level"]
    assert (il["tp"], il["fp"], il["fn"], il["tn"]) == (2, 1, 1, 0)
    assert e["failure_counts"] == {"false_positive": {"background": 1}, "false_negative": {"missed": 1}}
    assert e["failures"][0]["kind"] == "false_negative"  # misses first in the gallery
    assert e["performance"]["fps"] == 50.0 and e["fps"] == 50.0
    assert e["readiness"]["verdict"] == "not_ready"   # recall 0.75 < 0.80
    assert any("Only 4 test images" in w for w in e["warnings"])


def test_confidence_threshold_changes_result():
    e = core.evaluate_model(FIXTURE_GT, FIXTURE_PRED, FIXTURE_IMAGES, ["person"], 0.5, 0.75)
    assert (e["overall"]["tp"], e["overall"]["fp"], e["overall"]["fn"]) == (2, 0, 2)
    assert e["failure_counts"]["false_negative"] == {"low_confidence": 1, "missed": 1}


def test_unknown_image_in_ground_truth_rejected():
    with pytest.raises(ValueError):
        core.evaluate_model([G("nope.jpg", "person", (0, 0, 1, 1))], [], ["a.jpg"], ["person"])


def _big_perfect(n_img=40, per_img=1):
    imgs = [f"i{k}.jpg" for k in range(n_img)]
    gt = [G(i, "person", (0, 0, 10, 10)) for i in imgs for _ in range(per_img)]
    pr = [P(i, "person", (0, 0, 10, 10), 0.9) for i in imgs]
    return imgs, gt, pr


def test_readiness_ready_vs_insufficient_data():
    imgs, gt, pr = _big_perfect(40)
    e = core.evaluate_model(gt, pr, imgs, ["person"], latencies_ms=[20.0] * 5, n_sessions=3)
    assert e["readiness"]["verdict"] == "ready"
    assert "never proves production safety" in e["readiness"]["summary"]
    e = core.evaluate_model(gt[:5], pr[:5], imgs[:5], ["person"], latencies_ms=[20.0])
    assert e["readiness"]["verdict"] == "insufficient_data"
    e = core.evaluate_model(gt, pr, imgs, ["person"], latencies_ms=[500.0])  # 2 FPS
    assert e["readiness"]["verdict"] == "not_ready"


def test_compare_and_synthetic_ab():
    imgs, gt, pr = _big_perfect(40)
    good = core.evaluate_model(gt, pr, imgs, ["person"])
    half = core.evaluate_model(gt, pr[:20], imgs, ["person"])         # recall 0.5
    worse = core.evaluate_model(gt, pr[:30], imgs, ["person"])        # recall 0.75
    per = {"zs": half, "real": good, "syn": worse}
    meta = {"zs": {"kind": "zero_shot"}, "real": {"kind": "trained", "variant": "real_only"},
            "syn": {"kind": "trained", "variant": "real_plus_synthetic"}}
    c = core.compare(per, meta)
    assert c["baseline_id"] == "zs"
    assert c["deltas"]["real"]["recall"] == 0.5 and c["deltas"]["real"]["recall_clear"] is True
    assert c["synthetic_ab"]["verdict"] == "did_not_help"
    assert any("did NOT improve" in n for n in c["notes"])
    assert per["real"]["delta_vs_baseline"]["baseline_id"] == "zs"


def test_select_threshold_on_validation():
    imgs = ["v1.jpg", "v2.jpg"]
    gt = [G("v1.jpg", "person", (0, 0, 10, 10))]
    pr = [P("v1.jpg", "person", (0, 0, 10, 10), 0.6), P("v2.jpg", "person", (0, 0, 10, 10), 0.3)]
    s = core.select_threshold(gt, pr, imgs, ["person"])
    assert s["f1"] == 1.0 and 0.3 < s["threshold"] <= 0.6


# ---------------- fixtures in Kirill's DatasetManifest format ----------------
def manifest(ds_id, variant, root, frames, classes=("person",)):
    """frames: [(frame_id, split, session_id, synthetic)]; images live at images/<split>/<frame_id>.jpg."""
    from backend.data_engine.schema import DatasetManifest, ManifestImage

    return DatasetManifest(
        dataset_id=ds_id, task_id="task_test", variant=variant, root=str(root), data_yaml=f"{root}/data.yaml",
        class_names=list(classes), splits={}, split_strategy="by_session",
        images=[ManifestImage(frame_id=f, split=sp, image=f"images/{sp}/{f}.jpg", label=f"labels/{sp}/{f}.txt",
                              session_id=ses, source_file=f"{ses}.mp4", synthetic=syn, review_status="auto")
                for f, sp, ses, syn in frames])


def frames_of(ds):
    from backend.evaluation_engine.dataset_io import load_test_set

    return load_test_set(ds, annotations={"annotations": []})


TRAIN = lambda root="/d/train", sessions=("sess_a",): manifest(
    "ds_train", "real_only", root, [(f"t{i}", "train", s, False) for i, s in enumerate(sessions)])


# ---------------- leakage ----------------
def test_leakage_clean():
    test = manifest("ds_test", "test", "/d/test", [("f1", "test", "sess_test", False)])
    ts = frames_of(test)
    err, warn = leakage.check_test_set(test, [TRAIN()], ts.frames, ts.split)
    assert err == [] and warn == []


def test_leakage_detects_problems():
    test = manifest("ds_test", "real_plus_synthetic", "/d/test",
                    [("f1", "test", "sess_shared", False), ("synth_2", "test", "sess_shared", False), ("f3", "test", "sess_x", True)])
    ts = frames_of(test)
    err, _ = leakage.check_test_set(test, [TRAIN(sessions=("sess_a", "sess_shared"))], ts.frames, ts.split)
    text = " ".join(err)
    assert "real footage only" in text and "2 synthetic images" in text and "sess_shared" in text
    same = manifest("ds_test", "test", "/d/test", [("f1", "test", "s", False)])
    err, _ = leakage.check_test_set(same, [same], frames_of(same).frames, "test")
    assert any("same as training" in e for e in err)
    nosess = manifest("ds_n", "real_only", "/d/n", [("f1", "val", "", False)])
    ts = frames_of(nosess)
    _, warn = leakage.check_test_set(nosess, [], ts.frames, ts.split)
    assert any("cannot be verified" in w for w in warn) and any("no 'test' split" in w for w in warn)
    assert any("not created as an independent test set" in w for w in warn)


# ---------------- dataset IO ----------------
def test_load_test_set_from_annotations_skips_rejected_and_other_splits():
    from backend.evaluation_engine.dataset_io import load_test_set

    ds = manifest("ds_t", "test", "/d/t", [("f1", "test", "s1", False), ("f2", "test", "s1", False), ("v1", "val", "s0", False)])
    aset = {"annotations": [
        {"frame_id": "f1", "status": "reviewed", "boxes": [{"class_name": "person", "xyxy": [1, 2, 30, 40]}]},
        {"frame_id": "f2", "status": "rejected", "boxes": [{"class_name": "person", "xyxy": [0, 0, 5, 5]}]},
        {"frame_id": "v1", "status": "auto", "boxes": [{"class_name": "person", "xyxy": [0, 0, 5, 5]}]}]}
    ts = load_test_set(ds, annotations=aset)
    assert ts.split == "test" and [f.frame_id for f in ts.frames] == ["f1"]
    assert [(b.image, b.box) for b in ts.gt] == [("images/test/f1.jpg", (1.0, 2.0, 30.0, 40.0))]


def test_load_test_set_from_yolo_labels(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from backend.evaluation_engine.dataset_io import load_test_set

    root = tmp_path / "ds"
    for sub in ("images/test", "labels/test"):
        (root / sub).mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(root / "images/test/a.jpg")
    (root / "labels/test/a.txt").write_text("0 0.5 0.5 0.5 0.5" + chr(10))
    Image.new("RGB", (100, 50)).save(root / "images/test/b.jpg")
    ds = manifest("ds_y", "test", root, [("a", "test", "s1", False), ("b", "test", "s1", False)])
    ts = load_test_set(ds, reader=lambda p: open(p, "rb").read(), annotations={})
    assert ts.source == "yolo_labels" and [(b.cls, b.box) for b in ts.gt] == [("person", (25.0, 12.5, 75.0, 37.5))]


# ---------------- full hook with a fake model (no GPU, no network) ----------------
def test_full_hook_with_fake_model(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    monkeypatch.setenv("ZALU_DATA", str(tmp_path / "data"))
    import importlib
    from backend.infrastructure import storage
    importlib.reload(storage)
    from backend.evaluation_engine import engine

    buf = io.BytesIO(); Image.new("RGB", (100, 50)).save(buf, "JPEG"); jpg = buf.getvalue()
    test = manifest("ds_test", "test", "/gx10/test", [("a", "test", "sess_test", False), ("b", "test", "sess_test", False)])
    aset = {"annotations": [{"frame_id": "a", "status": "auto", "boxes": [{"class_name": "person", "xyxy": [25, 12.5, 75, 37.5]}]},
                            {"frame_id": "b", "status": "auto", "boxes": []}]}
    train = TRAIN()
    storage.save("datasets", train)
    art = schemas.ModelArtifact(base_model="yolo11n.pt", classes=["person"], dataset_id=train.dataset_id, variant="real_only")

    def fake_predict(a, image, conf):
        assert image == jpg and conf == engine.COLLECT_CONF
        return {"model_id": a.id, "width": 100, "height": 50, "latency_ms": 12.0,
                "detections": [{"cls": "person", "score": 0.9, "box": [25, 12.5, 75, 37.5]}]}

    req = schemas.EvaluationRequest(model_ids=[art.id], test_dataset_id=test.dataset_id)
    rep = engine.evaluate(req, [art], test, predict_fn=fake_predict, reader=lambda p: jpg, annotations=aset)
    e = rep.per_model[art.id]
    assert (e["overall"]["tp"], e["overall"]["fp"], e["overall"]["fn"]) == (1, 1, 0)  # FP on the empty frame b
    assert e["test"]["n_sessions"] == 1 and e["sample_data"] is False and e["test"]["dataset_id"] == "ds_test"
    f = e["failures"][0]
    assert f["image"] == "images/test/b.jpg" and f["image_path"].replace("\\", "/").endswith("images/test/b.jpg")
    assert f["thumbnail"].startswith("data:image/jpeg;base64,")
    assert not any(n.startswith("LEAKAGE") for n in rep.notes)
