"""Integration: the shared FastAPI app (backend/main.py) discovers the Evaluation Engine hook and serves
POST /api/evaluations and GET /api/reports/{id} in the nikopi5 + GX10 split. The GX10 worker is replaced by a fake
gpu_client (annotations, files, /infer), so no GPU or network is needed.

Run: python -m pytest tests/test_evaluation_api.py -q
"""
import importlib
import io
from json import loads

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")

from backend import schemas  # noqa: E402
from tests.test_evaluation import TRAIN, manifest  # noqa: E402

H = {"Authorization": "Bearer test-token"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setenv("ZALU_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("ZALU_TOKEN", "test-token")
    from backend.infrastructure import gpu_client, storage
    importlib.reload(storage)
    import backend.main as main
    importlib.reload(main)

    buf = io.BytesIO(); Image.new("RGB", (100, 50)).save(buf, "JPEG"); jpg = buf.getvalue()
    calls = {"infer": 0, "files": 0}
    worker = {"annotations": True, "width": 100, "review_status": "reviewed", "missing_label": False}

    def fake_post(path, json=None, timeout=60, **kw):
        if path == "/datasets/annotations":
            if not worker["annotations"]:
                raise gpu_client.GPUWorkerError(404, "no annotations")
            return {"annotations": [
                {"frame_id": "a", "status": worker["review_status"], "boxes": [{"class_name": "person", "xyxy": [25, 12.5, 75, 37.5]}]},
                {"frame_id": "b", "status": worker["review_status"], "boxes": []}]}
        if path == "/infer":
            calls["infer"] += 1
            return {"id": "inf_x", "model_id": loads(kw["data"]["model"])["id"], "width": worker["width"], "height": 50, "latency_ms": 9.0,
                    "detections": [{"cls": "person", "score": 0.9, "box": [25, 12.5, 75, 37.5]}], "events": []}
        raise AssertionError("unexpected worker call " + path)

    class Resp:
        def __init__(self, content):
            self.content = content

        def close(self):
            pass

    def fake_stream(path):
        calls["files"] += 1
        path = path.replace(chr(92), "/")
        if path.endswith("/a.txt"):
            return Resp(b"0 0.5 0.5 0.5 0.5")
        if path.endswith("/b.txt"):
            if worker["missing_label"]:
                raise gpu_client.GPUWorkerError(404, "label file missing")
            return Resp(b"")
        return Resp(jpg)

    monkeypatch.setattr(gpu_client, "post", fake_post)
    monkeypatch.setattr(gpu_client, "get", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(gpu_client, "stream_file", fake_stream)
    from fastapi.testclient import TestClient
    return TestClient(main.app), storage, calls, worker


def _setup(storage, test_sessions=("sess_test",), train_sessions=("sess_a",)):
    test = manifest("ds_test", "test", "/gx10/test", [("a", "test", test_sessions[0], False), ("b", "test", test_sessions[0], False)])
    train = TRAIN(sessions=train_sessions)
    storage.save("datasets", test)
    storage.save("datasets", train)
    m = schemas.ModelArtifact(base_model="yolo11n.pt", classes=["person"], dataset_id=train.dataset_id,
                              variant="real_only", weights="/gx10/best.pt")
    storage.save("models", m)
    return test, m


def test_evaluation_route_end_to_end(client):
    c, storage, calls, _ = client
    test, trained = _setup(storage)
    zs = c.post("/api/models/zero-shot", headers=H, json={"classes": ["person"]}).json()
    assert c.post("/api/evaluations", json={"model_ids": [zs["id"]], "test_dataset_id": "ds_test"}).status_code == 401
    assert c.get("/api/health").json()["engines"]["evaluation_engine"] == "Damir"

    r = c.post("/api/evaluations", headers=H, json={"model_ids": [zs["id"], trained.id], "test_dataset_id": "ds_test"})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["mock"] is False
    e = rep["per_model"][trained.id]
    assert (e["overall"]["tp"], e["overall"]["fp"], e["overall"]["fn"]) == (1, 1, 0)
    assert e["delta_vs_baseline"]["baseline_id"] == zs["id"]
    assert calls["infer"] == 4 and calls["files"] == 2  # 2 frames x 2 models; each frame fetched once
    assert schemas.EvaluationReport.model_validate(rep).model_dump(mode="json") == rep
    assert c.get(f"/api/reports/{rep['id']}", headers=H).json() == rep
    assert e["test"]["split"] == "test"
    assert (e["failures"][0]["width"], e["failures"][0]["height"]) == (100, 50)


def test_leaky_test_set_is_flagged(client):
    c, storage, _, _ = client
    test, m = _setup(storage, test_sessions=("sess_shared",), train_sessions=("sess_a", "sess_shared"))
    rep = c.post("/api/evaluations", headers=H, json={"model_ids": [m.id], "test_dataset_id": "ds_test"}).json()
    assert rep["per_model"][m.id]["readiness"]["verdict"] == "invalid_test_set"
    assert any(n.startswith("LEAKAGE") and "sess_shared" in n for n in rep["notes"])

def test_api_uses_yolo_when_annotation_set_is_unavailable(client):
    c, storage, calls, worker = client
    worker["annotations"] = False
    test, model = _setup(storage)
    for image in test.images:
        image.review_status = "reviewed"
    storage.save("datasets", test)
    res = c.post("/api/evaluations", headers=H, json={
        "model_ids": [model.id], "test_dataset_id": test.dataset_id,
    })
    assert res.status_code == 200, res.text
    payload = res.json()
    report = payload["per_model"][model.id]
    assert (report["overall"]["tp"], report["overall"]["fp"], report["overall"]["fn"]) == (1, 1, 0)
    assert report["readiness"]["verdict"] != "invalid_test_set"
    assert any("yolo_labels" in note for note in payload["notes"])
    assert calls["infer"] == 2


def test_api_rejects_validation_only_manifest_before_inference(client):
    c, storage, calls, _ = client
    test, model = _setup(storage)
    for image in test.images:
        image.split = "val"
    storage.save("datasets", test)
    res = c.post("/api/evaluations", headers=H, json={
        "model_ids": [model.id], "test_dataset_id": test.dataset_id,
    })
    assert res.status_code == 422
    assert "no 'test'" in res.json()["detail"]
    assert calls["infer"] == calls["files"] == 0


def test_api_rejects_wrong_worker_dimensions(client):
    c, storage, _, worker = client
    test, model = _setup(storage)
    worker["width"] = 50
    res = c.post("/api/evaluations", headers=H, json={
        "model_ids": [model.id], "test_dataset_id": test.dataset_id,
    })
    assert res.status_code == 422
    assert "dimensions" in res.json()["detail"]


def test_api_marks_unreviewed_ground_truth_ineligible(client):
    c, storage, _, worker = client
    test, model = _setup(storage)
    worker["review_status"] = "auto"
    res = c.post("/api/evaluations", headers=H, json={
        "model_ids": [model.id], "test_dataset_id": test.dataset_id,
    })
    assert res.status_code == 200, res.text
    entry = res.json()["per_model"][model.id]
    assert entry["readiness"]["verdict"] == "invalid_test_set"
    assert "reviewed ground truth" in entry["readiness"]["summary"]


def test_api_missing_yolo_file_is_not_treated_as_background(client):
    c, storage, calls, worker = client
    test, model = _setup(storage)
    worker.update(annotations=False, missing_label=True)
    res = c.post("/api/evaluations", headers=H, json={
        "model_ids": [model.id], "test_dataset_id": test.dataset_id,
    })
    assert res.status_code == 404
    assert calls["infer"] == 0
