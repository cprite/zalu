"""Integration: the shared FastAPI app (backend/main.py, unchanged) discovers the Evaluation Engine hook and serves
POST /api/evaluations and GET /api/reports/{id} in the nikopi5 + GX10 split. The GX10 worker is replaced by a fake
gpu_client (annotations, files, /infer), so no GPU or network is needed.

Run: python -m pytest tests/test_evaluation_api.py -q
"""
import importlib
import io

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

    def fake_post(path, json=None, timeout=60, **kw):
        if path == "/datasets/annotations":
            return {"annotations": [
                {"frame_id": "a", "status": "auto", "boxes": [{"class_name": "person", "xyxy": [25, 12.5, 75, 37.5]}]},
                {"frame_id": "b", "status": "auto", "boxes": []}]}
        if path == "/infer":
            calls["infer"] += 1
            return {"id": "inf_x", "model_id": "m", "width": 100, "height": 50, "latency_ms": 9.0,
                    "detections": [{"cls": "person", "score": 0.9, "box": [25, 12.5, 75, 37.5]}], "events": []}
        raise AssertionError("unexpected worker call " + path)

    class Resp:
        content = jpg

    def fake_stream(path):
        calls["files"] += 1
        return Resp()

    monkeypatch.setattr(gpu_client, "post", fake_post)
    monkeypatch.setattr(gpu_client, "stream_file", fake_stream)
    from fastapi.testclient import TestClient
    return TestClient(main.app), storage, calls


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
    c, storage, calls = client
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
    assert c.get(f"/api/reports/{rep['id']}", headers=H).json()["id"] == rep["id"]


def test_leaky_test_set_is_flagged(client):
    c, storage, _ = client
    test, m = _setup(storage, test_sessions=("sess_shared",), train_sessions=("sess_a", "sess_shared"))
    rep = c.post("/api/evaluations", headers=H, json={"model_ids": [m.id], "test_dataset_id": "ds_test"}).json()
    assert rep["per_model"][m.id]["readiness"]["verdict"] == "invalid_test_set"
    assert any(n.startswith("LEAKAGE") and "sess_shared" in n for n in rep["notes"])
