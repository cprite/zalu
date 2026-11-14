"""End-to-end through the backend's two apps: public API (backend/main.py, nikopi5) -> GPU worker
(backend/gpu_worker.py, GX10) -> Data Engine. The HTTP hop between them is routed in-process.

Claude and YOLO-World are replaced by fakes; routes, storage, split check, ingest, rules and export are real.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.data_engine.tests.helpers import VEST_SPEC, FakeDetector, make_video

main = pytest.importorskip("backend.main", reason="backend API not present on this branch")
gpu_worker = pytest.importorskip("backend.gpu_worker", reason="GPU worker not present on this branch")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.data_engine import service
    from backend.infrastructure import claude_cli, gpu_client, storage
    from backend.task_engine import TaskSpecDraft

    monkeypatch.setenv("ZALU_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(storage, "DATA_ROOT", (tmp_path / "data").resolve())
    monkeypatch.setattr(main, "TOKEN", "")

    worker = TestClient(gpu_worker.app)
    strip = lambda url: url.replace(gpu_client.GPU_URL, "")  # noqa: E731
    monkeypatch.setattr(gpu_client, "requests", SimpleNamespace(
        get=lambda url, timeout=None, **kw: worker.get(strip(url), **kw),
        post=lambda url, json=None, timeout=None, **kw: worker.post(strip(url), json=json, **kw),
        ConnectionError=ConnectionError))

    draft = json.loads(VEST_SPEC.read_text())
    for k in ("spec_version", "task_id", "prompt", "compiler"):
        draft.pop(k)

    def fake_cli_backend(*a, **k):
        fn = lambda system, messages: TaskSpecDraft.model_validate(draft)  # noqa: E731
        fn.name = "fake-claude-cli"
        return fn

    monkeypatch.setattr(claude_cli, "available", lambda: True)
    monkeypatch.setattr(claude_cli, "claude_cli_backend", fake_cli_backend)
    person, vest = [60.0, 40.0, 140.0, 220.0], [80.0, 90.0, 120.0, 140.0]
    service.set_detector(FakeDetector(lambda i, p: {"person": [(person, 0.9)], "vest": [(vest, 0.8)] if i % 2 else []}))
    yield TestClient(main.app)
    service.set_detector(None)


def _upload(client, task_id, paths, kind="train"):
    files = [("files", (p.name, p.read_bytes(), "video/mp4")) for p in paths]
    return client.post("/api/datasets", data={"task_id": task_id, "kind": kind}, files=files)


def test_full_flow_through_api(client, tmp_path):
    health = client.get("/api/health").json()
    assert health["gpu_worker"]["ok"] and "MOCK" not in health["engines"]["task_engine"]

    spec = client.post("/api/tasks", json={"prompt": "Detect workers without safety vests."}).json()
    assert spec["status"] == "ok" and spec["compiler"] == "fake-claude-cli"

    vids = [make_video(tmp_path / f"cam{i}.mp4", n_frames=60, fps=10, seed=i) for i in (1, 2, 3)]
    r = _upload(client, spec["task_id"], vids)
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["split_strategy"] == "by_session(auto)"
    assert set(ds["timings"]) >= {"ingest", "auto_label", "export", "total"}
    assert ds["timings"]["total"] >= ds["timings"]["ingest"]
    assert set(ds["splits"]["train"]["sessions"]).isdisjoint(ds["splits"]["val"]["sessions"])
    assert client.get(f"/api/datasets/{ds['dataset_id']}").json()["dataset_id"] == ds["dataset_id"]

    aset = client.get(f"/api/datasets/{ds['dataset_id']}/annotations").json()
    first, second = aset["annotations"][0], aset["annotations"][1]
    fixes = [{"frame_id": first["frame_id"], "status": "reviewed",
              "boxes": [{"class_name": "person_with_vest", "xyxy": [10, 10, 90, 200]}]},
             {"frame_id": second["frame_id"], "status": "rejected"}]
    r = client.post(f"/api/datasets/{ds['dataset_id']}/labels", json=fixes)
    assert r.status_code == 200, r.text
    total = lambda m: sum(s["images"] for s in m["splits"].values())  # noqa: E731
    stored = client.get(f"/api/datasets/{ds['dataset_id']}").json()
    assert total(stored) == total(ds) - 1  # backend re-saved the updated manifest
    assert yaml.safe_load(Path(stored["data_yaml"]).read_text())["path"] == stored["root"]
    label = Path(stored["root"]) / next(i["label"] for i in stored["images"] if i["frame_id"] == first["frame_id"])
    assert label.read_text().startswith("0 ")  # manual person_with_vest box


@pytest.mark.xfail(strict=True, reason="BLOCKER (Kolyan): training_engine/splits.check_splits rejects test-only "
                   "datasets (train is a sentinel, val == test by design), so kind=test uploads get HTTP 422")
def test_test_set_upload_through_api(client, tmp_path):
    spec = client.post("/api/tasks", json={"prompt": "Detect workers without safety vests."}).json()
    r = _upload(client, spec["task_id"], [make_video(tmp_path / "test_cam.mp4", seed=9)], kind="test")
    assert r.status_code == 200, r.text
    assert r.json()["variant"] == "test"


def test_unsupported_task_cannot_build_dataset(tmp_path):
    from backend.data_engine import service
    from backend.task_engine import TaskSpec

    spec = TaskSpec.model_validate_json(VEST_SPEC.read_text()).model_copy(update={"status": "unsupported"})
    with pytest.raises(ValueError, match="unsupported"):
        service.process_upload(spec, [str(make_video(tmp_path / "a.mp4"))])
