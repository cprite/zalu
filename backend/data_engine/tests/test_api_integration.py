"""End-to-end through the backend's FastAPI app (backend/main.py): task -> upload -> dataset -> corrections.

Claude and YOLO-World are replaced by fakes; everything else (routes, storage, split check,
ingest, rules, export) is the real code.
"""
import json
from pathlib import Path

import pytest
import yaml

from backend.data_engine.tests.helpers import VEST_SPEC, FakeDetector, make_video

main = pytest.importorskip("backend.main", reason="backend API not present on this branch")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.data_engine import service
    from backend.infrastructure import storage
    from backend.task_engine import TaskSpecDraft, compiler

    monkeypatch.setenv("ZALU_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(storage, "DATA_ROOT", (tmp_path / "data").resolve())
    monkeypatch.setattr(main, "TOKEN", "")

    draft = json.loads(VEST_SPEC.read_text())
    for k in ("spec_version", "task_id", "prompt", "compiler"):
        draft.pop(k)

    def fake_backend(*a, **k):
        fn = lambda system, messages: TaskSpecDraft.model_validate(draft)  # noqa: E731
        fn.name = "fake-claude"
        return fn

    monkeypatch.setattr(compiler, "claude_backend", fake_backend)
    person, vest = [60.0, 40.0, 140.0, 220.0], [80.0, 90.0, 120.0, 140.0]
    service.set_detector(FakeDetector(lambda i, p: {"person": [(person, 0.9)], "vest": [(vest, 0.8)] if i % 2 else []}))
    yield TestClient(main.app)
    service.set_detector(None)


def test_full_flow_through_api(client, tmp_path):
    health = client.get("/api/health").json()
    assert health["hooks"]["task_engine"] and health["hooks"]["data_engine"]

    spec = client.post("/api/tasks", json={"prompt": "Detect workers without safety vests."}).json()
    assert spec["status"] == "ok" and spec["compiler"] == "fake-claude"

    vids = [make_video(tmp_path / f"cam{i}.mp4", n_frames=60, fps=10, seed=i) for i in (1, 2, 3)]
    files = [("files", (v.name, v.read_bytes(), "video/mp4")) for v in vids]
    r = client.post("/api/datasets", data={"task_id": spec["task_id"]}, files=files)
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["split_strategy"] == "by_session(auto)"
    assert set(ds["splits"]["train"]["sessions"]).isdisjoint(ds["splits"]["val"]["sessions"])
    assert Path(ds["data_yaml"]).exists()
    assert client.get("/api/datasets").json()[0]["dataset_id"] == ds["dataset_id"]

    from backend.data_engine import get_annotations
    from backend.schemas import DatasetManifest
    aset = get_annotations(DatasetManifest(**ds))
    first = aset.annotations[0]
    fixes = [{"frame_id": first.frame_id, "status": "reviewed",
              "boxes": [{"class_name": "person_with_vest", "xyxy": [10, 10, 90, 200]}]},
             {"frame_id": aset.annotations[1].frame_id, "status": "rejected"}]
    r = client.post(f"/api/datasets/{ds['dataset_id']}/labels", json=fixes)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["applied"] == 2 and out["manifest"]["dataset_id"] == ds["dataset_id"]
    new = out["manifest"]
    total = lambda m: sum(s["images"] for s in m["splits"].values())  # noqa: E731
    assert total(new) == total(ds) - 1
    assert yaml.safe_load(Path(new["data_yaml"]).read_text())["path"] == new["root"]

    from backend.data_engine import validate_dataset
    assert validate_dataset(new["root"]).ok
    label = Path(new["root"]) / next(i["label"] for i in new["images"] if i["frame_id"] == first.frame_id)
    assert label.read_text().startswith("0 ")  # manual person_with_vest box


def test_unsupported_task_cannot_build_dataset(client, tmp_path, monkeypatch):
    from backend.data_engine import service
    from backend.task_engine import TaskSpec

    spec = TaskSpec.model_validate_json(VEST_SPEC.read_text()).model_copy(update={"status": "unsupported"})
    with pytest.raises(ValueError, match="unsupported"):
        service.process_upload(spec, [str(make_video(tmp_path / "a.mp4"))])
