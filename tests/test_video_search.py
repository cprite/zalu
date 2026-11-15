"""Video search API (backend/video_search) through the real FastAPI app, with the GX10 pipeline replaced by a fake.

What is real here: the HTTP routes + auth, streamed upload + ffprobe, the on-disk job ledger and background worker,
the IOU tracker, OpenCV frame decoding + cropping, the embedding index and cosine ranking.
What is FAKE: detection boxes (computed from the synthetic video's known geometry) and the "CLIP" vectors (mean crop
colour). So these tests prove plumbing, not semantic quality. The video is a synthetic test pattern generated here.

Run: python -m pytest tests/test_video_search.py -q
"""
import importlib
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("fastapi")

H = {"Authorization": "Bearer test-token"}
W, HH, FPS, N = 320, 240, 10, 40  # 4 s video


def box_at(i):
    """Red square moving right for the whole clip; a blue one present from frame 20 on."""
    out = [("red", [10 + 4 * i, 60, 60 + 4 * i, 160])]
    if i >= 20:
        out.append(("blue", [200, 120, 260, 220]))
    return out


def make_video(p):
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, HH))
    assert wr.isOpened()
    for i in range(N):
        f = np.full((HH, W, 3), 128, np.uint8)
        for col, (x1, y1, x2, y2) in box_at(i):
            f[y1:y2, x1:x2] = (0, 0, 255) if col == "red" else (255, 0, 0)  # BGR
        wr.write(f)
    wr.release()


class FakePipeline:
    def __init__(self):
        self.detect_calls = 0
        self.text_calls = 0

    def upload(self, batch_id, local):
        return f"/data/uploads/video_search/{batch_id}/{local.name}"

    def detect(self, model, path, fps, conf):
        assert model.kind == "zero_shot" and model.classes == ["person"]
        self.detect_calls += 1
        step = round(FPS / fps)
        return {"frames": [{"frame_index": i, "t_s": round(i / FPS, 3),
                            "detections": [{"cls": "person", "score": 0.9, "box": [float(v) for v in b]}
                                           for _, b in box_at(i)]} for i in range(0, N, step)]}

    def embed_images(self, jpegs):
        out = []
        for b in jpegs:
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR).reshape(-1, 3).mean(0)
            out.append([img[2], img[1], img[0]])  # RGB
        return np.array(out, np.float32)

    def embed_text(self, texts):
        self.text_calls += 1
        return np.array([[1, 0, 0] if "red" in t else [0, 0, 1] for t in texts], np.float32)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ZALU_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("ZALU_TOKEN", "test-token")
    from backend.infrastructure import storage
    importlib.reload(storage)
    import backend.main as main
    importlib.reload(main)
    from backend.video_search import service
    fake = FakePipeline()
    monkeypatch.setattr(service, "PIPELINE", fake)
    from fastapi.testclient import TestClient
    video = tmp_path / "cam1.mp4"
    make_video(video)
    return TestClient(main.app), fake, video


def _wait(c, bid, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/video-search/batches/{bid}", headers=H).json()
        if j["status"] in ("completed", "partial", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError(f"batch did not finish: {j}")


def test_auth_required(env):
    c, _, _ = env
    assert c.post("/api/video-search/batches", json={"recordings": [{"camera_id": "a", "filename": "a.mp4"}]}).status_code == 401


def test_batch_limit_10_ok_11_rejected(env):
    c, _, _ = env
    recs = [{"camera_id": f"cam{i}", "filename": f"c{i}.mp4"} for i in range(11)]
    ok = c.post("/api/video-search/batches", json={"recordings": recs[:10]}, headers=H)
    assert ok.status_code == 200 and len(ok.json()["recordings"]) == 10
    assert c.post("/api/video-search/batches", json={"recordings": recs}, headers=H).status_code == 422


def test_end_to_end_with_fake_gx10(env):
    c, fake, video = env
    b = c.post("/api/video-search/batches", headers=H, json={"name": "t", "recordings": [
        {"camera_id": "camera_01", "filename": "cam1.mp4", "recording_start_utc": "2026-11-15T09:00:00Z"},
        {"camera_id": "camera_02", "filename": "cam2.mp4"}]}).json()
    bid, r1, r2 = b["batch_id"], b["recordings"][0]["recording_id"], b["recordings"][1]["recording_id"]
    for rid in (r1, r2):
        with video.open("rb") as fh:
            up = c.post(f"/api/video-search/batches/{bid}/recordings/{rid}/upload", headers=H,
                        files={"file": ("cam.mp4", fh, "video/mp4")})
        assert up.status_code == 200, up.text
        assert up.json()["size_bytes"] == video.stat().st_size
    assert up.json()["probe"]["width"] == W and up.json()["probe"]["duration_s"] == pytest.approx(4.0, abs=0.2)

    p = c.post(f"/api/video-search/batches/{bid}/process", headers=H)
    assert p.status_code == 200 and p.json()["status"] == "queued" and p.json()["total_recordings"] == 2
    j = _wait(c, bid)
    assert j["status"] == "completed", j
    assert fake.detect_calls == 2
    rec = j["recordings"][0]
    assert rec["state"] == "completed" and rec["tracklets"] == 2 and "gx10_path" not in rec
    assert set(rec["timings_s"]) >= {"detection", "tracking", "embedding", "total"}

    trks = c.get(f"/api/video-search/batches/{bid}/tracklets", headers=H).json()
    assert len(trks) == 4
    red = next(t for t in trks if t["recording_id"] == r1 and t["start_ms"] == 0)
    assert red["end_ms"] == 3500 and red["detection_count"] == 8  # 2 fps sampling of a 10 fps, 40-frame clip: frames 0,5..35

    s = c.post(f"/api/video-search/batches/{bid}/search", headers=H, json={"query": "person in red"})
    assert s.status_code == 200, s.text
    s = s.json()
    top = s["results"][0]
    assert s["total_matches"] == 4 and top["score_type"] == "clip_cosine_similarity"
    assert top["start_ms"] == 0 and top["query_score"] > s["results"][-1]["query_score"]  # red track ranks first
    assert top["cross_camera_identity_status"] == "not_assessed"
    first = next(r for r in s["results"] if r["recording_id"] == r1 and r["start_ms"] == 0)
    assert first["start_utc"] == "2026-11-15T09:00:00Z" and first["end_utc"] == "2026-11-15T09:00:03.500000Z"
    assert next(r for r in s["results"] if r["recording_id"] == r2)["start_utc"] is None

    th = c.get(top["thumbnail_url"], headers=H)
    assert th.status_code == 200 and th.headers["content-type"] == "image/jpeg"
    crop = cv2.imdecode(np.frombuffer(th.content, np.uint8), cv2.IMREAD_COLOR)
    assert crop[crop.shape[0] // 2, crop.shape[1] // 2][2] > 200  # centre of the crop is red

    pb = c.get(top["playback_url"], headers={**H, "Range": "bytes=0-99"})
    assert pb.status_code == 206 and len(pb.content) == 100
    assert c.get(top["playback_url"]).status_code == 401

    # camera filter + stored search
    s2 = c.post(f"/api/video-search/batches/{bid}/search", headers=H,
                json={"query": "blue", "camera_ids": ["camera_02"]}).json()
    assert {r["camera_id"] for r in s2["results"]} == {"camera_02"}
    assert c.get(f"/api/video-search/searches/{s2['search_id']}", headers=H).json()["results"] == s2["results"]


def test_repeated_search_reuses_index(env):
    c, fake, video = env
    b = c.post("/api/video-search/batches", headers=H,
               json={"recordings": [{"camera_id": "camera_01", "filename": "cam1.mp4"}]}).json()
    bid, rid = b["batch_id"], b["recordings"][0]["recording_id"]
    with video.open("rb") as fh:
        c.post(f"/api/video-search/batches/{bid}/recordings/{rid}/upload", headers=H, files={"file": ("c.mp4", fh)})
    c.post(f"/api/video-search/batches/{bid}/process", headers=H)
    _wait(c, bid)
    assert fake.detect_calls == 1
    a = c.post(f"/api/video-search/batches/{bid}/search", headers=H, json={"query": "red"}).json()
    bb = c.post(f"/api/video-search/batches/{bid}/search", headers=H, json={"query": "red"}).json()
    assert fake.detect_calls == 1 and fake.text_calls == 2
    assert [r["tracklet_id"] for r in a["results"]] == [r["tracklet_id"] for r in bb["results"]]
    assert c.post(f"/api/video-search/batches/{bid}/process", headers=H).status_code == 409  # nothing to redo


def test_failed_recording_reported_and_retry(env, monkeypatch):
    c, fake, video = env
    from backend.infrastructure import gpu_client
    real = fake.detect

    def boom(*a, **kw):
        raise gpu_client.GPUWorkerError(503, "GX10 worker unreachable (test)")
    monkeypatch.setattr(fake, "detect", boom)
    b = c.post("/api/video-search/batches", headers=H,
               json={"recordings": [{"camera_id": "camera_01", "filename": "cam1.mp4"}]}).json()
    bid, rid = b["batch_id"], b["recordings"][0]["recording_id"]
    with video.open("rb") as fh:
        c.post(f"/api/video-search/batches/{bid}/recordings/{rid}/upload", headers=H, files={"file": ("c.mp4", fh)})
    c.post(f"/api/video-search/batches/{bid}/process", headers=H)
    j = _wait(c, bid)
    assert j["status"] == "failed" and "unreachable" in j["recordings"][0]["error"]
    monkeypatch.setattr(fake, "detect", real)
    assert c.post(f"/api/video-search/batches/{bid}/recordings/{rid}/retry", headers=H).status_code == 200
    assert _wait(c, bid)["status"] == "completed"


def test_tracker_splits_on_gap():
    from backend.video_search.service import build_tracks
    fr = lambda i, boxes: {"frame_index": i, "t_s": i / 2, "detections": [{"cls": "person", "score": .8, "box": b} for b in boxes]}
    frames = [fr(0, [[0, 0, 10, 10]]), fr(1, [[1, 0, 11, 10], [100, 100, 120, 130]]), fr(2, []), fr(3, []), fr(4, []),
              fr(5, [[1, 0, 11, 10]])]
    tracks = build_tracks(frames)
    assert [len(t) for t in tracks] == [2, 1, 1]  # same place after 3 missed frames -> a new tracklet
