import json

import cv2
import numpy as np

from backend.data_engine import FrameCollection, process_media
from backend.data_engine.ingest import dhash, hamming

from backend.data_engine.tests.helpers import make_video


def test_video_extraction_dedup_and_blur(tmp_path):
    # 10 fps, 8 s; motion freezes at frame 40 -> later samples are duplicates; frame 20 is blurred.
    v = make_video(tmp_path / "cam1.mp4", n_frames=80, fps=10, static_from=40, blur_frames={20})
    fc = process_media([v], tmp_path / "out", fps=2.0, max_frames=None)

    reasons = [r.reason for r in fc.rejected]
    assert "near_duplicate" in reasons
    assert "blurry" in reasons
    assert 5 <= len(fc.frames) <= 12
    assert len(fc.sessions) == 1
    sid = fc.frames[0].session_id
    assert all(f.session_id == sid for f in fc.frames)
    for f in fc.frames:
        assert cv2.imread(f.path) is not None
        assert f.timestamp_s is not None and f.source_file.endswith("cam1.mp4")
    # frames.json round-trips
    again = FrameCollection.model_validate_json((tmp_path / "out" / "frames.json").read_text())
    assert len(again.frames) == len(fc.frames)


def test_sessions_and_explicit_mapping(tmp_path):
    a = make_video(tmp_path / "a.mp4", seed=1)
    b = make_video(tmp_path / "b.mp4", seed=2)
    fc = process_media([a, b], tmp_path / "o1", fps=1.0, max_frames=None)
    assert len(fc.sessions) == 2

    fc2 = process_media([a, b], tmp_path / "o2", fps=1.0, max_frames=None,
                        session_ids={str(a): "day1", str(b): "day1"})
    assert set(fc2.sessions) == {"day1"}


def test_unreadable_and_images(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    imgdir = tmp_path / "shots"
    imgdir.mkdir()
    rng = np.random.default_rng(3)
    for i in range(3):
        cv2.imwrite(str(imgdir / f"{i}.jpg"), (rng.random((240, 320, 3)) * 255).astype(np.uint8))
    (imgdir / "corrupt.jpg").write_bytes(b"\xff\xd8garbage")
    fc = process_media([bad, imgdir], tmp_path / "out", max_frames=None)
    assert len(fc.frames) == 3
    assert {r.reason for r in fc.rejected} == {"unreadable"}
    assert len({f.session_id for f in fc.frames}) == 1  # loose images grouped by folder


def test_max_frames_keeps_every_session(tmp_path):
    a = make_video(tmp_path / "a.mp4", n_frames=100, fps=10, seed=1)
    b = make_video(tmp_path / "b.mp4", n_frames=20, fps=10, seed=2)
    fc = process_media([a, b], tmp_path / "out", fps=5.0, max_frames=10)
    assert len(fc.frames) <= 11
    assert len({f.session_id for f in fc.frames}) == 2
    assert any(r.reason == "not_selected" for r in fc.rejected)


def test_dhash_detects_near_duplicates():
    rng = np.random.default_rng(0)
    img = (rng.random((100, 100, 3)) * 255).astype(np.uint8)
    noisy = np.clip(img.astype(int) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    other = (rng.random((100, 100, 3)) * 255).astype(np.uint8)
    assert hamming(dhash(img), dhash(noisy)) <= 5
    assert hamming(dhash(img), dhash(other)) > 15
