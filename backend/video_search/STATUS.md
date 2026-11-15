# Video search backend: STATUS (2026-11-15, timeboxed P0)

Branch `feature/video-search-backend-mvp` (from `kolyan/backend-p0` @ 09991dd), worktree
`imperial-ai-hack-2026/zalu-video-search`. Not pushed. Not deployed: the running zalu-api / zalu-gpu are unchanged.

## The one thing to know first
**The real GX10 path has never run.** From the build sandbox the GPU worker (127.0.0.1:9200 tunnel) is blocked
("deny network-outbound 127.0.0.1:9200"), and the Pi has no torch/CLIP. So real person detection, real CLIP
embeddings and real search scores are **implemented but unverified**. Every automated test uses a fake GX10
pipeline. No real-video smoke test and no benchmark were run.

## Real and tested (tests/test_video_search.py, real FastAPI app via TestClient, fake GX10)
- `POST /api/video-search/batches`: 10 recordings OK, 11th -> 422. Bearer auth like every other /api route (401 tested).
- `POST .../batches/{id}/recordings/{rid}/upload`: streamed in 1 MiB chunks to `*.part`, then `os.replace`;
  size checked; ffprobe fills duration/fps/width/height/codec (nulls + `probe_error` if ffprobe is missing/fails).
  ffprobe also checked by hand on a real 1080p h264 file (deliverables/demo/raw.mp4: 202.767 s, 30 fps).
- `POST .../batches/{id}/process`: returns `queued` at once; one daemon worker thread; the batch JSON is the on-disk
  ledger (recordings left `processing` by a restart are requeued on the next worker start).
- Per recording: GX10 upload -> `/video_infer` (zero-shot `yoloe-11l-seg.pt`, classes ["person"], fps from batch,
  default 2) -> greedy IOU tracker (IOU >= 0.3, close after 2 missed samples) -> representative crop (best
  score x sqrt(area), 10 % padding) decoded with OpenCV from the Pi's copy -> CLIP image embedding -> stored per
  recording (`embeddings/{rec}.npy`, L2-normalised, `embedding_version` = `clip-ViT-B/32:crop-pad0.1:v1`).
  The zero-shot artifact is created once per batch and reused (stored in `models/`, like `POST /api/models/zero-shot`).
- `GET .../batches/{id}`: counts per state, per-recording state/stage/error, measured stage timings, sampled
  frames, detections, tracklets, progress = finished recordings / total (no invented percentages).
- `POST .../batches/{id}/search`: CLIP text embedding of `"a photo of {query}"`, cosine vs stored index, camera
  filter, top_k, min_score, result contract of spec section 11 (`score_type: clip_cosine_similarity`,
  `cross_camera_identity_status: not_assessed`, UTC only when `recording_start_utc` was given). Stored, `GET
  /api/video-search/searches/{id}`. Repeated search does not call detection (tested: detect_calls stays 1).
- `GET /api/video-search/tracklets/{id}/thumbnail` (JPEG crop, tested that it is the right object).
- `GET /api/video-search/recordings/{rid}/playback?start_ms=` authenticated, HTTP Range works (206 tested).
- Failed recording -> state `failed` + GX10 error text; `POST .../recordings/{rid}/retry` reprocesses it (tested).
- Also: `GET /api/video-search/batches`, `GET .../batches/{id}/tracklets`.

## Implemented but unverified
- `backend/gpu_worker.py` new `POST /clip/image` and `POST /clip/text` (ultralytics CLIP fork, ViT-B/32). Never
  executed. Assumes `~/.cache/clip/ViT-B-32.pt` is already in the zalu-gpu container (YOLO-World's text encoder uses
  the same model); if it is not cached, `clip.load` needs internet on the GX10 once. zalu-gpu must be rebuilt/
  restarted to get these endpoints (hack-lead/owner decision).
- Real `/video_infer` + Pi-side upload to GX10 `/upload` (streamed with requests_toolbelt if installed).
- Search quality on real CCTV: unknown. Scores are raw, uncalibrated cosine similarities.
- Frame alignment: GX10 counts frames with `grab()`; the Pi seeks with `CAP_PROP_POS_FRAMES`. Usually the same
  frame for mp4/h264, not guaranteed for every codec/VFR file.

## Known limitations (documented, not fixed)
- Timestamps = frame_index / container fps, not PTS (wrong for VFR footage). Uncertainty is one sampling interval
  (500 ms at 2 fps); reported as `sampling.timing_uncertainty_ms`.
- GX10 `/upload` (existing code) reads the whole file into memory on the GX10 side.
- Recordings are processed one at a time; `/video_infer` is one blocking call, so no frame-level progress.
- The video is stored twice (Pi + GX10); no cleanup/retention.
- Playback lookup scans all batches; no per-user/project scoping beyond the single shared bearer token.

## Not implemented
P1 cross-camera association, caching, dashboard adapter, clip export around a match, the full 20-item test matrix,
PTS-accurate timing, GX10 benchmark (1/5/10 videos).

## Tests run (on nikopi5, Pi Python, no torch)
- `python3 -m pytest -q` before the change: `2 failed, 96 passed, 2 skipped`.
- After: `2 failed, 102 passed, 2 skipped`. The same 2 fail before and after, both because torch/ultralytics is not
  installed on the Pi: `data_engine/tests/test_annotate_export.py::test_ultralytics_accepts_dataset`,
  `data_engine/tests/test_api_integration.py::test_full_flow_through_api`.
- `python3 -m pytest tests/test_video_search.py -q`: `6 passed`.
- Existing routes (incl. the Train a Model flow) are unchanged: main.py only gains an `include_router` at the end;
  gpu_worker.py only gains two new routes at the end.

## Try it (after deploy; API on nikopi5 127.0.0.1:9100)
```bash
A="Authorization: Bearer $ZALU_TOKEN"; U=http://127.0.0.1:9100/api/video-search
curl -s -H "$A" -H 'Content-Type: application/json' $U/batches -d '{"name":"demo","recordings":[
  {"camera_id":"camera_01","filename":"cam1.mp4","recording_start_utc":"2026-11-15T09:00:00Z"}]}'
curl -s -H "$A" -F file=@cam1.mp4 $U/batches/$BID/recordings/$RID/upload
curl -s -H "$A" -X POST $U/batches/$BID/process
curl -s -H "$A" $U/batches/$BID                     # poll until status completed/partial/failed
curl -s -H "$A" -H 'Content-Type: application/json' $U/batches/$BID/search -d '{"query":"person in a red jacket","top_k":10}'
curl -s -H "$A" -o t.jpg $U/tracklets/$TID/thumbnail
curl -s -H "$A" -H 'Range: bytes=0-999' -o part.mp4 $U/recordings/$RID/playback?start_ms=0
# GX10 CLIP check (inside zalu-gpu, after restart):
curl -s -X POST 127.0.0.1:9200/clip/text -H 'Content-Type: application/json' -d '{"texts":["a person"]}' | head -c 200
```

## Needed from outside this task
- Network to the GX10 worker (or someone on nikopi5) to run the real smoke test.
- Restart zalu-gpu with the new gpu_worker.py, and confirm CLIP ViT-B/32 weights are cached in the container.
- Push of this branch (no git push from the sandbox).
