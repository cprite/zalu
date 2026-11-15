# Dashboard data API (owner: Kolyan) — for Claude's dashboard renderer and Kirill

All endpoints: app on nikopi5 `https://nikopi5.tail4aaac7.ts.net:10000`, `Authorization: Bearer $ZALU_TOKEN`.
GPU work runs on the GX10 worker. **DashboardSpec v1 (Codex) was not in the repo on 2026-11-14 19:00 UTC**; the
data-source names below are the proposal to map DashboardSpec `source` fields onto. Live catalog: `GET /api/dashboard/sources`.

## Data sources
| name | status | endpoint | notes |
|---|---|---|---|
| `model.info` | supported | `GET /api/models/{id}` | existing endpoint |
| `model.status` | supported | `GET /api/dashboard/models/{id}/status` | deployable, run count, last run, latest report id |
| `detections.frame` | supported | `POST /api/inference` | existing endpoint, one image |
| `detections.video` | supported | `POST /api/runs` → `GET /api/runs/{id}` | model applied to a video at `fps` sampled frames/s |
| `counts.by_class` | supported | `GET /api/runs/{id}/summary` | **detections**, frames_with_class, max_in_one_frame — not unique objects |
| `timeseries.class_counts` | supported | `GET /api/runs/{id}/timeseries?bucket_s=1` | max per class in one frame per bucket; video time |
| `events` | supported if the TaskSpec has `event_rules` | `GET /api/runs/{id}/events` | trigger_class present ≥ min_consecutive_frames sampled frames |
| `evaluation.report` | supported | `GET /api/reports/{id}` | existing (Damir's engine) |
| `counts.unique_objects` | **unavailable** | — | needs tracking |
| `occupancy.percent` | **unavailable** | — | needs capacity or regions per camera |
| `history.wall_clock` | needs processing | `GET /api/runs` | runs are retained with created_at; no continuous live logging |

## Create a run (video → detections)
```bash
curl -H "Authorization: Bearer $ZALU_TOKEN" -F model_id=model_... -F task_id=task_... -F fps=2 -F conf=0.25 \
     -F file=@video.mp4 $ZALU/api/runs            # → VideoRun {id, frames:[{frame_index, t_s, detections:[{cls,score,box}]}], ...}
```
`task_id` is optional; it is needed only for `events`.

## Real responses (GX10 smoke test, 2026-11-14 19:03 UTC)
Model `model_899ac71407` (yolo11n, trained on the Intel worker-zone video), task "worker without hi-vis vest".
Files in `backend/examples/dashboard/` are the unedited API responses:
- `run_vest.json`: 25 s video, 50 sampled frames, 4.3 s wall, 18.6 ms GPU per frame; `person_with_vest` in 74% of frames.
- `run_test-no-vest.json`: 33 s independent footage (no hi-vis vests in it), 70 frames, 1.7 s wall. The model still says
  `person_with_vest` in 74% of frames and never `person_without_vest`, so `events` is empty. That is a **model error**
  (it was trained without people lacking vests), shown honestly; the event logic itself is covered by `tests/test_dashboard.py`.
- `sources.json`, `summary_*.json`, `timeseries_*.json` (bucket 2 s), `events_*.json`, `model_status.json`.

Tests: `PYTHONPATH=. python -m pytest tests/test_dashboard.py` (3 deterministic tests, no GPU).

## Dashboard planning routes (planner: Kirill `backend/task_engine/dashboard`, spec: Codex DashboardSpec v1)
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/dashboards` | `{prompt, task_id, model_id?}` | `{id, status, spec, ai_generated, generator, limitations, ...}` (PlannerResult + id) |
| POST | `/api/dashboards/{id}/refine` | `{instruction}` | same shape, updated spec; never retrains |
| GET | `/api/dashboards/{id}` | — | latest result + `history` of refinements |

- The planner gets `model.classes` (when `model_id` is given) and this backend's catalogue (`GET /api/dashboard/sources`).
- LLM = the owner's Claude subscription via the `claude` CLI on nikopi5 (no API key). CLI failures or invalid output
  raise the planner's `PlannerUnavailable`, so the planner returns its deterministic fallback (`ai_generated: false`).
- Live results (2026-11-14 ~19:40 UTC, vest task + model_899ac71407), files in `backend/examples/dashboard/planner_*.json`:
  create 15.6 s, status ok: 2 KPIs → no-vest alert table → live camera → 10-min trend → class split;
  refine "Make the live camera larger and put it first" 14.3 s, ok; refine "Add a counter for forklifts" 14.0 s,
  **partial** with the limitation "the model only detects person_with_vest and person_without_vest … retrain".
