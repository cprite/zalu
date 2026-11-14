# Dashboard Planner (owner: Kirill)

User prompt + TaskSpec (+ model classes, + Kolyan's data-source catalogue) → **DashboardSpec v1** (Codex's contract,
`shared/dashboard_spec/`). Claude chooses widgets, order, sizes and labels. It never writes React/HTML/CSS, and the
output is always validated with Codex's `validate_dashboard_spec` plus our capability checks.

```python
from backend.task_engine.dashboard import plan_dashboard, refine_dashboard

r = plan_dashboard(prompt, task, model_classes=model.classes, sources_catalog=catalog)   # PlannerResult
r.spec              # valid DashboardSpec v1, always present
r.status            # ok | partial | needs_clarification | unsupported
r.ai_generated      # False => deterministic fallback layout (r.generator == "fallback:rules", r.fallback_reason)
r.limitations       # requested statistics that cannot be shown, and why (show these in the UI)

r2 = refine_dashboard(r.spec, "Make the video larger.", task, model_classes=model.classes)  # no retraining
```

## Guarantees
- Only `inference_session` data, the 3 v1 metric operations, and classes from the model/task vocabulary.
  Event tables only if the task has an alert class (`target_event`).
- Requests for unavailable statistics (unique people, hourly/daily totals, occupancy %, zones, history) produce
  `status: "partial"` with a limitation and the closest honest proxy. Windows over 600 s are clamped and reported.
- Claude errors (network, auth, timeout after 2 SDK retries, refusal, output still invalid after one repair round) →
  deterministic fallback built from the TaskSpec. A safety task leads with alerts; a counting task leads with the camera.
  `refine_dashboard` falls back to rule-based edits (add / remove / enlarge / shrink widgets).
- The API key stays server-side (`ANTHROPIC_API_KEY`, standard SDK resolution). Model: `claude-opus-5-5` at low effort
  (`VF_DASHBOARD_MODEL` overrides it). Any `(system, messages) -> DashboardDraft` callable can be passed as `llm=`.

## Measured (live Claude, 2026-11-14)
| Prompt | Time | Status | First widgets |
|---|---|---|---|
| "Detect workers without safety vests and alert me…" | 11.3 s | ok | 3 KPIs (no vest / vest / in view) → alert table → camera → 10-min trend |
| "Detect vehicles in my parking lot and show their counts over time." | 8.0 s | ok | camera → vehicles KPI → trends → by-type distribution (no alert table) |
| "…how many people were wearing a vest during each hour of the day." | 7.6 s | **partial** | live count + 10-min trend; limitation: no hourly history |
| Refine: add chart / larger video / remove event table | ~5 s each | ok | applied |
| Refine: "Add a counter for forklifts." | 5.6 s | partial | refused: class not in model, retraining needed |
| API down (bad key / unreachable) | <1 s | partial | fallback layout, `ai_generated: false` |

Outputs: `examples/*.plan.json` (real responses) and `examples/safety_vest.refined.spec.json`.

## Integration (proposed routes, Kolyan)
- `POST /api/dashboards` `{prompt, task_id, model_id?}` → `PlannerResult`. Pass `model.classes` and the catalogue from
  `GET /api/dashboard/sources`; store `spec` with the task/model.
- `POST /api/dashboards/{id}/refine` `{instruction}` → `PlannerResult`.
- Renderer: render `spec` as is. Show `limitations`, and badge `ai_generated: false` as a fallback layout.

Tests: `python -m pytest backend/task_engine/dashboard -q` (offline); `VF_LIVE_TESTS=1` adds the live personalisation check.
