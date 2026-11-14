# DashboardSpec v1

Shared configuration for generated ZALU dashboards. The input is untrusted until validated against the [JSON Schema](dashboard-spec.v1.schema.json) and the selected task or model's class names. The renderer can import the [TypeScript interfaces](dashboard-spec.ts).

## Generate and validate

Generate a single JSON object with `version: "1"`. Supply the generator with the schema and a trusted class vocabulary (selected model classes if available, otherwise the task classes). Validate again when binding to a model.

```python
from shared.dashboard_spec.validation import DashboardSpecError, validate_dashboard_spec

try:
    validate_dashboard_spec(generated_spec, class_names=selected_model.classes)
except DashboardSpecError as exc:
    # Pass the structured errors back to the generator or to the API caller.
    validation_errors = exc.errors
```

Errors carry JSON Pointer paths, with the empty string referring to the document root. Validation checks widget config, source and metric references, unique IDs within each collection, the class vocabulary, and widget spans. Use the schema for shape validation and the Python validator for all checks. Within v1, keep existing field meanings and defaults stable so stored v1 specs continue to validate and render. A new required field or a change in meaning requires a new version.

## Sources and layout

V1 binds one `inference_session` to the currently selected model and its camera or sample images. The runtime submits the image to the existing `POST /api/inference` and keeps the submitted frame and corresponding `InferenceResult` together. Assign a receipt timestamp to each successful response for the in-memory session buffer. The spec contains a logical source ID only. Runtime wiring supplies the selected model, frames, connection and credentials.

Use `layout.columns` as the wide-screen grid column count. Render widgets in array order with the configured `span`, wrapping to the next row if the span exceeds remaining space. On narrow screens the renderer can stack widgets in the same order.

## Data requirements

| Widget | Required runtime data | Empty or unavailable behavior |
| --- | --- | --- |
| `live_preview` | Current frame. Box overlays require the response for that same frame, its `width` and `height`, and `detections[].{cls,score,box}` in original-image pixels. | With no frame, show a waiting state. A frame can render while inference is pending. Draw an empty overlay for a valid empty detections array. |
| `kpi` | Latest successful response and an allowlisted metric bound to this source. | Missing response or field yields unavailable. A valid empty detection array yields zero counts. |
| `time_series` | Metric readings from successful responses in the last `window_seconds`, using receipt timestamps within this session. One point per response. | An empty window has no points. Unavailable readings are gaps. |
| `detection_table` | Latest response's `detections` rows or `events` strings, in response order up to `max_rows`. Event availability depends on a runtime `target_event` configured for inference. | Missing data or disabled events yield unavailable. A valid empty array yields an empty table. |
| `class_distribution` | Latest response's `detections[].cls` grouped by class, counting boxes. | A valid empty list yields an empty chart. Missing response or detections field yields unavailable. |

The source supports three metric operations. `detection_count` is the number of boxes in the latest response. `class_count` counts boxes with an exact, trusted class name. `inference_latency_ms` is the model time from `InferenceResult.latency_ms`. `read_metric` is a Python reference implementation for one frame. A series repeats the same operation over sampled frames. Counts are per frame and can include the same object in several samples. Event rows describe the latest frame because there is no event store behind this contract.

The schema accepts no URLs, query text, expressions or custom operations. Occupancy, unique vehicles, incident totals and historical aggregates require future backend capabilities.

## Fixtures and verification

- [Workplace safety](examples/workplace-safety.json) uses `person_without_vest` from the safety task and the configured target-event stream.
- [Parking and vehicle detection](examples/parking-vehicles.json) uses `car` from the selected model and groups vehicle detections by class. Use a vehicle-only model or confirmed vehicle taxonomy for this example.

Both are configuration fixtures and require no real inference.

```sh
python -m pip install pytest -r shared/dashboard_spec/requirements.txt
python -m pytest -q shared/dashboard_spec/tests
npx -p typescript@5.9.3 tsc --strict --noEmit shared/dashboard_spec/dashboard-spec.ts
```

## Export integration

The export ZIP will vendor the reusable React renderer once it is available, plus this same DashboardSpec, an offline mock inference adapter, an opt-in API adapter, styles and install instructions. Keep mock mode as the default and obtain endpoint and credential configuration at runtime. The current checked-in frontend has a task-specific Deploy view. A shared DashboardSpec renderer is required before implementing and testing export parity.
