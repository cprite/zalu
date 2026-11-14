# Task Engine + Data Engine (owner: Kirill)

Prompt → TaskSpec → frames → auto-labels → (optional synthetic) → YOLO dataset.
Standalone: no FastAPI, no GPU backend, no frontend required.

## Setup

```bash
uv venv -p 3.11 .venv && source .venv/bin/activate
uv pip install -r backend/data_engine/requirements.txt
export ANTHROPIC_API_KEY=...     # Task Engine
export FAL_KEY=...               # only for synthetic data (P1)
python -m pytest backend/task_engine backend/data_engine -q     # offline, ~2 s
VF_LIVE_TESTS=1 python -m pytest backend/task_engine -q         # also calls Claude
```

YOLO-World weights (`yolov8s-worldv2.pt`) download on first use to `~/.cache/visionforge/weights` (override with `VF_WEIGHTS_DIR`).

## CLI

```bash
# 1. Prompt -> TaskSpec (Claude, ~10-15 s)
python -m backend.task_engine "Detect workers without safety vests." -o spec.json
python -m backend.task_engine --schema            # JSON Schema of TaskSpec

# 2. Everything in one go (ingest + calibrate + auto-label + export + validate)
python -m backend.data_engine run --task spec.json --media cam_day1.mp4 cam_day2.mp4 \
    --held-out s_cam_test_ab12cd --out runs/vest --device mps

# Individual steps
python -m backend.data_engine ingest   --media clip.mp4 --session clip.mp4=day1 --out work
python -m backend.data_engine annotate --task spec.json --frames work/frames.json --out work/annotations.json --preview work/preview
python -m backend.data_engine correct  --annotations work/annotations.json --corrections fixes.json
python -m backend.data_engine export   --task spec.json --frames work/frames.json --annotations work/annotations.json --out ds --held-out <test sessions>
python -m backend.data_engine export-test --task spec.json --frames work/frames.json --annotations work/annotations.json --sessions <test sessions> --out test_ds
python -m backend.data_engine synthesize --task spec.json --frames work/frames.json --annotations work/annotations.json --exclude-sessions <val+test> --out work/syn
python -m backend.data_engine validate ds

# Train with stock Ultralytics, no VisionForge code involved
yolo detect train data=ds/data.yaml model=yolov8n.pt
```

## Python API (integration contract)

| Function | Returns | Notes |
|---|---|---|
| `task_engine.compile_task(prompt)` | `TaskSpec` | `status` is `ok` / `needs_clarification` / `unsupported`; raises `TaskCompilationError` on refusal or repeated invalid output |
| `data_engine.process_media(files, out_dir, session_ids=None, fps=1, max_frames=200)` | `FrameCollection` | One video = one session unless mapped; drops unreadable / blurry / near-duplicate frames, then diversity sampling |
| `data_engine.auto_annotate(frames, task, detector=None)` | `AnnotationSet` | YOLO-World + derivation rules + prompt calibration + review flags |
| `data_engine.apply_corrections(aset, [LabelCorrection])` | `AnnotationSet` | Body of `POST /api/datasets/{id}/labels`; `boxes` replaces the frame's boxes |
| `data_engine.export_dataset(aset, task, out_dir, frames, held_out_sessions=[...])` | `DatasetManifest` | `manifest.data_yaml` is what Training needs |
| `data_engine.export_test_set(aset, task, out_dir, sessions)` | `DatasetManifest` | For Evaluation; `train:` points at a sentinel so it cannot be trained on |
| `data_engine.synthetic.generate_synthetic(aset, frames, task, out_dir, exclude_sessions=[...])` | `SyntheticDataset` | fal edits, re-labelled, inconsistent ones rejected |
| `data_engine.validate_dataset(path)` / `check_leakage(train_root, test_root)` | report / problems | Standalone checks |

Sample payloads: `backend/task_engine/examples/*.json`, `backend/data_engine/examples/*.sample.json`.

## Data-integrity guarantees

- Train/val split is **by recording session**. With a single session the fallback is a contiguous time block with one boundary frame dropped, plus a warning.
- `held_out_sessions` never enter the training dataset; the validator and `check_leakage` (session ids plus near-identical images) verify it.
- Synthetic images inherit their parent's session and go to **train only**, and only when the parent is a train session. `--no-synthetic` / `include_synthetic=False` gives the real-only variant for A/B comparison.

## Known limitations (measured, not assumed)

- Auto-label quality depends heavily on detector wording. On the Intel `worker-zone-detection` sample, "vest" found worn vests in 23/26 frames and "high-visibility safety vest" in 3/26. Calibration chooses between Claude's alternatives with an unsupervised heuristic, so check `stats.prompt_calibration`.
- YOLO-World misses small or low-resolution people (confidence about 0.2–0.3). Those frames are flagged `possible unlabelled person` and need review in the UI before training.
- A person cut off at the frame edge with no visible vest is labelled `without_vest` and flagged for review.
- `image_classification` TaskSpecs compile, but auto-labelling supports detection only.
- The fal path is tested only with a fake editor. It has not been run against the live API (no `FAL_KEY` yet).
