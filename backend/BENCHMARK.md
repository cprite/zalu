# Backend benchmarks (GX10: ASUS GX10 / NVIDIA GB10, aarch64, torch 2.14+cu130). Measured 2026-11-14.

All numbers are single runs. Datasets are public samples, not our task.

| What | Setup | Result |
|---|---|---|
| Training, tiny | coco8 (4 train / 4 val), yolo11n, 10 epochs, imgsz 640 | 5.0-6.9 s incl. pretrained baseline val, ONNX export + check |
| Training, small | coco128 split 96 train / 32 val, yolo11n, 30 epochs, batch 16 | **28.2 s** |
| Baseline vs fine-tuned (coco128 split, 80 COCO classes) | pretrained yolo11n vs fine-tuned 30 ep | mAP50 **0.663 → 0.573**, recall 0.615 → 0.492 (fine-tuning a COCO model on 96 COCO images makes it WORSE; fine-tuning only pays off for new classes) |
| ONNX check | .pt vs ONNX on one val image | same detection count (2 vs 2); ONNX on CPU 27.5 ms |
| API inference, warm | POST /api/inference, 810x1080 | ~15-17 ms (trained yolo11n and zero-shot YOLOE-L); first call ~1 s (model load) |
| Camera client, video file, 300 frames 1080p | `tools/webcam_client.py --onnx ... --headless` on GX10 CPU | 30.4 FPS, p50 29.3 ms, p95 32.7 ms |
| Camera client, video file, 300 frames 1080p | `--api http://127.0.0.1:9100` (GPU behind the API, no network hop) | 51.2 FPS, p50 13.5 ms, p95 14.2 ms |
| Long video | restaurant clip looped to 1 h, yolo11m | every frame + ByteTrack 15 min (4x realtime); 1 frame/s 2.7 min |

Gotchas found:
- Docker's default /dev/shm (64 MB) deadlocks the PyTorch dataloader (GPU 0 %, CPU 0 %): run with `--shm-size 8g` (in run_gx10.sh).
- Ultralytics' sample `coco128.yaml` uses the SAME 128 images for train and val; our split check rejects it, so the benchmark uses a 96/32 split.
- Laptop numbers will differ: the demo laptop's CPU, and the network hop to the Funnel URL, are not measured here.
