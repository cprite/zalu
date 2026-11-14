"""Live camera client for the demo laptop.

Two modes:
  --onnx path/to/best.onnx   run the exported model locally (no network), shows FPS
  --api URL --model MODEL_ID send frames to POST /api/inference on the backend

Usage:
  pip install opencv-python onnxruntime requests numpy
  python tools/webcam_client.py --onnx best.onnx --names person,vest
  ZALU_TOKEN=... python tools/webcam_client.py --api https://HOST --model model_abc123
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np


def onnx_session(path: str):
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=ort.get_available_providers())
    inp = sess.get_inputs()[0]
    return sess, inp.name, inp.shape[2]


def onnx_predict(sess, name, size, frame, conf=0.25):
    h, w = frame.shape[:2]
    s = size / max(h, w)
    nh, nw = int(h * s), int(w * s)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {name: x})[0][0]          # (4+nc, N) for YOLO11 detect
    out = out.T
    scores = out[:, 4:]
    cls = scores.argmax(1)
    sc = scores.max(1)
    keep = sc > conf
    boxes, cls, sc = out[keep, :4], cls[keep], sc[keep]
    xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                     boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1) / s
    idx = cv2.dnn.NMSBoxes(xyxy.tolist(), sc.tolist(), conf, 0.5)
    idx = np.array(idx).flatten() if len(idx) else []
    return [(int(cls[i]), float(sc[i]), xyxy[i]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--onnx")
    ap.add_argument("--names", default="")
    ap.add_argument("--api")
    ap.add_argument("--model")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--video", help="read a video file instead of the camera")
    ap.add_argument("--headless", action="store_true", help="no window; print FPS/latency summary at the end")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until the end / q)")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video if a.video else a.cam)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {'video ' + a.video if a.video else 'camera ' + str(a.cam)}")
    names = a.names.split(",") if a.names else []
    if a.onnx:
        sess, iname, size = onnx_session(a.onnx)
    else:
        import requests

        hdr = {"Authorization": f"Bearer {os.environ['ZALU_TOKEN']}"} if os.environ.get("ZALU_TOKEN") else {}
    t_last, fps = time.time(), 0.0
    lat, n, t_start, counts = [], 0, time.time(), {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_inf = time.perf_counter()
        if a.onnx:
            dets = [(names[c] if c < len(names) else str(c), s, b) for c, s, b in onnx_predict(sess, iname, size, frame, a.conf)]
        else:
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            r = requests.post(f"{a.api}/api/inference", headers=hdr, timeout=10,
                              data={"model_id": a.model, "conf": a.conf}, files={"image": ("f.jpg", jpg.tobytes(), "image/jpeg")})
            r.raise_for_status()
            dets = [(d["cls"], d["score"], d["box"]) for d in r.json()["detections"]]
        lat.append((time.perf_counter() - t_inf) * 1000)
        n += 1
        for c, _, _ in dets:
            counts[c] = counts.get(c, 0) + 1
        if a.headless:
            if a.frames and n >= a.frames:
                break
            continue
        for c, s, b in dets:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, f"{c} {s:.2f}", (x1, max(15, y1 - 5)), 0, 0.6, (0, 220, 0), 2)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
        t_last = now
        cv2.putText(frame, f"{fps:.1f} FPS  {'local ONNX' if a.onnx else 'backend'}", (10, 25), 0, 0.7, (0, 0, 255), 2)
        cv2.imshow("ZALU", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")) or (a.frames and n >= a.frames):
            break
    if lat:
        lat_sorted = sorted(lat)
        el = time.time() - t_start
        print(f"frames={n} fps={n / el:.1f} latency_ms p50={lat_sorted[len(lat)//2]:.1f} "
              f"p95={lat_sorted[int(len(lat)*0.95)-1]:.1f} mode={'onnx-local' if a.onnx else 'backend'} "
              f"detections_per_class={counts}")


if __name__ == "__main__":
    main()
