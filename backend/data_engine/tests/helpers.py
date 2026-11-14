import json
from pathlib import Path

import cv2
import numpy as np


VEST_SPEC = Path(__file__).resolve().parents[2] / "task_engine" / "examples" / "vest.json"


def make_video(path: Path, n_frames: int = 60, fps: int = 10, seed: int = 0, static_from: int = None,
               blur_frames=()) -> Path:
    """Textured background with a moving rectangle; optionally freeze motion (duplicates) or blur frames."""
    rng = np.random.default_rng(seed)
    bg = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (3, 3), 0)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(n_frames):
        t = i if static_from is None or i < static_from else static_from
        img = bg.copy()
        x = int(10 + (t * 7) % 250)
        cv2.rectangle(img, (x, 60), (x + 50, 180), (0, 200, 255), -1)
        if i in blur_frames:
            img = cv2.GaussianBlur(img, (31, 31), 0)
        vw.write(img)
    vw.release()
    return path



class FakeDetector:
    """Returns pre-programmed raw detections keyed by image path order."""

    name = "fake"

    def __init__(self, fn):
        self.fn = fn

    def detect(self, paths, prompts):
        return [self.fn(i, p) for i, p in enumerate(paths)]
