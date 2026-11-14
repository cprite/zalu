"""Client the nikopi5 app uses to reach the GX10 GPU worker (through the ssh tunnel on 127.0.0.1:9200)."""
from __future__ import annotations

import os
from typing import Any, Optional

import requests

GPU_URL = os.environ.get("ZALU_GPU_URL", "http://127.0.0.1:9200")


class GPUWorkerError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"GPU worker {status}: {detail}")
        self.status, self.detail = status, detail


def _check(r: requests.Response) -> Any:
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise GPUWorkerError(r.status_code, str(detail)[:1000])
    return r.json()


def get(path: str, timeout: float = 30, **kw) -> Any:
    try:
        return _check(requests.get(GPU_URL + path, timeout=timeout, **kw))
    except requests.ConnectionError as e:
        raise GPUWorkerError(503, f"GX10 worker unreachable at {GPU_URL} ({e.__class__.__name__})")


def post(path: str, json: Optional[dict] = None, timeout: float = 60, **kw) -> Any:
    try:
        return _check(requests.post(GPU_URL + path, json=json, timeout=timeout, **kw))
    except requests.ConnectionError as e:
        raise GPUWorkerError(503, f"GX10 worker unreachable at {GPU_URL} ({e.__class__.__name__})")


def stream_file(path: str):
    r = requests.get(GPU_URL + "/files", params={"path": path}, stream=True, timeout=120)
    if r.status_code >= 400:
        raise GPUWorkerError(r.status_code, r.text[:300])
    return r
