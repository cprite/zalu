# GX10 GPU worker image (GB10, aarch64, CUDA 13). Models + the Python needed to run them; no app, no LLM, no keys.
# Reuses the local image that already has a working torch 2.14+cu130.
FROM spark-comfyui:latest
USER root
RUN /opt/comfyui-env/bin/python -m pip install --no-cache-dir \
      "ultralytics>=8.4" onnxruntime fastapi "uvicorn[standard]" python-multipart onnx onnxslim lap requests pyyaml pytest fal-client httpx \
      "git+https://github.com/ultralytics/CLIP.git"
WORKDIR /data
COPY backend /app/backend
COPY tests /app/tests
COPY tools /app/tools
ENV PYTHONPATH=/app ZALU_DATA=/data PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/data/.ultralytics
EXPOSE 9200
ENTRYPOINT ["/opt/comfyui-env/bin/python", "-m", "uvicorn", "backend.gpu_worker:app", "--host", "0.0.0.0", "--port", "9200"]
