# GX10 (GB10, aarch64, CUDA 13): reuse the local image that already has a working torch 2.14+cu130.
FROM spark-comfyui:latest
USER root
RUN /opt/comfyui-env/bin/python -m pip install --no-cache-dir \
      "ultralytics>=8.4" onnxruntime fastapi "uvicorn[standard]" python-multipart onnx onnxslim lap requests \
      "git+https://github.com/ultralytics/CLIP.git"
WORKDIR /data
COPY backend /app/backend
COPY tests /app/tests
ENV PYTHONPATH=/app ZALU_DATA=/data PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/data/.ultralytics
EXPOSE 9100
ENTRYPOINT ["/opt/comfyui-env/bin/python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "9100"]
