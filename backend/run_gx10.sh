#!/usr/bin/env bash
# Build and (re)start the GPU worker on the GX10. Run on the GX10 from the repo root.
# Bound to 127.0.0.1:9200 only; the app on nikopi5 reaches it through the zalu-tunnel ssh tunnel.
set -euo pipefail
mkdir -p "$HOME/zalu-data"
docker build -f backend/Dockerfile -t zalu-gpu:dev .
docker rm -f zalu-backend zalu-gpu >/dev/null 2>&1 || true
docker run -d --name zalu-gpu --gpus all --memory 24g --shm-size 8g --restart unless-stopped \
  -p 127.0.0.1:9200:9200 -v "$HOME/zalu-data:/data" zalu-gpu:dev
echo "started; health: curl -s localhost:9200/health"
