#!/usr/bin/env bash
# Build and (re)start the backend on the GX10. Run on the GX10 from the repo root.
# Port 9100 is bound to localhost only; reach it via an ssh tunnel or the nikopi5 proxy.
set -euo pipefail
: "${ZALU_TOKEN:?set ZALU_TOKEN}"
mkdir -p "$HOME/zalu-data"
docker build -f backend/Dockerfile -t zalu-backend:dev .
docker rm -f zalu-backend >/dev/null 2>&1 || true
# Optional API keys for Kirill's engines: ~/.zalu_env (chmod 600) with ANTHROPIC_API_KEY=... and FAL_KEY=...
ENVFILE=()
[ -f "$HOME/.zalu_env" ] && ENVFILE=(--env-file "$HOME/.zalu_env")
docker run -d --name zalu-backend --gpus all --memory 24g --shm-size 8g --restart unless-stopped \
  -p 127.0.0.1:9100:9100 -e ZALU_TOKEN="$ZALU_TOKEN" "${ENVFILE[@]}" -v "$HOME/zalu-data:/data" zalu-backend:dev
echo "started; health: curl -s localhost:9100/api/health"
