#!/usr/bin/env bash
# Install / restart the app on nikopi5 as a systemd user service (127.0.0.1:9100, published by Tailscale Funnel :10000).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/zalu-app/.venv"
DATA="$HOME/zalu-app/data"            # on disk, NOT /tmp (tmpfs = RAM on nikopi5)
CRED="$HOME/.openclaw/credentials/hackathons/imperial-ai-hack-2026"
mkdir -p "$DATA"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q fastapi "uvicorn[standard]" "pydantic>=2.7" pyyaml requests python-multipart \
  opencv-python-headless numpy "fal-client>=1.0"
cat > "$HOME/.config/systemd/user/zalu-api.service" <<UNIT
[Unit]
Description=ZALU hackathon app API (nikopi5); GPU work goes to the GX10 worker via zalu-tunnel
After=network-online.target zalu-tunnel.service

[Service]
WorkingDirectory=$REPO
Environment=PYTHONPATH=$REPO ZALU_DATA=$DATA ZALU_GPU_URL=http://127.0.0.1:9200 ZALU_CLAUDE_BIN=$HOME/.local/bin/claude PYTHONUNBUFFERED=1
EnvironmentFile=$CRED/zalu_api.env
ExecStart=$VENV/bin/uvicorn backend.main:app --host 127.0.0.1 --port 9100
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now zalu-api
systemctl --user restart zalu-api
echo "zalu-api restarted"
