#!/usr/bin/env bash
# Serve the base model (+ the Adaption LoRA when present) on the GX10 with vLLM, OpenAI-compatible, 127.0.0.1:9300.
# Run ON THE GX10. The LoRA dir is copied to ~/zalu-data/adaption/lora by deploy step; HF_TOKEN only in this process env.
set -euo pipefail
BASE="${ZALU_ADAPTION_BASE:-Qwen/Qwen3.5-4B}"
LORA="$HOME/zalu-data/adaption/lora"
ARGS=(--model "$BASE" --served-model-name "$BASE" --host 0.0.0.0 --port 9300 --max-model-len 8192
      --gpu-memory-utilization 0.15 --dtype bfloat16)
MOUNT=()
if [ -f "$LORA/adapter_config.json" ]; then
  ARGS+=(--enable-lora --max-lora-rank 64 --lora-modules "zalu-taskspec=/lora"); MOUNT=(-v "$LORA:/lora:ro")
fi
docker rm -f zalu-llm >/dev/null 2>&1 || true
docker run -d --name zalu-llm --gpus all --memory 32g --shm-size 8g --restart unless-stopped \
  -p 127.0.0.1:9300:9300 -e HF_TOKEN="${HF_TOKEN:-}" -v "$HOME/zalu-data/hf-cache:/root/.cache/huggingface" "${MOUNT[@]}" \
  nvcr.io/nvidia/vllm:26.03.post1-py3 vllm serve "${ARGS[@]}"
echo "zalu-llm starting: curl -s localhost:9300/v1/models"
