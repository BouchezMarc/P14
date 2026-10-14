#!/bin/sh
set -eu

python3 -m uvicorn api:app \
    --host 0.0.0.0 \
    --port 8080 &

API_PID=$!

python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B-Base \
    --host 127.0.0.1 \
    --port 8000 \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules dpo_v8=/models/dpo_v8 &

VLLM_PID=$!

cleanup() {
    kill "$API_PID" "$VLLM_PID" 2>/dev/null || true
    wait "$API_PID" "$VLLM_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

while kill -0 "$API_PID" 2>/dev/null && kill -0 "$VLLM_PID" 2>/dev/null; do
    sleep 1
done

exit 1
