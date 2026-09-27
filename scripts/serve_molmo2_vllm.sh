#!/usr/bin/env bash
# Serve Molmo2-8B (pinned revision) with vLLM 0.28 from ~/ptr1-env for the datagen
# listener.  The card's transformers remote code does not run under transformers
# 5.16; vLLM has a native Molmo2 implementation.  One model on the card at a time.
set -euo pipefail
PY="${VLMG_VLLM_PYTHON:-$HOME/ptr1-env/bin/python}"
exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model allenai/Molmo2-8B --revision e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b \
  --dtype bfloat16 --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization "${VLLM_UTIL:-0.62}" \
  --trust-remote-code --port "${PORT:-8010}"
