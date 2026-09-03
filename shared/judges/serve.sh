#!/usr/bin/env bash
# Serve one judge on the local card (design 4.5). One at a time: a policy and a
# judge do not fit together on 32 GB.
#
#   ./serve.sh edit_verifier          # picks the judge assigned to the role
#   ./serve.sh edit_verifier 8011
set -euo pipefail

ROLE="${1:?usage: serve.sh <role> [port]}"
PORT="${2:-8010}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${VLMG_PYTHON:-$HOME/vlmg-env/bin/python}"

read -r MODEL REVISION MAXLEN UTIL < <(
  "$PY" -c "
import sys; sys.path.insert(0, '$REPO')
from shared.judges.lineage import serving_args
a = serving_args('$ROLE')
print(a['hf_path'], a['revision'], a['max_model_len'], a['gpu_memory_utilization'])
"
)

echo "serving role=$ROLE model=$MODEL revision=$REVISION port=$PORT"
echo "  max_model_len=$MAXLEN gpu_memory_utilization=$UTIL (from measured idle VRAM)"
exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --revision "$REVISION" \
  --dtype bfloat16 \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --port "$PORT" \
  --disable-log-requests
