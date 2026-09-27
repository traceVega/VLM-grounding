#!/bin/bash
# The PoC in order: 4B baseline evaluation -> LoRA SFT with the unlock curve -> the same
# evaluation on the adapter.  Every stage checkpoints per item; a stage is retried up to
# three times because this host drops a transient CUDA error now and then.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
set -o pipefail
PY=~/vlmg-env/bin/python
NAME=${1:-sft_v1}
stage() {  # name, command...
  local name=$1; shift
  for attempt in 1 2 3; do
    echo "=== $(date +%H:%M:%S) $name (attempt $attempt) ==="
    "$@" 2>&1 | grep -v "Loading weights" && return 0
    echo "!!! $name failed (attempt $attempt)"
  done
  return 1
}
stage "eval base4b"  $PY -m train.eval_suite --tag base4b --model 4b
stage "sft $NAME"    $PY -m train.sft_lora --name $NAME --model 4b --drop-4b-refused
stage "eval $NAME"   $PY -m train.eval_suite --tag $NAME --model 4b --adapter ~/vlmg-data/train/$NAME/adapter
echo "=== $(date +%H:%M:%S) POC DONE ==="
