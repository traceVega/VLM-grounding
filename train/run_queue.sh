#!/bin/bash
# Run a queue of experiment lines sequentially, each retried up to three times (transient CUDA
# errors on this host; every stage resumes per item).  A queue file holds one shell command
# per line; lines starting with # are skipped.  Usage: bash train/run_queue.sh <queue-file>
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
set -o pipefail
QUEUE=$1
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue;; esac
  for attempt in 1 2 3; do
    echo "=== $(date +%H:%M:%S) [$attempt] $line ==="
    bash -c "$line" 2>&1 | grep --line-buffered -v "Loading weights" && break
    echo "!!! failed (attempt $attempt)"
  done
done < "$QUEUE"
echo "=== $(date +%H:%M:%S) QUEUE DONE ==="
