#!/bin/bash
# Block until a queue log holds at least N distinct stages ("=== hh:mm:ss [k] cmd ===" headers,
# counted by unique cmd so a retried stage is not counted twice).
# Usage: bash train/wait_stage.sh <log> <N> [poll-seconds]
LOG=$1; N=$2; POLL=${3:-60}
while true; do
  c=$(grep '^=== [0-9:]* \[[0-9]\] ' "$LOG" 2>/dev/null | sed 's/^=== [0-9:]* \[[0-9]\] //' | sort -u | wc -l)
  if [ "${c:-0}" -ge "$N" ]; then break; fi
  sleep "$POLL"
done
date
grep -v 'Loading weights' "$LOG" | grep -E '^===|\[eval step|saved|!!!|Traceback' | tail -6
