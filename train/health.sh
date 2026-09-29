#!/bin/bash
# One health check of the experiment queue: prints one STATUS line, and ALERT lines when something is wrong
# (no runner, no job process, a failed attempt in the last 10 min, or the newest log not written for 15 min).
# Usage: bash train/health.sh <queue-tag e.g. v17>
Q=$1
LOG=~/vlmg-results/$Q.log
now=$(date +%s)
runner=$(pgrep -f "run_queue.sh train/queues/${Q}.txt" | head -1)
job=$(pgrep -af "train\.(grpo_lora|sft_lora|eval_suite) " | grep -v pgrep | head -1 | cut -c1-110)
age=$(( now - $(stat -c %Y "$LOG" 2>/dev/null || echo 0) ))
step=$(grep -E "step [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "step [0-9]+/[0-9]+" )
tag=$(echo "$job" | grep -oE -- "--tag [A-Za-z0-9_]+" | cut -d" " -f2)
evalp=""
if [ -n "$tag" ] && [ -d ~/vlmg-data/train/eval/$tag ]; then
  for f in ~/vlmg-data/train/eval/$tag/*.jsonl; do [ -f "$f" ] && evalp="$evalp $(basename $f .jsonl)=$(wc -l < $f)"; done
fi
name=$(echo "$job" | grep -oE -- "--name [A-Za-z0-9_]+" | cut -d" " -f2)
[ -n "$name" ] && [ -f ~/vlmg-data/train/$name/train_log.jsonl ] && step="$(tail -1 ~/vlmg-data/train/$name/train_log.jsonl | grep -oE '"step": [0-9]+, "total": [0-9]+' | tr -d '"')"
fails=$(grep -c "failed (attempt" "$LOG" 2>/dev/null)
newest=$(stat -c %Y "$LOG" ~/vlmg-data/train/eval/${tag:-none}/*.jsonl ~/vlmg-data/train/${name:-none}/train_log.jsonl 2>/dev/null | sort -n | tail -1)
age=$(( now - ${newest:-0} ))
echo "STATUS $(date +%H:%M) queue=$Q runner=${runner:-none} log_age=${age}s step=[${step:-none}] eval=[${evalp:-none}] fails=$fails gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')"
echo "JOB ${job:-none}"
[ -z "$runner" ] && grep -q "QUEUE DONE" "$LOG" 2>/dev/null && echo "DONE queue $Q finished"
[ -z "$runner" ] && ! grep -q "QUEUE DONE" "$LOG" 2>/dev/null && echo "ALERT no runner for $Q and the queue is not done"
[ -n "$runner" ] && [ -z "$job" ] && echo "ALERT runner alive but no training/eval process"
[ "$age" -gt 1800 ] && [ -n "$runner" ] && echo "ALERT no log or record written for $((age/60)) min"
if [ "$fails" -gt 0 ]; then
  last=$(grep -n "failed (attempt" "$LOG" | tail -1 | cut -d: -f1)
  tot=$(wc -l < "$LOG")
  [ $((tot - last)) -lt 40 ] && echo "ALERT a stage failed recently: $(grep -B1 'failed (attempt' "$LOG" | grep -E 'error|Error|Traceback' | tail -1 | cut -c1-160)"
fi
exit 0
