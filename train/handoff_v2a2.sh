#!/bin/bash
# Start the v2a rerun once the first v2a queue (now only the paired baseline) has finished.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/v2a.log 2>/dev/null; do sleep 30; done
sleep 10
setsid nohup bash train/run_queue.sh train/queues/v2a2.txt > ~/vlmg-results/v2a2.log 2>&1 < /dev/null &
echo "v2a2 started at $(date)"
