#!/bin/bash
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/v22.log 2>/dev/null; do sleep 60; done
sleep 5
setsid nohup bash train/run_queue.sh train/queues/v23.txt > ~/vlmg-results/v23.log 2>&1 < /dev/null &
echo "v23 started at $(date)"
