#!/bin/bash
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/v20.log 2>/dev/null; do sleep 60; done
sleep 5
setsid nohup bash train/run_queue.sh train/queues/v21.txt > ~/vlmg-results/v21.log 2>&1 < /dev/null &
echo "v21 started at $(date)"
