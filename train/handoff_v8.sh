#!/bin/bash
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/v7.log 2>/dev/null; do sleep 60; done
sleep 5
setsid nohup bash train/run_queue.sh train/queues/v8.txt > ~/vlmg-results/v8.log 2>&1 < /dev/null &
echo "v8 started at $(date)"
