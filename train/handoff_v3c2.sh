#!/bin/bash
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/v3e.log 2>/dev/null; do sleep 60; done
sleep 5
setsid nohup bash train/run_queue.sh train/queues/v3c.txt > ~/vlmg-results/v3c.log 2>&1 < /dev/null &
echo "v3c started at $(date)"
