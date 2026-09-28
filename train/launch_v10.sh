#!/bin/bash
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
setsid nohup bash train/run_queue.sh train/queues/v10.txt > ~/vlmg-results/v10.log 2>&1 < /dev/null &
echo "v10 started at $(date)"
