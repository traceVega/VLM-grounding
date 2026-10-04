#!/bin/bash
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
# start v30 when v29b finishes (the health loop restarts the queue named in CURRENT_QUEUE if its runner dies)
until grep -q "QUEUE DONE" ~/vlmg-results/v29b.log 2>/dev/null; do sleep 60; done
sleep 5
if pgrep -f "run_queue.sh train/queues/v30.txt" >/dev/null; then echo "v30 already running"; exit 0; fi
echo v30 > ~/vlmg-results/CURRENT_QUEUE
setsid nohup bash train/run_queue.sh train/queues/v30.txt >> ~/vlmg-results/v30.log 2>&1 < /dev/null &
echo "v30 started at $(date)"
