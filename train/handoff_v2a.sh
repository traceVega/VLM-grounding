#!/bin/bash
# Start the v2 SFT probes once the 8B rationale labeling has finished.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/rationales.log 2>/dev/null; do sleep 30; done
sleep 10
setsid nohup bash train/run_queue.sh train/queues/v2a.txt > ~/vlmg-results/v2a.log 2>&1 < /dev/null &
echo "v2a started at $(date)"
