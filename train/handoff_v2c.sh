#!/bin/bash
# Start v2c (RL-only arm + Gemini-rationale SFT arm) once v2b has finished.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/v2b.log 2>/dev/null; do sleep 60; done
sleep 10
setsid nohup bash train/run_queue.sh train/queues/v2c.txt > ~/vlmg-results/v2c.log 2>&1 < /dev/null &
echo "v2c started at $(date)"
