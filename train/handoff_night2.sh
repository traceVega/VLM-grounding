#!/bin/bash
# Hand over from queue 1 to queue 2 the moment queue 1 starts its V5 stage (9th stage header):
# stop the queue-1 runner and the V5 SFT it just launched, then start queue 2 detached.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
LOG=~/vlmg-results/trace_night.log
bash train/wait_stage.sh "$LOG" 9 30
echo "handoff at $(date): stopping queue 1"
pkill -f "run_queue.sh train/queues/trace_night.txt"
sleep 2
pkill -f "sft_lora --name sft_trace_obs"
sleep 10
nvidia-smi --query-gpu=memory.used --format=csv,noheader
setsid nohup bash train/run_queue.sh train/queues/trace_night2.txt > ~/vlmg-results/trace_night2.log 2>&1 < /dev/null &
echo "queue 2 started at $(date)"
