#!/bin/bash
# When round 1d reaches its V3c stage (3rd distinct stage), stop it (V3c is re-run inside round 2 after
# the hint-crop smoke tests) and start round 2 detached.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
bash train/wait_stage.sh ~/vlmg-results/round1d.log 3 20
echo "handoff at $(date): stopping round 1d"
pkill -f "run_queue.sh train/queues/round1d.txt"
sleep 2
pkill -f "grpo_lora --name grpo_v3c_pair_derived"
sleep 10
nvidia-smi --query-gpu=memory.used --format=csv,noheader
setsid nohup bash train/run_queue.sh train/queues/round2.txt > ~/vlmg-results/round2.log 2>&1 < /dev/null &
echo "round 2 started at $(date)"
