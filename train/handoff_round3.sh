#!/bin/bash
# When round 2b reaches its V3c stage (4th distinct stage, i.e. V3d and its eval are done), stop it and start round 3.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
bash train/wait_stage.sh ~/vlmg-results/round2b.log 4 30
echo "handoff at $(date): stopping round 2b"
pkill -f "run_queue.sh train/queues/round2b.txt"
sleep 2
pkill -f "grpo_lora --name grpo_v3c_pair_derived"
sleep 10
rm -rf ~/vlmg-data/train/grpo_v3c_pair_derived
nvidia-smi --query-gpu=memory.used --format=csv,noheader
setsid nohup bash train/run_queue.sh train/queues/round3.txt > ~/vlmg-results/round3.log 2>&1 < /dev/null &
echo "round 3 started at $(date)"
