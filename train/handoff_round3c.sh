#!/bin/bash
# When round 3 reaches its V3c stage (4th distinct stage), stop it and run the multi-crop SFT instead (round 3c).
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
bash train/wait_stage.sh ~/vlmg-results/round3.log 4 30
echo "handoff at $(date): stopping round 3 (V3c)"
pkill -f "run_queue.sh train/queues/round3.txt"
sleep 2
pkill -f "grpo_lora --name grpo_v3c_pair_derived"
sleep 10
rm -rf ~/vlmg-data/train/grpo_v3c_pair_derived
nvidia-smi --query-gpu=memory.used --format=csv,noheader
setsid nohup bash train/run_queue.sh train/queues/round3c.txt > ~/vlmg-results/round3c.log 2>&1 < /dev/null &
echo "round 3c started at $(date)"
