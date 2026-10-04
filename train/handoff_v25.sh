#!/bin/bash
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
# v24 runner was stopped while seed-1 RL9 trained (orphaned job keeps running): start v25 when that job exits with an adapter
while pgrep -f "grpo_lora --name grpo_coa9_seed1" >/dev/null; do sleep 60; done
sleep 5
if [ ! -f ~/vlmg-data/train/grpo_coa9_seed1/adapter/adapter_config.json ]; then echo "ALERT seed-1 RL9 exited without an adapter at $(date); v25 not started"; exit 1; fi
setsid nohup bash train/run_queue.sh train/queues/v25.txt > ~/vlmg-results/v25.log 2>&1 < /dev/null &
echo "v25 started at $(date)"
