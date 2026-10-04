#!/bin/bash
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
# v28 runner stopped while RL11 trains (the training job keeps running); start v29b when RL11 exits with an adapter
while pgrep -f "grpo_lora --name grpo_coa11" >/dev/null; do sleep 60; done
sleep 5
if [ ! -f ~/vlmg-data/train/grpo_coa11/adapter/adapter_config.json ]; then echo "ALERT RL11 exited without an adapter at $(date); v29b not started"; exit 1; fi
setsid nohup bash train/run_queue.sh train/queues/v29b.txt > ~/vlmg-results/v29b.log 2>&1 < /dev/null &
echo "v29b started at $(date)"
