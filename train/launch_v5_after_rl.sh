#!/bin/bash
# Start the probe as soon as the grpo_v3 process has exited and its adapter is saved (skipping the old-line screening).
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until [ -f ~/vlmg-data/train/grpo_v3/adapter/adapter_config.json ] && ! pgrep -f "grpo_lora --name grpo_v[3] " > /dev/null; do sleep 30; done
sleep 10
setsid nohup bash train/run_queue.sh train/queues/v5.txt > ~/vlmg-results/v5.log 2>&1 < /dev/null &
echo "v5 started at $(date)"
