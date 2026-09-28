#!/bin/bash
# arm only after gme_style_items.jsonl exists: wait for RL3's step-50 checkpoint, stop the v13 runner (so it does not retry) and
# the RL process, then run v14 (the step-50 screening comes first).
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until [ -d ~/vlmg-data/train/grpo_coa3/adapter_step50 ]; do sleep 60; done
sleep 30
pkill -f "run_queue.sh train/queues/v13[b]"
sleep 2
pkill -f "grpo_lora --name grpo_coa[3] "
while pgrep -f "grpo_lora --name grpo_coa[3] " > /dev/null; do sleep 10; done
sleep 10
setsid nohup bash train/run_queue.sh train/queues/v14.txt > ~/vlmg-results/v14.log 2>&1 < /dev/null &
echo "v14 started at $(date)"
