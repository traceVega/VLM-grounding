#!/bin/bash
# wait for the sft_coa3 training process (started by the v10 runner, which was stopped) to exit, then run v11
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
while pgrep -f "sft_lora --name sft_coa[3]" > /dev/null; do sleep 30; done
sleep 5
setsid nohup bash train/run_queue.sh train/queues/v11.txt > ~/vlmg-results/v11.log 2>&1 < /dev/null &
echo "v11 started at $(date)"
