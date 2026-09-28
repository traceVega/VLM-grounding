#!/bin/bash
# wait for the scr_sft_coa3 screening (left running after the v11 runner was stopped) to finish, then run v13
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until [ -f ~/vlmg-data/train/eval/scr_sft_coa3/summary.json ]; do sleep 30; done
while pgrep -f "eval_suite --tag scr_sft_coa[3]" > /dev/null; do sleep 15; done
sleep 5
setsid nohup bash train/run_queue.sh train/queues/v13.txt > ~/vlmg-results/v13.log 2>&1 < /dev/null &
echo "v13 started at $(date)"
