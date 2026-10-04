#!/bin/bash
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
# v21 was cut after the SFT7 screening (RL8 dropped): wait for that screening to finish, then start v22
until [ -f ~/vlmg-data/train/eval/scr_sft_coa7/summary.json ] && ! pgrep -f "eval_suite --tag scr_sft_coa7" >/dev/null; do sleep 60; done
sleep 5
setsid nohup bash train/run_queue.sh train/queues/v22.txt > ~/vlmg-results/v22.log 2>&1 < /dev/null &
echo "v22 started at $(date)"
