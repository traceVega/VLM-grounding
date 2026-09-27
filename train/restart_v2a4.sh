#!/bin/bash
# Stop the memory-bound screening eval and its queue runner (which holds the old --batch 8 line), then run the remaining
# v2a lines from a fresh queue and re-arm the v2b handoff on its log.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
pkill -f "run_queue[.]sh train/queues/v2a3"
pkill -f "v2b_after[.]sh"
pkill -f "eval_suite --tag scr_sft_v[2] "
sleep 5
setsid nohup bash train/run_queue.sh train/queues/v2a4.txt > ~/vlmg-results/v2a4.log 2>&1 < /dev/null &
setsid nohup bash train/v2b_after.sh v2a4 > ~/vlmg-results/v2b_handoff.log 2>&1 < /dev/null &
sleep 2
echo "restarted: $(pgrep -af 'run_queue[.]sh train/queues/v2a4' | wc -l) queue runner, $(pgrep -af 'v2b_after[.]sh v2a4' | wc -l) v2b waiter"
