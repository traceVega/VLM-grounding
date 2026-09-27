#!/bin/bash
# Stop the memory-bound sft_v2 run and its queue, then restart the v2 SFT probes with the sparse-logits loss.
# Usage (WSL): bash train/restart_v2a.sh
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
pkill -f "run_queue[.]sh train/queues/v2a2"
pkill -f "v2b[.]sh"
pkill -f "sft_lora --name sft_v[2] "
sleep 5
rm -rf ~/vlmg-data/train/sft_v2
setsid nohup bash train/run_queue.sh train/queues/v2a3.txt > ~/vlmg-results/v2a3.log 2>&1 < /dev/null &
setsid nohup bash train/v2b_after.sh v2a3 > ~/vlmg-results/v2b_handoff.log 2>&1 < /dev/null &
sleep 2
echo "restarted: $(pgrep -af 'run_queue[.]sh train/queues/v2a3' | wc -l) queue runner, $(pgrep -af 'v2b_after[.]sh' | wc -l) v2b waiter"
