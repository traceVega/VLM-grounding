#!/bin/bash
# Start round 3 once the (orphaned) V3d evaluation has written its summary.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until [ -f ~/vlmg-data/train/eval/scr_grpo_v3d_hint/summary.json ]; do sleep 30; done
sleep 20
rm -rf ~/vlmg-data/train/eval/scr_sft_hint_v3_h5mp
setsid nohup bash train/run_queue.sh train/queues/round3.txt > ~/vlmg-results/round3.log 2>&1 < /dev/null &
echo "round 3 started at $(date)"
