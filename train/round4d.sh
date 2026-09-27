#!/bin/bash
# After night3.txt ends: the V3d recipe on sft_hint_v3 (single crop) for 2 epochs (longer RL), then its dev evaluation.
# The multi-crop SFT v4 gained only +2 net rejection at 2-3x the cost, so the single-crop SFT stays the RL base.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/night3.log 2>/dev/null; do sleep 60; done
BEST=sft_hint_v3
NCROPS=1
echo "round 4 at $(date): base SFT = $BEST, n-crops = $NCROPS"
cat > train/queues/round4.txt <<EOF
~/vlmg-env/bin/python -m train.grpo_lora --name grpo_v3e_long --trace yn --order commit --hint-crop --n-crops $NCROPS --reward v3 --answer derived --r-commit 0.5 --init-adapter ~/vlmg-data/train/$BEST/adapter --extra ~/vlmg-data/train/augment/cross_v1.jsonl@40,~/vlmg-data/train/alt_negatives.jsonl@60,~/vlmg-data/train/short_items.jsonl@60,~/vlmg-data/train/refcoco_train.jsonl@100 --epochs 2 --group 8 --prompts-per-step 4 --max-scenes 50 --r-pos-null -1 --inject-gt neg --eval-every 1000 --skip-eval0 --no-gray-eval
~/vlmg-env/bin/python -m train.eval_suite --tag scr_grpo_v3e_long --adapter ~/vlmg-data/train/grpo_v3e_long/adapter --trace yn --hint-crop --n-crops $NCROPS --sets gme,own,refcoco --gme-pos-n 150 --refcoco-n 100 --batch 8
EOF
setsid nohup bash train/run_queue.sh train/queues/round4.txt > ~/vlmg-results/round4.log 2>&1 < /dev/null &
echo "round 4 started at $(date)"
