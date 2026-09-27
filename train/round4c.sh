#!/bin/bash
# After night3.txt ends: pick the best hint-crop SFT (v3, v3b noise, v4 multi-crop) by GME-dev net rejection and run the
# V3d recipe on it for 2 epochs (matching --n-crops), then its dev evaluation.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/night3.log 2>/dev/null; do sleep 60; done
PY=~/vlmg-env/bin/python
CHOICE=$($PY - <<'EOF'
import json, os
def net(tag):
    p = os.path.expanduser(f"~/vlmg-data/train/eval/{tag}/summary.json")
    try:
        return json.load(open(p))["gme"]["net_rejection"]
    except Exception:
        return -9
c = [("sft_hint_v3", "scr_sft_hint_v3", 1), ("sft_hint_v3b_noise", "scr_sft_hint_v3b_noise", 1), ("sft_multicrop_v4", "scr_sft_multicrop_v4", 3)]
best = max(c, key=lambda x: net(x[1]))
print(best[0], best[2])
EOF
)
BEST=${CHOICE% *}; NCROPS=${CHOICE#* }
echo "round 4 at $(date): base SFT = $BEST, n-crops = $NCROPS"
cat > train/queues/round4.txt <<EOF
~/vlmg-env/bin/python -m train.grpo_lora --name grpo_v3e_long --trace yn --order commit --hint-crop --n-crops $NCROPS --reward v3 --answer derived --r-commit 0.5 --init-adapter ~/vlmg-data/train/$BEST/adapter --extra ~/vlmg-data/train/augment/cross_v1.jsonl@40,~/vlmg-data/train/alt_negatives.jsonl@60,~/vlmg-data/train/short_items.jsonl@60,~/vlmg-data/train/refcoco_train.jsonl@100 --epochs 2 --group 8 --prompts-per-step 4 --max-scenes 50 --r-pos-null -1 --inject-gt neg --eval-every 1000 --skip-eval0 --no-gray-eval
~/vlmg-env/bin/python -m train.eval_suite --tag scr_grpo_v3e_long --adapter ~/vlmg-data/train/grpo_v3e_long/adapter --trace yn --hint-crop --n-crops $NCROPS --sets gme,own,refcoco --gme-pos-n 150 --refcoco-n 100 --batch 6
EOF
setsid nohup bash train/run_queue.sh train/queues/round4.txt > ~/vlmg-results/round4.log 2>&1 < /dev/null &
echo "round 4 started at $(date)"
