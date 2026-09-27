#!/bin/bash
# Round 4 (overnight 2026-09-25): after round 3 ends, pick the better hint-crop SFT (plain v3 vs hint-noise v3b) by
# GME-dev net rejection, run the V3d recipe on it for 3 epochs, evaluate on the dev sets, then the full GME set and
# the gray controls for the resulting candidate.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/round3.log 2>/dev/null; do sleep 60; done
PY=~/vlmg-env/bin/python
BEST=$($PY - <<'EOF'
import json, os
def net(tag):
    p = os.path.expanduser(f"~/vlmg-data/train/eval/{tag}/summary.json")
    try:
        return json.load(open(p))["gme"]["net_rejection"]
    except Exception:
        return -9
print("sft_hint_v3b_noise" if net("scr_sft_hint_v3b_noise") > net("scr_sft_hint_v3") else "sft_hint_v3")
EOF
)
echo "round 4 at $(date): base SFT = $BEST"
cat > train/queues/round4.txt <<EOF
~/vlmg-env/bin/python -m train.grpo_lora --name grpo_v3e_long --trace yn --order commit --hint-crop --reward v3 --answer derived --r-commit 0.5 --init-adapter ~/vlmg-data/train/$BEST/adapter --extra ~/vlmg-data/train/augment/cross_v1.jsonl@40,~/vlmg-data/train/alt_negatives.jsonl@60,~/vlmg-data/train/short_items.jsonl@60,~/vlmg-data/train/refcoco_train.jsonl@100 --epochs 3 --group 8 --prompts-per-step 4 --max-scenes 50 --r-pos-null -1 --inject-gt neg --eval-every 1000 --skip-eval0 --no-gray-eval
~/vlmg-env/bin/python -m train.eval_suite --tag scr_grpo_v3e_long --adapter ~/vlmg-data/train/grpo_v3e_long/adapter --trace yn --hint-crop --sets gme,own,refcoco --gme-pos-n 150 --refcoco-n 100 --batch 8
~/vlmg-env/bin/python -m train.eval_suite --tag full_grpo_v3d_hint --adapter ~/vlmg-data/train/grpo_v3d_hint/adapter --trace yn --hint-crop --sets gme --batch 8
~/vlmg-env/bin/python -m train.eval_suite --tag scr_grpo_v3d_hint --adapter ~/vlmg-data/train/grpo_v3d_hint/adapter --trace yn --hint-crop --sets gmegray,gray --gme-gray-pos-n 100 --batch 8
EOF
setsid nohup bash train/run_queue.sh train/queues/round4.txt > ~/vlmg-results/round4.log 2>&1 < /dev/null &
echo "round 4 started at $(date)"
