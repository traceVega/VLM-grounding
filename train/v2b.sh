#!/bin/bash
# After the v2 SFT probes (v2a) end: pick the better v2 SFT (vision frozen vs merger trainable) by screening net rejection,
# run the v2 RL (two-turn, pair reward, derived answer, commit + recall rewards) for one epoch on 40 scenes, then the
# screening and dev evaluations.
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/v2a2.log 2>/dev/null; do sleep 60; done
PY=~/vlmg-env/bin/python
CHOICE=$($PY - <<'EOF'
import json, os
def net(tag):
    p = os.path.expanduser(f"~/vlmg-data/train/eval/{tag}/summary.json")
    try:
        g = json.load(open(p))["gme"]
        return g["net_rejection"] + 0.5 * g["positive_acc"]  # net rejection with a positive-accuracy tie-breaker
    except Exception:
        return -9
have = lambda n: os.path.isfile(os.path.expanduser(f"~/vlmg-data/train/{n}/adapter/adapter_config.json"))
a, b = net("scr_sft_v2_half_b400"), net("scr_sft_v2h_rand")
print("sft_v2h_rand" if (have("sft_v2h_rand") and b >= a - 0.02) else ("sft_v2_half" if have("sft_v2_half") else "sft_v2"))
EOF
)
BEST=${CHOICE%% *}
echo "v2b at $(date): base SFT = $BEST (screening nets: see summaries)"
cat > train/queues/v2b.txt <<EOF
~/vlmg-env/bin/python -m train.grpo_lora --name grpo_v2 --trace rat --order random --turns 2 --max-new1 400 --reward v3 --answer derived --r-commit 0.3 --r-recall 0.3 --r-reason 0.3 --iou-soft --token-level --max-new 512 --init-adapter ~/vlmg-data/train/$BEST/adapter --extra ~/vlmg-data/train/augment/cross_v1.jsonl@20,~/vlmg-data/train/alt_negatives.jsonl@30,~/vlmg-data/train/short_items.jsonl@30,~/vlmg-data/train/mosaic_items.jsonl@60,~/vlmg-data/train/refcoco_train.jsonl@60 --epochs 1 --group 8 --prompts-per-step 4 --max-scenes 40 --r-pos-null -1 --inject-gt neg --eval-every 40 --mini-gme 20 --skip-eval0 --no-gray-eval
~/vlmg-env/bin/python -m train.eval_suite --tag scr_grpo_v2 --adapter ~/vlmg-data/train/grpo_v2/adapter --trace rat --turns 2 --max-new 512 --max-new1 400 --sets gme,own,refcoco --gme-rej-n 100 --gme-pos-n 100 --refcoco-n 50 --batch 4
EOF
setsid nohup bash train/run_queue.sh train/queues/v2b.txt > ~/vlmg-results/v2b.log 2>&1 < /dev/null &
echo "v2b started at $(date)"
