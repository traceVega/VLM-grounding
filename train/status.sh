#!/bin/bash
# State of the training runs and evaluations on disk (adapters, per-set evaluation line counts, last queue log lines).
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
date
echo "distro uptime (s): $(ps -o etimes= -p 1)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "--- adapters"
for d in sft_hint_v3 sft_hint_v3b_noise sft_multicrop_v4 grpo_v3d_hint grpo_v3e_long; do
  f=~/vlmg-data/train/$d/adapter/adapter_model.safetensors
  if [ -f "$f" ]; then printf "%-22s present  %s\n" "$d" "$(date -r "$f" +%m-%d_%H:%M)"; else printf "%-22s missing\n" "$d"; fi
done
echo "--- evaluations (lines per set)"
for t in scr_sft_hint_v3 scr_sft_hint_v3b_noise scr_sft_multicrop_v4 scr_grpo_v3d_hint full_grpo_v3d_hint scr_grpo_v3e_long; do
  line="$t:"
  for s in gme own refcoco gmegray gray; do
    f=~/vlmg-data/train/eval/$t/$s.jsonl
    [ -f "$f" ] && line="$line $s=$(wc -l < "$f")"
  done
  [ -f ~/vlmg-data/train/eval/$t/summary.json ] && line="$line [summary]"
  echo "$line"
done
echo "--- queue logs"
for l in round3c round4; do
  f=~/vlmg-results/$l.log
  [ -f "$f" ] && { echo "$l:"; grep -v "Loading weights" "$f" | grep -E "^===|saved|QUEUE|!!!" | tail -3; }
done
echo "--- processes"
pgrep -af "train[.]|run_queu[e]|handoff|round4" | cut -c1-100
