#!/bin/bash
# Same as train/v2b.sh but waits on the queue log named by $1 (default v2a3).
LOG=${1:-v2a3}
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding || exit 1
until grep -q "QUEUE DONE" ~/vlmg-results/$LOG.log 2>/dev/null; do sleep 60; done
sed "s#~/vlmg-results/v2a2.log#~/vlmg-results/$LOG.log#" train/v2b.sh > /tmp/v2b_now.sh
tail -n +7 /tmp/v2b_now.sh | grep -v "^until grep" > /tmp/v2b_body.sh   # the body after the wait: choose the SFT, write and start the queue
cd /mnt/d/Dev/ArcNova/auto-research/VLM-grounding && bash /tmp/v2b_body.sh
