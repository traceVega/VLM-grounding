#!/usr/bin/env bash
# Stage 1 on its own, so it can start while the rest of the chain is written.
set -u
cd "$(dirname "$0")/.."
. ~/vlmg-env/bin/activate
export VLMG_LAMA_DIR="${VLMG_LAMA_DIR:-$HOME/vlmg-data/raw/big-lama/big-lama}"
count() { python -c "import json,os;p='tables/gme_removals_index.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)"; }
TARGET=${1:-333}
prev=0
for attempt in $(seq 1 25); do
    now=$(count)
    [ "$now" -ge "$TARGET" ] && { echo "[edits] complete $now/$TARGET"; break; }
    echo "[edits] attempt $attempt at $now/$TARGET"
    python scripts/build_gme_removals.py
    now=$(count)
    [ "$now" -ge "$TARGET" ] && { echo "[edits] complete $now/$TARGET"; break; }
    [ "$now" -le "$prev" ] && { echo "[edits] NO PROGRESS at $now/$TARGET"; break; }
    prev=$now
done
echo "[edits] final: $(count)/$TARGET"
