#!/usr/bin/env bash
# Supervise the abstention run on progress, not on exit code.
#
# This host throws a transient `cudaErrorUnknown` mid-generation often enough
# that a few hundred items will not finish in one attempt, and a poisoned CUDA
# context cannot be recovered inside the process. The script checkpoints per
# item and resumes, so the only thing missing is something to restart it.
#
# The loop stops when the file stops growing, which covers both "finished" and
# "one image kills the driver every time" without needing to tell them apart.
set -u

cd "$(dirname "$0")/.."
. ~/vlmg-env/bin/activate

TARGET=${1:-332}
LABEL=${2:-clean}
if [ "$LABEL" = "clean" ]; then
    FILE=tables/pilot_abstain.jsonl
else
    FILE=tables/pilot_abstain_${LABEL}.jsonl
fi
prev=0

for attempt in $(seq 1 30); do
    python scripts/pilot_abstain_signal.py --no-report --label "$LABEL"
    now=$(wc -l < "$FILE" 2>/dev/null || echo 0)
    echo "=== attempt ${attempt}: ${now}/${TARGET} records ==="
    if [ "$now" -ge "$TARGET" ]; then
        echo "complete"
        break
    fi
    if [ "$now" -le "$prev" ]; then
        echo "no progress this attempt; stopping so it can be looked at"
        break
    fi
    prev=$now
done

echo
python scripts/pilot_abstain_signal.py --report --label "$LABEL"
