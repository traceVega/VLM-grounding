#!/usr/bin/env bash
# Same progress-based supervision as the abstention run: this host drops a
# transient cudaErrorUnknown often enough that a thousand items will not finish
# in one attempt, and a poisoned CUDA context cannot be recovered in-process.
set -u

cd "$(dirname "$0")/.."
. ~/vlmg-env/bin/activate

FILE=tables/gme_original.jsonl
TARGET=${1:-1005}
prev=0

for attempt in $(seq 1 30); do
    python scripts/gme_original_precheck.py --no-report
    now=$(wc -l < "$FILE" 2>/dev/null || echo 0)
    echo "=== attempt ${attempt}: ${now}/${TARGET} records ==="
    if [ "$now" -ge "$TARGET" ]; then echo complete; break; fi
    if [ "$now" -le "$prev" ]; then echo "no progress; stopping"; break; fi
    prev=$now
done

echo
python scripts/gme_original_precheck.py --report
