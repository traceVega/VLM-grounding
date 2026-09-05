#!/usr/bin/env bash
# The whole GroundingME chain, unattended, supervised on progress at each stage.
#
# This host drops a transient cudaErrorUnknown often enough that no stage of a
# few hundred items finishes in one attempt, and a poisoned CUDA context cannot
# be recovered in-process. Every stage checkpoints per item and resumes, so the
# loop only has to restart it and notice when the file stops growing.
set -u

cd "$(dirname "$0")/.."
. ~/vlmg-env/bin/activate
export VLMG_LAMA_DIR="${VLMG_LAMA_DIR:-$HOME/vlmg-data/raw/big-lama/big-lama}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# stage <name> <target> <count-command> <run-command...>
stage() {
    local name=$1 target=$2 count_cmd=$3; shift 3
    local prev=0 now
    for attempt in $(seq 1 25); do
        now=$(eval "$count_cmd" 2>/dev/null || echo 0)
        if [ "$now" -ge "$target" ]; then log "$name: $now/$target done"; return 0; fi
        log "$name: attempt $attempt at $now/$target"
        "$@"
        now=$(eval "$count_cmd" 2>/dev/null || echo 0)
        if [ "$now" -ge "$target" ]; then log "$name: complete $now/$target"; return 0; fi
        if [ "$now" -le "$prev" ]; then
            log "$name: NO PROGRESS at $now/$target -- moving on with what exists"
            return 1
        fi
        prev=$now
    done
    log "$name: attempts exhausted at $now/$target"
    return 1
}

TARGET=${1:-333}

log "=== stage 1/4: build REMOVE edits (SAM 3 + LaMa) ==="
stage edits "$TARGET" \
    "python -c \"import json,os;print(len(json.load(open('tables/gme_removals_index.json'))) if os.path.exists('tables/gme_removals_index.json') else 0)\"" \
    python scripts/build_gme_removals.py

BUILT=$(python -c "import json,os;print(len(json.load(open('tables/gme_removals_index.json'))) if os.path.exists('tables/gme_removals_index.json') else 0)")
log "edits built: $BUILT"

log "=== stage 2/4: REMOVE inference ==="
stage infer "$BUILT" \
    "wc -l < tables/gme_remove.jsonl" \
    python scripts/gme_remove_infer.py --no-report

log "=== stage 3/4: translate the expressions for review ==="
stage translate "$BUILT" \
    "wc -l < tables/gme_expr_zh.jsonl" \
    python scripts/translate_expressions.py

log "=== stage 4/4: render the review panels ==="
python scripts/review_gme.py --build
log "panels: $(ls ~/vlmg-data/gme_review/panels 2>/dev/null | wc -l)"

log "=== starting the review server on 8902 ==="
pkill -f "review_gm[e].py --serve" 2>/dev/null || true
sleep 1
nohup python scripts/review_gme.py --serve > /tmp/gme_review_server.log 2>&1 &
sleep 4
curl -s -o /dev/null -w "server HTTP %{http_code}\n" http://127.0.0.1:8902/ || echo "server DID NOT COME UP"

log "=== ALL STAGES DONE ==="
python scripts/review_gme.py --status
