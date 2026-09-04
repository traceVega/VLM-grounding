#!/bin/bash
# Everything the critical path can do unattended once the edit bank is built,
# stopping at the point where a human is required.
#
#     scripts/after_bank.sh <pid of the supervisor>
#
# Order is P20's. Check 1a's ladders and the check-1b nulls run BEFORE the K1
# freeze -- the design permits it explicitly, because they see no real removal
# and because check 1a may raise the gate resolution, which is the one declared
# pre-freeze contingency. The gate rows are NOT run here: they train on real
# removals, so they wait for a signed B0a. run_row refuses them anyway.
set -u

SUPERVISOR_PID="${1:?usage: after_bank.sh <supervisor pid>}"
REPO="${VLMG_REPO:-/mnt/d/Dev/ArcNova/auto-research/VLM-grounding}"
PYTHON="${VLMG_PYTHON:-$HOME/vlmg-env/bin/python}"
LOG="$HOME/vlmg-data/after_bank.log"
export VLMG_LAMA_DIR="${VLMG_LAMA_DIR:-$HOME/vlmg-data/raw/big-lama/big-lama}"

cd "$REPO" || exit 1
exec >>"$LOG" 2>&1

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for the edit bank (supervisor pid $SUPERVISOR_PID)"
while kill -0 "$SUPERVISOR_PID" 2>/dev/null; do sleep 60; done
say "supervisor exited"

# A truncated bank would give the freeze a count table that misdescribes it.
if ! tail -40 "$HOME/vlmg-data/k1_edits.log" | grep -q "edits done in"; then
  say "the edits stage did not report completion; stopping here"
  tail -20 "$HOME/vlmg-data/k1_edits.log"
  exit 1
fi
say "=== bank complete ==="
"$PYTHON" -m idea91.run_k1 status

# Q-3 first: it is minutes, and it is the last thing blocking the K2 harness.
say "=== Q-3: Qwen3-VL's coordinate convention ==="
"$PYTHON" scripts/probe_coordinates.py --n 8 || say "Q-3 probe failed; continuing"

# B4a. Pre-freeze by P20, and the step that may raise the gate resolution.
say "=== B4a: check 1a ladders and the check-1b nulls ==="
"$PYTHON" -m idea91.run_k1 checks
say "=== checks exited $? ==="

# Render the sign-off sheet against the real count table, and stop.
say "=== B0a sheet (nothing is frozen by this) ==="
"$PYTHON" -m idea91.run_k1 freeze

say "=== STOP: B0a needs a human ==="
say "  read   tables/FREEZE-B0a-sheet.md"
say "  then   python -m idea91.run_k1 freeze --sign-off '<name>'"
say "  then   python -m idea91.run_k1 rows --tiers verdict"
