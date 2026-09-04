#!/bin/bash
# Run a resumable run_k1 stage until it finishes, restarting it if the
# environment kills it.
#
#     scripts/supervise.sh edits --shard 1000
#
# The K1 stages are resumable by design: `instances` skips images already in the
# store and `edits` skips edit keys already in the index, so a relaunch costs at
# most one unwritten shard.  That makes a supervisor the right answer to a host
# that terminates processes -- 2026-09-03, the WSL distro was restarting every
# minute or two, killing nohup, a systemd --user unit and a bare sleep alike.
# Preventing the kill is the host's problem; surviving it is ours.
#
# A restart is only worth attempting when the stage was killed rather than when
# it failed on its own: a traceback will repeat forever, so only a signal death
# (exit code above 128, or 0 progress) is retried, and the loop stops after
# MAX_RESTARTS consecutive attempts that made no progress.
set -u

STAGE="${1:?usage: supervise.sh <stage> [args...]}"
shift

REPO="${VLMG_REPO:-/mnt/d/Dev/ArcNova/auto-research/VLM-grounding}"
PYTHON="${VLMG_PYTHON:-$HOME/vlmg-env/bin/python}"
LOG="${VLMG_LOG:-$HOME/vlmg-data/k1_${STAGE}.log}"
MAX_RESTARTS="${VLMG_MAX_RESTARTS:-40}"
export VLMG_LAMA_DIR="${VLMG_LAMA_DIR:-$HOME/vlmg-data/raw/big-lama/big-lama}"

cd "$REPO" || exit 1

# What counts as progress.  Counted at depth 1 only: a finished bank holds
# hundreds of thousands of files, and a recursive du between every restart would
# cost more than the work it is guarding.
progress() {
  case "$STAGE" in
    edits)
      find "$HOME/vlmg-data/edits/openimages_pool" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
      ;;
    instances)
      find "$HOME/vlmg-data/prepared/openimages_pool/instances" -maxdepth 1 -name "shard-*.parquet" 2>/dev/null | wc -l
      ;;
    *) echo 0 ;;
  esac
}

attempt=0
stalled=0
while :; do
  attempt=$((attempt + 1))
  before=$(progress)
  echo "[$(date +%H:%M:%S)] === attempt $attempt: run_k1 $STAGE $* ===" | tee -a "$LOG"

  "$PYTHON" -u -m "idea91.run_k1" "$STAGE" "$@" >>"$LOG" 2>&1
  code=$?
  after=$(progress)

  if [ "$code" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] === $STAGE completed after $attempt attempt(s) ===" | tee -a "$LOG"
    exit 0
  fi

  # An ordinary failure is a bug and will repeat; only a killed process is
  # worth relaunching.  128+N is the shell's encoding of death by signal N.
  if [ "$code" -lt 129 ]; then
    echo "[$(date +%H:%M:%S)] === $STAGE exited $code (not a signal); not retrying ===" | tee -a "$LOG"
    tail -20 "$LOG"
    exit "$code"
  fi

  if [ "${after:-0}" -gt "${before:-0}" ]; then
    stalled=0
  else
    stalled=$((stalled + 1))
  fi
  echo "[$(date +%H:%M:%S)] killed with $code; progress ${before:-0} -> ${after:-0} " \
       "(stalled $stalled/$MAX_RESTARTS)" | tee -a "$LOG"

  if [ "$stalled" -ge "$MAX_RESTARTS" ]; then
    echo "[$(date +%H:%M:%S)] === $MAX_RESTARTS restarts with no progress; giving up ===" | tee -a "$LOG"
    exit 1
  fi
  sleep 5
done
