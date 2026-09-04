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
# When to retry. The first version only relaunched signal deaths, on the
# reasoning that a traceback repeats forever. That was wrong for the failure it
# then met: the checks stage died with `CUDA error: unknown error` out of
# cuMemcpyHtoDAsync, which is the WSL GPU interface hiccupping and not a bug --
# the same copy ran thirty times immediately afterwards. Exit code cannot tell
# a transient fault from a deterministic one.
#
# So the rule is progress, not exit code: any death is retried, and the loop
# gives up after MAX_STALLED consecutive attempts that got nothing further done.
# A real bug costs a handful of attempts and stops; a flaky host gets as many
# chances as it keeps earning.
set -u

STAGE="${1:?usage: supervise.sh <stage> [args...]}"
shift

REPO="${VLMG_REPO:-/mnt/d/Dev/ArcNova/auto-research/VLM-grounding}"
PYTHON="${VLMG_PYTHON:-$HOME/vlmg-env/bin/python}"
LOG="${VLMG_LOG:-$HOME/vlmg-data/k1_${STAGE}.log}"
MAX_STALLED="${VLMG_MAX_STALLED:-4}"
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
    checks|rows)
      # each trained row is one cached json, so the count is the work done
      find "$HOME/vlmg-data/gate_cache" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l
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

  if [ "${after:-0}" -gt "${before:-0}" ]; then
    stalled=0
  else
    stalled=$((stalled + 1))
  fi
  echo "[$(date +%H:%M:%S)] exited $code; progress ${before:-0} -> ${after:-0}" \
       "(stalled $stalled/$MAX_STALLED)" | tee -a "$LOG"

  if [ "$stalled" -ge "$MAX_STALLED" ]; then
    echo "[$(date +%H:%M:%S)] === $MAX_STALLED attempts with no progress; giving up ===" \
      | tee -a "$LOG"
    tail -30 "$LOG"
    exit 1
  fi
  sleep 10
done
