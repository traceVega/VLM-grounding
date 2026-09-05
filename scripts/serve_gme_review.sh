#!/usr/bin/env bash
# Bring the review UI back up. Panels and labels are on disk, so a restart
# costs nothing and loses nothing -- useful because a WSL distro that gets
# reaped takes every background process with it.
set -u
cd "$(dirname "$0")/.."
. ~/vlmg-env/bin/activate
pkill -f "review_gm[e].py --serve" 2>/dev/null || true
sleep 1
nohup python scripts/review_gme.py --serve > /tmp/gme_review_server.log 2>&1 &
sleep 3
curl -s -o /dev/null -w "http://127.0.0.1:8902  HTTP %{http_code}\n" http://127.0.0.1:8902/
python scripts/review_gme.py --status
