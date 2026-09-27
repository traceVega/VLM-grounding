#!/bin/bash
# Wait until file $1 has at least $2 lines matching pattern $3 (default "sft"), polling every $4 seconds (default 60).
# Use this instead of $(...) inside a wsl.exe -- bash -lc string, where the substitution is expanded by the outer shell.
FILE=$1; N=$2; PAT=${3:-sft}; POLL=${4:-60}
until [ -f "$FILE" ] && [ "$(grep -c "$PAT" "$FILE")" -ge "$N" ]; do sleep "$POLL"; done
