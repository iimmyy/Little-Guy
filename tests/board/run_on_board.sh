#!/bin/bash
# Run a board script with hard limits, from the Mac: connect within 5 s, drop a dead link within ~15 s, and kill the
# remote job after <max minutes>, so a frozen board shows up as "timed out" instead of an endless wait.
# Prints the remote output as it comes, plus a heartbeat every 60 s so silence never looks like progress.
# usage: tests/board/run_on_board.sh <max minutes> '<remote command>' <local log>
set -u
MAX=$1 CMD=$2 LOG=$3
( while sleep 60; do echo "[heartbeat $(date +%H:%M)] still running"; done ) & HB=$!
ssh -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 q6a "timeout $((MAX * 60)) bash -c '$CMD'; echo \"[remote exit \$?]\"" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}; kill $HB 2>/dev/null
[ $rc -eq 255 ] && echo "[board unreachable or link dropped]" | tee -a "$LOG"
exit $rc
