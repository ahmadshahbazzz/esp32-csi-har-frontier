#!/usr/bin/env bash
# Record one activity with audible cues: a single beep when recording actually starts
# (perform now) and a triple beep when the time is up (stop). Usage:
#   bash rec_with_cue.sh <label> <session> <seconds> [board_ip]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LABEL="$1"; SESS="$2"; SECS="$3"; IP="${4:-192.168.1.17}"
SND="$HERE/sounds"; LOG=/tmp/rec_cue.log
play(){ pw-play "$1" 2>/dev/null || aplay "$1" 2>/dev/null || printf '\a'; }

python3 "$HERE/record_labeled_csi.py" --board-ip "$IP" --label "$LABEL" --session "$SESS" \
        --seconds "$SECS" --out "$HERE/live_dataset" > "$LOG" 2>&1 &
PID=$!
# beep the instant recording begins
until grep -q 'PERFORM THE ACTIVITY NOW' "$LOG" 2>/dev/null; do
  kill -0 "$PID" 2>/dev/null || break
  sleep 0.2
done
play "$SND/start.wav"
wait "$PID"
play "$SND/stop.wav"
grep -E 'saved|no windows' "$LOG" | tail -1
