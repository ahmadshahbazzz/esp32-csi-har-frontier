#!/usr/bin/env bash
# R2-7 guided in-domain recording session.
# Generates WiFi traffic to the board (so CSI flows), then walks you through recording
# each of the 7 activities. Run once per session: bash run_recording_session.sh s1
# Then move/change position a little and run again: bash run_recording_session.sh s2
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${2:-/dev/ttyUSB0}"
BOARD_IP="${3:-192.168.1.17}"
SESSION="${1:?usage: bash run_recording_session.sh <session-id e.g. s1> [port] [board_ip]}"
SECS="${SECS:-120}"       # seconds per activity (override: SECS=90 bash ...)
OUT="$HERE/live_dataset"
ACTS=(bend fall lie_down run sit_down stand_up walk)

echo "Board IP $BOARD_IP, port $PORT, session $SESSION, ${SECS}s per activity."
echo "Making sure the board is reachable ..."
ping -c 2 -W 2 "$BOARD_IP" >/dev/null 2>&1 || { echo "Cannot reach $BOARD_IP. Is the board on the AP? (check the boot log)"; exit 1; }

# background traffic generator so CSI callbacks keep firing during recording
python3 - "$BOARD_IP" <<'PYEOF' &
import socket,sys,time
ip=sys.argv[1]; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); p=b'x'*200
while True:
    for _ in range(50): s.sendto(p,(ip,5001))
    time.sleep(0.004)
PYEOF
TRAFFIC=$!
trap 'kill $TRAFFIC 2>/dev/null' EXIT
sleep 1

for a in "${ACTS[@]}"; do
  echo
  echo "==================================================================="
  echo "  NEXT ACTIVITY: ${a^^}"
  echo "  Get into position. Press ENTER, then perform '${a}' continuously"
  echo "  for ${SECS} seconds (repeat the motion the whole time)."
  echo "==================================================================="
  read -r _
  echo "Recording ${a} ... GO"
  python3 "$HERE/record_labeled_csi.py" --port "$PORT" --label "$a" --session "$SESSION" --seconds "$SECS" --out "$OUT"
  echo "Done ${a}."
done

kill $TRAFFIC 2>/dev/null
echo
echo "Session $SESSION complete. Files in $OUT."
echo "Recorded windows per activity:"
python3 - "$OUT" "$SESSION" <<'PYEOF'
import sys,glob,numpy as np,os
out,sess=sys.argv[1],sys.argv[2]
for f in sorted(glob.glob(os.path.join(out,f"{sess}_*.npz"))):
    d=np.load(f,allow_pickle=True); print(f"  {os.path.basename(f)}: {len(d['X'])} windows")
PYEOF
echo
echo "Now either run session 2 (move a little first):  bash run_recording_session.sh s2"
echo "or, when you have >=2 sessions, train + evaluate:  python3 $HERE/train_eval_live.py --data $OUT --out $HERE/live_out"
