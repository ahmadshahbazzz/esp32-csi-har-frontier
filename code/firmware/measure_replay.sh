#!/usr/bin/env bash
# Item #11 on-device replay demo. Flashes the deployed tiny CNN together with stored
# real CSI windows and captures the device's per-window predictions.
# Usage: bash measure_replay.sh <replay_model_int8.tflite> <replay_windows.h>
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
TFL="${1:?usage: measure_replay.sh <replay_model_int8.tflite> <replay_windows.h>}"
HDR="${2:?usage: measure_replay.sh <replay_model_int8.tflite> <replay_windows.h>}"
LABEL=replay_demo

# port (re-enumerates under heavy reflashing)
PORT="${3:-/dev/ttyUSB0}"
[ -e "$PORT" ] || PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -1)
echo "using port $PORT"
echo cout098 | sudo -S chmod 666 "$PORT" 2>/dev/null

PY="$ROOT/esp-idf/tools/idf.py"
CAP="$ROOT/capture_esp32.py"
OUTDIR="$ROOT/../frontier_results/ondevice"; mkdir -p "$OUTDIR"
PROJ="$ROOT/csi_bench"

# stage model + windows into the firmware
python "$ROOT/tflite_to_c.py" "$TFL" "$PROJ/main/model_tiny.cc"
cp "$HDR" "$PROJ/main/replay_windows.h"

cd "$PROJ"
. "$ROOT/esp-idf/export.sh" >/dev/null 2>&1
export REPLAY_DEMO=1
idf.py fullclean >/dev/null 2>&1
idf.py build >/tmp/build_$LABEL.log 2>&1 || { echo "BUILD FAILED"; tail -25 /tmp/build_$LABEL.log; unset REPLAY_DEMO; exit 1; }
idf.py -p "$PORT" -b 115200 flash >/tmp/flash_$LABEL.log 2>&1 || { echo "FLASH FAILED"; tail -20 /tmp/flash_$LABEL.log; unset REPLAY_DEMO; exit 1; }

python "$CAP" "$PORT" 120 | tee "$OUTDIR/$LABEL.txt"
echo "=== [$LABEL] saved to $OUTDIR/$LABEL.txt ==="

# restore a clean (no-REPLAY_DEMO) configure so later latency builds are unaffected
unset REPLAY_DEMO
idf.py fullclean >/dev/null 2>&1
echo "[replay] firmware config restored to non-demo"
