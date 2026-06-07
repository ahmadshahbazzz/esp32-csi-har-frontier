#!/bin/bash
# Build + flash + measure ONE tiny int8 model on the classic ESP32.
# Usage: bash measure_tiny.sh <path-to-int8.tflite> <label> [port]
set -e
ROOT=/home/ahmad/Downloads/Crypto_Project/esp32_firmware
TFL="${1:?usage: measure_tiny.sh <model_int8.tflite> <label> [port]}"
LABEL="${2:?need a label, e.g. TinyCNN16_uthar}"
PY=~/.espressif/python_env/idf5.1_py3.13_env/bin/python

# The classic ESP32 re-enumerates (ttyUSB0 -> ttyUSB1 -> ...) under heavy reflashing.
# Auto-detect the current serial port and make it accessible each run.
detect_port() {
  for _ in $(seq 1 10); do
    p=$(ls /dev/ttyUSB* 2>/dev/null | head -1)
    [ -n "$p" ] && { echo "$p"; return; }
    sleep 1
  done
}
PORT="${3:-$(detect_port)}"
[ -n "$PORT" ] || { echo "NO SERIAL PORT FOUND"; exit 1; }
echo cout098 | sudo -S chmod 666 "$PORT" 2>/dev/null
echo "using port $PORT"
CAP=/tmp/capture_esp32.py
OUTDIR="$ROOT/../frontier_results/ondevice"; mkdir -p "$OUTDIR"

echo "=== [$LABEL] generating C array from $TFL ==="
python3 "$ROOT/tflite_to_c.py" "$TFL" "$ROOT/csi_bench/main/model_tiny.cc" model_data_tflite

echo "=== [$LABEL] building ==="
cd "$ROOT/csi_bench"
. "$ROOT/esp-idf/export.sh" >/dev/null 2>&1
idf.py build >/tmp/build_$LABEL.log 2>&1 || { echo "BUILD FAILED, see /tmp/build_$LABEL.log"; tail -20 /tmp/build_$LABEL.log; exit 1; }

echo "=== [$LABEL] flashing to $PORT ==="
idf.py -p "$PORT" -b 115200 flash >/tmp/flash_$LABEL.log 2>&1 || { echo "FLASH FAILED, see /tmp/flash_$LABEL.log"; tail -20 /tmp/flash_$LABEL.log; exit 1; }

echo "=== [$LABEL] capturing serial ==="
"$PY" "$CAP" "$PORT" 120 | tee "$OUTDIR/$LABEL.txt"
echo "=== [$LABEL] saved to $OUTDIR/$LABEL.txt ==="
