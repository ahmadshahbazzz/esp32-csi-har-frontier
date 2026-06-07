#!/bin/bash
# Flash a prebuilt model to the classic ESP32 and capture latency + RAM.
# Usage:  bash flash_measure.sh <cnn|transformer|kan> [PORT]
# Prereq: board attached; if perms error -> sudo chmod 666 /dev/ttyUSB0
set -e
MODEL="${1:?usage: flash_measure.sh <cnn|transformer|kan> [port]}"
PORT="${2:-/dev/ttyUSB0}"
PB=~/Downloads/Crypto_Project/esp32_firmware/csi_bench/prebuilt
PY=~/.espressif/python_env/idf5.1_py3.13_env/bin/python
ESPTOOL=~/Downloads/Crypto_Project/esp32_firmware/esp-idf/components/esptool_py/esptool/esptool.py
CAP=/tmp/capture_esp32.py

[ -f "$PB/$MODEL.bin" ] || { echo "no $PB/$MODEL.bin (run build first)"; exit 1; }

echo "=== flashing $MODEL to $PORT (classic ESP32 offsets) ==="
"$PY" "$ESPTOOL" -p "$PORT" -b 460800 --before default_reset --after hard_reset --chip esp32 \
  write_flash --flash_mode dio --flash_size 4MB --flash_freq 40m \
  0x1000  "$PB/bootloader.bin" \
  0x8000  "$PB/partition-table.bin" \
  0x10000 "$PB/$MODEL.bin"

echo "=== capturing serial ($MODEL) ==="
"$PY" "$CAP" "$PORT" 240
