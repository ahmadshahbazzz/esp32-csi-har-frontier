#!/usr/bin/env bash
# R2-7 in-domain data collection: build+flash the RECORD_CSI firmware, which forms CSI
# windows exactly as the deployed LIVE pipeline but streams each window over serial
# ("REC <idx> v0 ...") instead of classifying. Then use live_accuracy/record_labeled_csi.py
# on the host to capture labelled windows while a person performs each activity.
#
# Prereqisite: csi_bench/main/wifi_creds.h must define WIFI_SSID and WIFI_PASS (2.4 GHz AP).
# Usage: bash measure_record_live.sh
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
PORT="${1:-/dev/ttyUSB0}"; [ -e "$PORT" ] || PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -1)
echo "using port $PORT"; echo cout098 | sudo -S chmod 666 "$PORT" 2>/dev/null

export LIVE=1 RECORD_CSI=1
rm -f csi_bench/sdkconfig
( cd csi_bench && . "$ROOT/esp-idf/export.sh" >/dev/null 2>&1 && idf.py fullclean >/dev/null 2>&1 \
  && idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.live" set-target esp32 >/dev/null 2>&1 \
  && idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.live" build \
  && idf.py -p "$PORT" -b 115200 flash )
unset LIVE RECORD_CSI
rm -f csi_bench/sdkconfig
echo
echo "RECORD firmware flashed. Now record each activity from the host, e.g.:"
echo "  python3 live_accuracy/record_labeled_csi.py --port $PORT --label walk --session s1 --seconds 90 --out live_dataset"
echo "Repeat for all 7 activities and at least 2 sessions, then:"
echo "  python3 live_accuracy/train_eval_live.py --data live_dataset --out live_out"
