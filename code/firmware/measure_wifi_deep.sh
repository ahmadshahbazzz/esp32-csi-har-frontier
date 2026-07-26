#!/usr/bin/env bash
# Item #2 (deep models): WiFi ON with a larger app partition so the deep-model
# firmware + WiFi stack fits the 4 MB flash. Confirms the deep models allocate and
# run within the reduced (radio-on) free block.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
DEEP=../frontier_results/deeptier_csihar/tflite
DEF=csi_bench/sdkconfig.defaults
cp "$DEF" "$DEF.bak"
sed -i 's/^CONFIG_ESP_WIFI_ENABLED=n/CONFIG_ESP_WIFI_ENABLED=y/' "$DEF"
grep -q "CONFIG_ESP_WIFI_ENABLED=y" "$DEF" || echo "CONFIG_ESP_WIFI_ENABLED=y" >> "$DEF"
# larger single-app partition (2 MB app in the 4 MB flash) so the WiFi+deep image fits
{ echo "CONFIG_PARTITION_TABLE_SINGLE_APP_LARGE=y";
  echo "# CONFIG_PARTITION_TABLE_SINGLE_APP is not set"; } >> "$DEF"
rm -f csi_bench/sdkconfig
export WIFI_ON=1
( cd csi_bench && . "$ROOT/esp-idf/export.sh" >/dev/null 2>&1 && idf.py fullclean >/dev/null 2>&1 )
run(){ echo "######## $2 (WiFi ON + large part, 240 MHz) ########"; bash measure_tiny.sh "$1" "$2"; }
run $DEEP/CNN_csihar_int8.tflite         CNN_csihar_wifi
run $DEEP/Transformer_csihar_int8.tflite Transformer_csihar_wifi
unset WIFI_ON
mv "$DEF.bak" "$DEF"; rm -f csi_bench/sdkconfig
( cd csi_bench && . "$ROOT/esp-idf/export.sh" >/dev/null 2>&1 && idf.py fullclean >/dev/null 2>&1 )
echo "######## deep WiFi DONE (config restored) ########"
