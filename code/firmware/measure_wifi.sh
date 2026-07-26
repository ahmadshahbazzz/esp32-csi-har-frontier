#!/usr/bin/env bash
# Item #2: measure inference with the WiFi stack ENABLED. Reports the largest free
# internal block with the radio up (vs the 152 kB radio-off budget), and whether the
# representative models still allocate and run. 240 MHz. Sequential.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
FR=../frontier_results; DEEP=$FR/deeptier_csihar/tflite
DEF=csi_bench/sdkconfig.defaults
cp "$DEF" "$DEF.bak"
# enable WiFi (keep BT/coex off)
sed -i 's/^CONFIG_ESP_WIFI_ENABLED=n/CONFIG_ESP_WIFI_ENABLED=y/' "$DEF"
grep -q "CONFIG_ESP_WIFI_ENABLED=y" "$DEF" || echo "CONFIG_ESP_WIFI_ENABLED=y" >> "$DEF"
rm -f csi_bench/sdkconfig
export WIFI_ON=1
( cd csi_bench && . "$ROOT/esp-idf/export.sh" >/dev/null 2>&1 && idf.py fullclean >/dev/null 2>&1 )
run(){ echo "######## $2 (WiFi ON, 240 MHz) ########"; bash measure_tiny.sh "$1" "$2"; }
run $FR/tflite/TinyCNN16_csihar_int8.tflite TinyCNN16_csihar_wifi
run $DEEP/CNN_csihar_int8.tflite            CNN_csihar_wifi
run $DEEP/Transformer_csihar_int8.tflite    Transformer_csihar_wifi
# restore
unset WIFI_ON
mv "$DEF.bak" "$DEF"; rm -f csi_bench/sdkconfig
( cd csi_bench && . "$ROOT/esp-idf/export.sh" >/dev/null 2>&1 && idf.py fullclean >/dev/null 2>&1 )
echo "######## WiFi experiment DONE (config restored) ########"
