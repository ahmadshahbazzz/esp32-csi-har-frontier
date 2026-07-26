#!/usr/bin/env bash
# Item #8: measure with ESP-NN DISABLED (portable reference kernels) to quantify the
# speedup ESP-NN gives. Compare to the *_240 results (ESP-NN on, the default). 240 MHz.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
FR=../frontier_results; DEEP=$FR/deeptier_csihar/tflite
export NO_ESP_NN=1
rm -f csi_bench/sdkconfig
( cd csi_bench && . "$ROOT/esp-idf/export.sh" >/dev/null 2>&1 && idf.py fullclean >/dev/null 2>&1 )
run(){ echo "######## $2 (ESP-NN OFF, 240 MHz) ########"; bash measure_tiny.sh "$1" "$2"; }
run $FR/tflite/TinyCNN16_csihar_int8.tflite TinyCNN16_csihar_noespnn
run $DEEP/CNN_csihar_int8.tflite            CNN_csihar_noespnn
unset NO_ESP_NN
rm -f csi_bench/sdkconfig
( cd csi_bench && . "$ROOT/esp-idf/export.sh" >/dev/null 2>&1 && idf.py fullclean >/dev/null 2>&1 )
echo "######## ESP-NN experiment DONE (reverted to ESP-NN on) ########"
