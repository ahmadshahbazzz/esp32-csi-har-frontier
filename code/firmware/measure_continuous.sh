#!/usr/bin/env bash
# Items #10/#11: continuous-sensing / runtime-stability test. Runs the 16-channel tiny
# CNN back-to-back for ~12000 inferences (several minutes) at 240 MHz, sampling the free
# internal heap and largest free block to detect leaks or fragmentation and confirming
# latency stays stable.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
FR=../frontier_results
export CONTINUOUS=1
rm -f csi_bench/sdkconfig
( cd csi_bench && . "$ROOT/esp-idf/export.sh" >/dev/null 2>&1 && idf.py fullclean >/dev/null 2>&1 )
echo "######## continuous run (TinyCNN16 CSI-HAR, 240 MHz) ########"
bash measure_tiny.sh $FR/tflite/TinyCNN16_csihar_int8.tflite TinyCNN16_csihar_cont
unset CONTINUOUS
rm -f csi_bench/sdkconfig
echo "######## continuous run DONE (reverted) ########"
