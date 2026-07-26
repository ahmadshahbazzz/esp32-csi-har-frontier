#!/usr/bin/env bash
# Item #17: -Os (size) vs -O2 (perf) at 240 MHz. Measures representative models built
# with CONFIG_COMPILER_OPTIMIZATION_SIZE; compare to the *_240 (perf) results.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
FR=../frontier_results; BM=../benchmark_results; DEEP=$FR/deeptier_csihar/tflite
DEF=csi_bench/sdkconfig.defaults
cp "$DEF" "$DEF.bak"
# switch PERF -> SIZE, keep 240 MHz
sed -i 's/^CONFIG_COMPILER_OPTIMIZATION_PERF=y/CONFIG_COMPILER_OPTIMIZATION_SIZE=y/' "$DEF"
rm -f csi_bench/sdkconfig
run(){ echo "######## $2 (-Os, 240 MHz) ########"; bash measure_tiny.sh "$1" "$2"; }
run $FR/tflite/TinyCNN16_uthar_int8.tflite  TinyCNN16_uthar_Os
run $FR/tflite/TinyCNN16_csihar_int8.tflite TinyCNN16_csihar_Os
run $BM/tflite/CNN_int8.tflite              CNN_uthar_Os
run $DEEP/CNN_csihar_int8.tflite            CNN_csihar_Os
# restore perf config
mv "$DEF.bak" "$DEF"; rm -f csi_bench/sdkconfig
echo "######## -Os DONE (config restored to PERF) ########"
