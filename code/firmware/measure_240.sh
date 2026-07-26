#!/usr/bin/env bash
# Re-measure all deployable models at 240 MHz (item #1 + corrected main numbers).
# The existing *_lat.txt / summary.csv files are the 160 MHz results (boot log
# confirmed cpu freq 160 MHz); sdkconfig.defaults now sets 240 MHz. Sequential.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
FR=../frontier_results
BM=../benchmark_results
DEEP=$FR/deeptier_csihar/tflite

run() { echo "######## $2 (240 MHz) ########"; bash measure_tiny.sh "$1" "$2"; }

# tiny tier
run $FR/tflite/TinyCNN8_uthar_int8.tflite   TinyCNN8_uthar_240
run $FR/tflite/TinyCNN16_uthar_int8.tflite  TinyCNN16_uthar_240
run $FR/tflite/TinyCNN32_uthar_int8.tflite  TinyCNN32_uthar_240
run $FR/tflite/TinyMLP_nn_uthar_int8.tflite TinyMLP_nn_uthar_240
run $FR/tflite/TinyCNN8_csihar_int8.tflite   TinyCNN8_csihar_240
run $FR/tflite/TinyCNN16_csihar_int8.tflite  TinyCNN16_csihar_240
run $FR/tflite/TinyCNN32_csihar_int8.tflite  TinyCNN32_csihar_240
run $FR/tflite/TinyMLP_nn_csihar_int8.tflite TinyMLP_nn_csihar_240
# deep tier
run $BM/tflite/CNN_int8.tflite          CNN_uthar_240
run $BM/tflite/Transformer_int8.tflite  Transformer_uthar_240
run $DEEP/CNN_csihar_int8.tflite         CNN_csihar_240
run $DEEP/Transformer_csihar_int8.tflite Transformer_csihar_240
echo "######## ALL 240 MHz MEASUREMENTS DONE ########"
