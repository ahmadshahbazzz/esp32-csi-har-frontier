#!/usr/bin/env bash
# Item #16: tensor-arena allocation breakdown for the deployable deep models, via the
# RecordingMicroInterpreter (RECORD_ALLOC=1). Prints how the arena splits across
# persistent tensors, activation/eval tensors, and scratch buffers. 240 MHz.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
BM=../benchmark_results; DEEP=../frontier_results/deeptier_csihar/tflite
export RECORD_ALLOC=1
rm -f csi_bench/sdkconfig
( cd csi_bench && . "$ROOT/esp-idf/export.sh" >/dev/null 2>&1 && idf.py fullclean >/dev/null 2>&1 )
run(){ echo "######## $2 (RECORD_ALLOC, 240 MHz) ########"; bash measure_tiny.sh "$1" "$2"; }
run $BM/tflite/CNN_int8.tflite          CNN_uthar_rec
run $BM/tflite/Transformer_int8.tflite  Transformer_uthar_rec
run $DEEP/CNN_csihar_int8.tflite         CNN_csihar_rec
run $DEEP/Transformer_csihar_int8.tflite Transformer_csihar_rec
unset RECORD_ALLOC
rm -f csi_bench/sdkconfig
echo "######## RECORD_ALLOC DONE (reverted) ########"
