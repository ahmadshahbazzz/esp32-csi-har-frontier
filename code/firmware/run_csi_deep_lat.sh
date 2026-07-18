#!/usr/bin/env bash
# Clean sequential re-measurement of the two CSI-HAR deep models (CNN, Transformer)
# with the fixed watchdog firmware (yield every 10 invokes). One board, one port,
# strictly sequential. UT-HAR deep models were already measured by the b3rgegbt3 batch.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DEEP=../frontier_results/deeptier_csihar/tflite

echo "[driver] CNN CSI-HAR ..."
bash measure_tiny.sh "$DEEP/CNN_csihar_int8.tflite" CNN_csihar_lat
echo "[driver] CNN_csihar_lat done"

echo "[driver] Transformer CSI-HAR ..."
bash measure_tiny.sh "$DEEP/Transformer_csihar_int8.tflite" Transformer_csihar_lat
echo "[driver] Transformer_csihar_lat done"
echo "[driver] ALL CSI-HAR DEEP DONE"
