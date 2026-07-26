#!/usr/bin/env bash
# Item #25: tensor-arena-size ablation. Builds the 16-channel tiny CNN (CSI-HAR,
# arena_used ~= 8.5 kB) with a forced ARENA_BYTES and records whether AllocateTensors
# succeeds and the resulting latency, to map the fit threshold and show that latency is
# flat once the arena is large enough. 240 MHz.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
MODEL=../frontier_results/tflite/TinyCNN16_csihar_int8.tflite
OUT=../frontier_results/ondevice
: > "$OUT/arena_ablation.txt"
for A in 7168 8192 8448 8704 16384 32768; do
  export ARENA_BYTES=$A
  echo "######## arena=$A B ########"
  # CMake reads $ENV{ARENA_BYTES} at configure time and caches it; touch CMakeLists so
  # idf.py reconfigures and picks up the new value on every iteration.
  touch csi_bench/main/CMakeLists.txt
  bash measure_tiny.sh "$MODEL" "arena_${A}"
  t="$OUT/arena_${A}.txt"
  ok=$(grep -aq "ALLOC FAIL" "$t" && echo FAIL || (grep -aq "arena_used_bytes" "$t" && echo OK || echo ?))
  lat=$(grep -a -oE 'latency_ms = [0-9.]+' "$t" | tail -1 | grep -oE '[0-9.]+')
  used=$(grep -a -oE 'arena_used_bytes = [0-9]+' "$t" | tail -1 | grep -oE '[0-9]+')
  echo "arena=$A  allocate=$ok  used=${used:-na}  latency=${lat:-na}" | tee -a "$OUT/arena_ablation.txt"
done
unset ARENA_BYTES
rm -f csi_bench/sdkconfig  # drop the ARENA define from the cached config
echo "######## arena ablation DONE ########"
cat "$OUT/arena_ablation.txt"
