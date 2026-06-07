#!/bin/bash
# Measure every tiny int8 model on the classic ESP32, sequentially.
ROOT=/home/ahmad/Downloads/Crypto_Project
TFLDIR="$ROOT/frontier_results/tflite"
OUT="$ROOT/frontier_results/ondevice"; mkdir -p "$OUT"
SUMMARY="$OUT/summary.csv"
echo "model,arena_used_bytes,arena_kB,latency_ms,fits" > "$SUMMARY"

for f in TinyCNN8_uthar TinyCNN16_uthar TinyCNN32_uthar TinyMLP_nn_uthar \
         TinyCNN8_csihar TinyCNN16_csihar TinyCNN32_csihar TinyMLP_nn_csihar; do
    tfl="$TFLDIR/${f}_int8.tflite"
    [ -f "$tfl" ] || { echo "SKIP $f (no $tfl)"; continue; }
    echo "############ $f ############"
    bash "$ROOT/esp32_firmware/measure_tiny.sh" "$tfl" "$f" || { echo "$f,ERROR,,," >> "$SUMMARY"; continue; }
    sleep 2  # let the board settle / re-enumerate before the next reflash
    res="$OUT/$f.txt"
    arena=$(grep -oE 'arena_used_bytes = [0-9]+' "$res" | grep -oE '[0-9]+' | head -1)
    lat=$(grep -oE 'latency_ms = [0-9.]+' "$res" | grep -oE '[0-9.]+' | head -1)
    if [ -n "$arena" ]; then
        kb=$(python3 -c "print(f'{$arena/1024:.1f}')")
        echo "$f,$arena,$kb,$lat,yes" >> "$SUMMARY"
    else
        fail=$(grep -oE 'ALLOC FAIL[^"]*' "$res" | head -1)
        echo "$f,,,,no(${fail:-noalloc})" >> "$SUMMARY"
    fi
done
echo "===== SUMMARY ====="
cat "$SUMMARY"
echo "ALL_DONE"
