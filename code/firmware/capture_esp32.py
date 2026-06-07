#!/usr/bin/env python3
# Reset the ESP32 into run mode and capture the benchmark prints.
# Persistent copy (lives in the repo so a reboot wiping /tmp does not lose it).
import serial, sys, time
port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
timeout_s = int(sys.argv[2]) if len(sys.argv) > 2 else 240
s = serial.Serial(port, 115200, timeout=2)
s.setDTR(False)   # IO0 high (run, not bootloader)
s.setRTS(True)    # EN low (assert reset)
time.sleep(0.15)
s.setRTS(False)   # EN high (release -> boot)
t0 = time.time()
got = {"arena": None, "latency": None}
while time.time() - t0 < timeout_s:
    try:
        line = s.readline().decode("utf-8", "ignore").strip()
    except Exception:
        continue
    if not line:
        continue
    print("DEV:", line, flush=True)
    if "arena_used_bytes" in line: got["arena"] = line
    if "latency_ms" in line: got["latency"] = line
    if "DONE" in line or "ALLOC FAIL" in line or "Guru Meditation" in line:
        break
s.close()
print("---CAPTURED---", flush=True)
print("ARENA:", got["arena"], flush=True)
print("LATENCY:", got["latency"], flush=True)
