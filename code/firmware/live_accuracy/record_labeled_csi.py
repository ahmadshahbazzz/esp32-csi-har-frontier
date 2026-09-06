#!/usr/bin/env python3
"""
R2-7 in-domain live CSI recorder.

Reads the RECORD_CSI firmware serial stream (lines "REC <idx> v0 v1 ... v(T*F-1)") while
the person performs ONE activity, and appends the labelled windows to a per-session .npz.

Flash the RECORD firmware first (see measure_record_live.sh), then, for each activity, run:

    python3 record_labeled_csi.py --port /dev/ttyUSB0 --label walk --session s1 \
            --seconds 90 --out live_dataset

Do one activity per invocation. Repeat for all seven activities (bend, fall, lie_down, run,
sit_down, stand_up, walk). Record at least two sessions; to measure a held-out-session number,
move the person/board slightly between sessions and label them s1, s2, s3.

Each window is (T, F) = (64, 52) row-major amplitude, exactly as the deployed pipeline forms it.
"""
import argparse, sys, time, os
import numpy as np

T, F = 64, 52
ACTS = ["bend", "fall", "lie_down", "run", "sit_down", "stand_up", "walk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", required=True, help="activity being performed now")
    ap.add_argument("--session", required=True, help="session id, e.g. s1")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--out", default="live_dataset", help="output directory")
    a = ap.parse_args()
    if a.label not in ACTS:
        print(f"warning: label '{a.label}' not in {ACTS}", file=sys.stderr)

    try:
        import serial
    except ImportError:
        sys.exit("pip install pyserial")

    os.makedirs(a.out, exist_ok=True)
    ser = serial.Serial(a.port, a.baud, timeout=2)
    print(f"recording label={a.label} session={a.session} for {a.seconds}s on {a.port} ...")
    print("PERFORM THE ACTIVITY NOW.")
    wins, t0 = [], time.time()
    while time.time() - t0 < a.seconds:
        line = ser.readline().decode(errors="ignore").strip()
        if not line.startswith("REC "):
            if line.startswith("REC_STAT"):
                print("  " + line)
            continue
        parts = line.split()
        vals = parts[2:]
        if len(vals) != T * F:
            continue  # partial line; skip
        try:
            arr = np.asarray([float(x) for x in vals], dtype=np.float32).reshape(T, F)
        except ValueError:
            continue
        wins.append(arr)
        if len(wins) % 10 == 0:
            print(f"  captured {len(wins)} windows", end="\r")
    ser.close()

    if not wins:
        sys.exit("no windows captured; check the RECORD firmware is flashed and the AP is up.")
    X = np.stack(wins)
    y = np.array([ACTS.index(a.label)] * len(X), dtype=np.int64) if a.label in ACTS else None
    fn = os.path.join(a.out, f"{a.session}_{a.label}.npz")
    np.savez_compressed(fn, X=X, y=y, label=a.label, session=a.session)
    print(f"\nsaved {len(X)} windows -> {fn}")


if __name__ == "__main__":
    main()
