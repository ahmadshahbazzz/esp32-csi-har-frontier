#!/usr/bin/env python3
"""
R2-7 all-in-one guided in-domain recording session.

Resets the board ONCE (fresh 30-minute LIVE window + fresh IP), detects the board IP, starts
a continuous unicast-UDP traffic generator so CSI keeps flowing, and keeps the serial port open
for the whole session (so the board stays warm and the window rate stays high). Then it walks
you through the seven activities, recording each for a fixed time and saving per-activity .npz.

    python3 record_session.py --session s1 --seconds 120
    # move/change position a little, then:
    python3 record_session.py --session s2 --seconds 120
    # when you have >=2 sessions:
    python3 train_eval_live.py --data live_dataset --out live_out
"""
import argparse, sys, time, os, re, socket, threading
import numpy as np

T, F = 64, 52
ACTS = ["bend", "fall", "lie_down", "run", "sit_down", "stand_up", "walk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--session", required=True)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--udp-port", type=int, default=5001)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_dataset")
    os.makedirs(out, exist_ok=True)
    try:
        import serial
    except ImportError:
        sys.exit("pip install pyserial")

    ser = serial.Serial(a.port, a.baud, timeout=1)
    # reset once (EN via RTS) for a fresh boot log + fresh 30-min LIVE timer
    ser.setDTR(False); ser.setRTS(True); time.sleep(0.15); ser.setRTS(False); time.sleep(0.1)
    ser.reset_input_buffer()
    print("resetting board; waiting for WiFi + IP ...")
    ip = None; t0 = time.time()
    while time.time() - t0 < 25:
        ln = ser.readline().decode(errors="ignore")
        m = re.search(r"sta ip:\s*(\d+\.\d+\.\d+\.\d+)", ln)
        if m: ip = m.group(1); break
        if "no IP" in ln:
            sys.exit("board could not join the AP (check wifi_creds.h / 2.4 GHz).")
    if not ip:
        sys.exit("did not see the board's IP; is it joining the AP?")
    print(f"board IP {ip}. traffic on. keeping the board warm for the whole session.")

    stop = threading.Event()
    def traffic():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); pkt = b"x" * 200
        while not stop.is_set():
            for _ in range(80):
                try: s.sendto(pkt, (ip, a.udp_port))
                except OSError: pass
            time.sleep(0.002)
    threading.Thread(target=traffic, daemon=True).start()
    time.sleep(3)  # let the CSI rate ramp up before recording

    def record(label, secs):
        ser.reset_input_buffer()
        wins = []; t = time.time()
        while time.time() - t < secs:
            line = ser.readline().decode(errors="ignore").strip()
            if not line.startswith("REC "): continue
            vals = line.split()[2:]
            if len(vals) != T * F: continue
            try: wins.append(np.asarray([float(x) for x in vals], np.float32).reshape(T, F))
            except ValueError: continue
            if len(wins) % 5 == 0: print(f"    {len(wins)} windows", end="\r")
        return wins

    total = 0
    for a_name in ACTS:
        print("\n" + "=" * 60)
        print(f"  NEXT: {a_name.upper()}   (get in position)")
        try: input(f"  Press ENTER, then do '{a_name}' for {int(a.seconds)} s ...")
        except EOFError: pass
        print(f"  RECORDING {a_name} ... PERFORM NOW")
        wins = record(a_name, a.seconds)
        if not wins:
            print(f"  WARNING: 0 windows for {a_name}; check the board is still streaming.")
            continue
        X = np.stack(wins)
        y = np.array([ACTS.index(a_name)] * len(X), np.int64)
        fn = os.path.join(out, f"{a.session}_{a_name}.npz")
        np.savez_compressed(fn, X=X, y=y, label=a_name, session=a.session)
        print(f"  saved {len(X)} windows -> {os.path.basename(fn)}")
        total += len(X)

    stop.set(); ser.close()
    print(f"\nSession {a.session} done: {total} windows total in {out}.")
    print("Run session 2 (move a little first) or, with >=2 sessions, run train_eval_live.py.")


if __name__ == "__main__":
    main()
