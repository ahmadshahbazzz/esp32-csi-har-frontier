#!/usr/bin/env python3
"""
R2-7 in-domain live CSI recorder (self-contained).

Opening the serial port resets the board, so this script:
  1. reads the boot log and auto-detects the board's IP (line "sta ip: X.X.X.X"),
  2. starts an internal unicast UDP traffic generator to that IP so CSI callbacks fire
     (the classic ESP32 only produces CSI for unicast frames addressed to it), then
  3. records the streamed windows ("REC <idx> v0 ... v(T*F-1)") for --seconds, labelling
     them with --label and --session, and appends to a per-session .npz.

Do one activity per invocation:
    python3 record_labeled_csi.py --port /dev/ttyUSB0 --label walk --session s1 --seconds 120

Each window is (T,F)=(64,52) row-major amplitude, exactly as the deployed pipeline forms it.
"""
import argparse, sys, time, os, re, socket, threading
import numpy as np

T, F = 64, 52
ACTS = ["bend", "fall", "lie_down", "run", "sit_down", "stand_up", "walk"]


def traffic_thread(ip, port, stop):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pkt = b"x" * 200
    while not stop.is_set():
        for _ in range(60):
            try: s.sendto(pkt, (ip, port))
            except OSError: pass
        time.sleep(0.003)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default="live_dataset")
    ap.add_argument("--udp-port", type=int, default=5001)
    ap.add_argument("--board-ip", default=None, help="skip autodetect and use this IP")
    a = ap.parse_args()
    if a.label not in ACTS:
        print(f"warning: label '{a.label}' not in {ACTS}", file=sys.stderr)
    try:
        import serial
    except ImportError:
        sys.exit("pip install pyserial")

    os.makedirs(a.out, exist_ok=True)
    ser = serial.Serial(a.port, a.baud, timeout=1)

    # 1. force a hardware reset (EN via RTS) so we get a fresh boot log with the IP and a
    #    fresh 30-minute LIVE window, then wait for the board to (re)join and report its IP
    ip = a.board_ip
    if not ip:
        ser.setDTR(False); ser.setRTS(True); time.sleep(0.15)
        ser.setRTS(False); time.sleep(0.1)
        ser.reset_input_buffer()
        print("resetting board; waiting for it to join WiFi and report its IP ...")
        t0 = time.time()
        while time.time() - t0 < 25:
            ln = ser.readline().decode(errors="ignore")
            m = re.search(r"sta ip:\s*(\d+\.\d+\.\d+\.\d+)", ln)
            if m:
                ip = m.group(1); break
            if "no IP" in ln:
                ser.close(); sys.exit("board could not join the AP (check wifi_creds.h / 2.4GHz).")
        if not ip:
            ser.close(); sys.exit("did not see the board's IP; is it joining the AP?")
    print(f"board IP {ip}; starting traffic and recording '{a.label}' for {a.seconds}s ...")

    # 2. start traffic so CSI flows
    stop = threading.Event()
    th = threading.Thread(target=traffic_thread, args=(ip, a.udp_port, stop), daemon=True)
    th.start()

    # 3. record windows
    print("PERFORM THE ACTIVITY NOW.")
    wins = []
    ser.reset_input_buffer()
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        line = ser.readline().decode(errors="ignore").strip()
        if not line.startswith("REC "):
            continue
        vals = line.split()[2:]
        if len(vals) != T * F:
            continue
        try:
            wins.append(np.asarray([float(x) for x in vals], np.float32).reshape(T, F))
        except ValueError:
            continue
        if len(wins) % 5 == 0:
            print(f"  captured {len(wins)} windows", end="\r")
    stop.set(); ser.close()

    if not wins:
        sys.exit("\nno windows captured (traffic/CSI problem). Check the board is streaming.")
    X = np.stack(wins)
    y = np.array([ACTS.index(a.label)] * len(X), np.int64) if a.label in ACTS else None
    fn = os.path.join(a.out, f"{a.session}_{a.label}.npz")
    np.savez_compressed(fn, X=X, y=y, label=a.label, session=a.session)
    print(f"\nsaved {len(X)} windows -> {fn}")


if __name__ == "__main__":
    main()
