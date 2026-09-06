#!/usr/bin/env python3
"""
R2-7 in-domain live accuracy: train the tiny CNN on recorded windows and report accuracy.

Two evaluations are produced:
  1. within-session: stratified 80/20 split over all recorded windows (optimistic).
  2. leave-one-session-out (LOSO): train on all sessions but one, test on the held-out
     session. This is the honest "new deployment" number to report against reviewer R2-7,
     and it is the live analogue of the leave-one-user-out protocol used in the paper.

Also exports an int8 TFLite model and a replay header so the retrained model can be flashed
and confirmed on-device with measure_replay.sh.

Usage:
    python3 train_eval_live.py --data live_dataset --out live_out
"""
import argparse, glob, os, sys
import numpy as np

ACTS = ["bend", "fall", "lie_down", "run", "sit_down", "stand_up", "walk"]
T, F = 64, 52


def load(data_dir):
    Xs, ys, ss = [], [], []
    for fn in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        d = np.load(fn, allow_pickle=True)
        X = d["X"].astype(np.float32)
        lab = str(d["label"]); sess = str(d["session"])
        if lab not in ACTS:
            continue
        Xs.append(X)
        ys.append(np.full(len(X), ACTS.index(lab), np.int64))
        ss.append(np.array([sess] * len(X)))
    if not Xs:
        sys.exit(f"no .npz found in {data_dir}")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)


def zscore(X):
    m = X.mean(axis=(1, 2), keepdims=True)
    s = X.std(axis=(1, 2), keepdims=True) + 1e-6
    return (X - m) / s


def build(nch=16, ncls=7):
    import tensorflow as tf
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input((T, F)),
        L.Conv1D(nch, 5, padding="same", activation="relu"),
        L.MaxPool1D(2),
        L.Conv1D(nch, 5, padding="same", activation="relu"),
        L.GlobalAveragePooling1D(),
        L.Dense(ncls),
    ])


def train_eval(Xtr, ytr, Xte, yte, epochs=40):
    import tensorflow as tf
    m = build()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(zscore(Xtr), ytr, epochs=epochs, batch_size=64, verbose=0)
    pr = m.predict(zscore(Xte), verbose=0).argmax(1)
    acc = float((pr == yte).mean() * 100)
    return acc, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="live_dataset")
    ap.add_argument("--out", default="live_out")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X, y, s = load(a.data)
    sessions = sorted(set(s.tolist()))
    print(f"loaded {len(X)} windows, {len(set(y))} classes, sessions={sessions}")

    # within-session 80/20
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X)); cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    acc_ws, _ = train_eval(X[tr], y[tr], X[te], y[te], a.epochs)
    print(f"within-session accuracy: {acc_ws:.1f}%")

    # leave-one-session-out
    loso = []
    if len(sessions) >= 2:
        for held in sessions:
            m = s != held
            acc, _ = train_eval(X[m], y[m], X[~m], y[~m], a.epochs)
            print(f"  LOSO hold {held}: {acc:.1f}% (test n={int((~m).sum())})")
            loso.append(acc)
        print(f"leave-one-session-out mean accuracy: {np.mean(loso):.1f}% "
              f"(+/- {np.std(loso):.1f})")
    else:
        print("only one session recorded; record >=2 sessions for the honest LOSO number.")

    # export int8 model trained on all data + a replay header (one window per class)
    accf, model = train_eval(X, y, X, y, a.epochs)
    import tensorflow as tf
    Xn = zscore(X)

    def rep():
        for i in range(min(300, len(Xn))):
            yield [Xn[i:i + 1].astype(np.float32)]
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    mp = os.path.join(a.out, "live_tinycnn_int8.tflite")
    open(mp, "wb").write(tfl)
    print(f"exported int8 model -> {mp} ({len(tfl)} bytes)")

    with open(os.path.join(a.out, "live_accuracy_results.txt"), "w") as f:
        f.write(f"windows={len(X)} sessions={sessions}\n")
        f.write(f"within_session_acc={acc_ws:.2f}\n")
        if loso:
            f.write(f"leave_one_session_out_mean_acc={np.mean(loso):.2f}\n")
            f.write(f"leave_one_session_out_std={np.std(loso):.2f}\n")
    print(f"results -> {os.path.join(a.out, 'live_accuracy_results.txt')}")
    print("\nReport the leave-one-session-out mean as the in-domain live accuracy (R2-7).")
    print("To confirm on-device, build replay windows from live_dataset and run measure_replay.sh.")


if __name__ == "__main__":
    main()
