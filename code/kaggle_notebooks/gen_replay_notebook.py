#!/usr/bin/env python3
"""Generates csi_replay.ipynb (review item #11: on-device replay demo).

Trains the deployed 16-channel tiny CNN on CSI-HAR, quantizes it to full int8,
then exports (a) the tflite model and (b) one real, correctly-classified test
window per activity class as an int8 C header quantized to the model's own input
scale. The firmware replays these stored windows through the on-device interpreter
and prints its prediction per window, demonstrating end-to-end classification of
real CSI on the bare ESP32 (not just latency on a dummy input).

Attach sayakghorai34/csi-har-dataset. Run: python gen_replay_notebook.py
"""
import json
from pathlib import Path
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})

md(r"""# On-device replay demo (item #11)

Train the deployed 16-channel tiny CNN on CSI-HAR, quantize to full int8, and
export one real correctly-classified test window per class as an int8 C array
(quantized to the model's input scale/zero-point). The firmware replays these
stored windows through the on-device TFLite-Micro interpreter and prints its
prediction, showing real-CSI classification on the bare ESP32. Attach
**sayakghorai34/csi-har-dataset**; GPU T4, Internet On.
""")

code(r"""import os, re, json, warnings
from pathlib import Path
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
warnings.filterwarnings("ignore")
print("TensorFlow", tf.__version__)
OUT = Path("/kaggle/working"); TARGET_T = 64; EPOCHS = 50
""")

code(r"""def find_csi_har_root():
    for c in Path("/kaggle/input").rglob("CSI-HAR-Dataset"):
        if c.is_dir(): return c
    for c in Path("/kaggle/input").rglob("*"):
        if c.is_dir() and (c/"walk").is_dir() and (c/"run").is_dir(): return c
    return None
def load_csi_har_raw(T=TARGET_T):
    root = find_csi_har_root()
    if root is None: raise FileNotFoundError("CSI-HAR-Dataset not attached")
    files = [p for p in root.rglob("*_A.csv") if not p.name.startswith("Annotation")]
    acts = sorted({p.parent.name for p in files}); lmap = {a: i for i, a in enumerate(acts)}
    X=[]; y=[]
    for p in files:
        try: a = np.genfromtxt(str(p), delimiter=",")
        except Exception: continue
        if a.ndim == 1: a = a.reshape(-1, 1)
        if a.shape[0] < 2 or a.shape[1] < 2: continue
        idx = np.linspace(0, a.shape[0]-1, T).astype(int)
        X.append(a[idx, :].astype(np.float32)); y.append(lmap[p.parent.name])
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64)
    m = X.mean(axis=(1,2), keepdims=True); s = X.std(axis=(1,2), keepdims=True)+1e-8
    X = ((X-m)/s).astype(np.float32)
    return X, y, acts
X, Y, ACTS = load_csi_har_raw(); NCLS = len(ACTS)
tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(X, Y))
Xtr, ytr, Xte, yte = X[tr], Y[tr], X[te], Y[te]
print("classes", ACTS, "| train", Xtr.shape, "test", Xte.shape)
""")

code(r"""# ---- deployed 16-channel tiny CNN (same as the frontier paper) ----
def tiny_cnn(T, F, n, ch=16):
    i = layers.Input((T, F)); x = layers.Conv1D(ch, 7, padding="same", activation="relu")(i)
    x = layers.MaxPool1D(2)(x); x = layers.Conv1D(ch*2, 5, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x); return Model(i, layers.Dense(n)(x))
tf.keras.utils.set_random_seed(0)
model = tiny_cnn(TARGET_T, X.shape[2], NCLS, 16)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=64, verbose=0)
acc = accuracy_score(yte, model.predict(Xte, verbose=0).argmax(-1))
print(f"float test accuracy = {acc*100:.1f}%")
""")

code(r"""# ---- full int8 quantization ----
def rep_gen():
    for i in range(min(300, len(Xtr))):
        yield [Xtr[i:i+1].astype(np.float32)]
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8; conv.inference_output_type = tf.int8
tfl = conv.convert()
(OUT/"replay_model_int8.tflite").write_bytes(tfl)
print("int8 tflite bytes =", len(tfl))

interp = tf.lite.Interpreter(model_content=tfl); interp.allocate_tensors()
ind = interp.get_input_details()[0]; outd = interp.get_output_details()[0]
in_scale, in_zp = ind["quantization"]; print("input scale", in_scale, "zp", in_zp)
def tflite_predict(x_float):
    xq = np.round(x_float/in_scale + in_zp).clip(-128,127).astype(np.int8)
    interp.set_tensor(ind["index"], xq[None, ...]); interp.invoke()
    return int(interp.get_tensor(outd["index"])[0].argmax()), xq
""")

code(r"""# ---- pick one correctly-classified test window per class, export int8 C header ----
sel = []  # (int8_window[T,F], true_label)
for c in range(NCLS):
    idxs = np.where(yte == c)[0]
    for j in idxs:
        pred, xq = tflite_predict(Xte[j])
        if pred == c:
            sel.append((xq, c)); break
    else:  # none correct: take the first, still a real window
        pred, xq = tflite_predict(Xte[idxs[0]]); sel.append((xq, int(yte[idxs[0]])))
N = len(sel); T, F = sel[0][0].shape
print(f"exported {N} windows, {T}x{F} each")

lines = []
lines.append("// Auto-generated by csi_replay.ipynb (item #11 on-device replay demo).")
lines.append("// Real CSI-HAR test windows, int8-quantized to the deployed tiny CNN input scale.")
lines.append("#pragma once")
lines.append(f"#define REPLAY_N {N}")
lines.append(f"#define REPLAY_T {T}")
lines.append(f"#define REPLAY_F {F}")
names = ", ".join(f'"{a}"' for a in ACTS)
lines.append(f"static const char* kReplayClassNames[{NCLS}] = {{ {names} }};")
labs = ", ".join(str(l) for _, l in sel)
lines.append(f"static const int kReplayTrueLabel[REPLAY_N] = {{ {labs} }};")
lines.append("static const signed char kReplayWindows[REPLAY_N][REPLAY_T*REPLAY_F] = {")
for xq, _ in sel:
    flat = xq.reshape(-1).astype(np.int8)
    body = ",".join(str(int(v)) for v in flat)
    lines.append("  { " + body + " },")
lines.append("};")
(OUT/"replay_windows.h").write_text("\n".join(lines))
print("wrote replay_windows.h  (", (OUT/'replay_windows.h').stat().st_size, "bytes )")

# reference predictions from the desktop interpreter (what the device should reproduce)
import csv
with open(OUT/"replay_reference.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["i","true","tflite_pred"])
    for i,(xq,l) in enumerate(sel):
        # re-run the interpreter on the stored int8 window directly
        interp.set_tensor(ind["index"], xq[None,...].astype(np.int8)); interp.invoke()
        p = int(interp.get_tensor(outd["index"])[0].argmax())
        w.writerow([i, ACTS[l], ACTS[p]])
print("wrote replay_reference.csv")
""")

md(r"""## Next
Download `replay_windows.h`, `replay_model_int8.tflite`, and `replay_reference.csv`.
The firmware compiles the header, runs each stored window through the on-device
interpreter, and prints its prediction; comparing to `replay_reference.csv` (and the
true labels) shows real-CSI classification reproduced on the bare ESP32.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent/"csi_replay.ipynb"; out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
