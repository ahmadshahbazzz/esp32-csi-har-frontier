#!/usr/bin/env python3
"""Generates csi_hpsweep.ipynb (review item #8): learning-rate x batch-size sweep
for CNN, Transformer, and BiGRU on CSI-HAR, to justify the selected training
hyperparameters. Reports accuracy per configuration and the best per model.
Attach sayakghorai34/csi-har-dataset. Run: python gen_hpsweep_notebook.py"""
import json
from pathlib import Path
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})

md(r"""# Hyperparameter sweep (item #8): learning rate x batch size

For CNN, Transformer, and BiGRU on CSI-HAR (stratified 80/20 split), we sweep
learning rate in {1e-2, 1e-3, 1e-4} and batch size in {32, 64, 128} and report the
accuracy of each configuration and the best per model, to justify the training
settings used throughout the paper. Attach **sayakghorai34/csi-har-dataset**.
""")

code(r"""import os, re, json, warnings
from pathlib import Path
import numpy as np, tensorflow as tf, pandas as pd
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPool1D, GlobalAveragePooling1D, Dense, MultiHeadAttention, LayerNormalization, Bidirectional, GRU
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
warnings.filterwarnings("ignore")
print("TensorFlow", tf.__version__)
OUT = Path("/kaggle/working"); TARGET_T = 64; EPOCHS = 40
""")

code(r"""def find_csi_har_root():
    for c in Path("/kaggle/input").rglob("CSI-HAR-Dataset"):
        if c.is_dir(): return c
    for c in Path("/kaggle/input").rglob("*"):
        if c.is_dir() and (c/"walk").is_dir() and (c/"run").is_dir(): return c
    return None
def load_csi_har_raw(T=TARGET_T):
    root = find_csi_har_root()
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
    m = X.mean(axis=(1,2), keepdims=True); s = X.std(axis=(1,2), keepdims=True)+1e-8; X = ((X-m)/s).astype(np.float32)
    return X, y, acts
X, Y, ACTS = load_csi_har_raw(); NCLS = len(ACTS)
tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, Y))
Xtr, ytr, Xte, yte = X[tr], Y[tr], X[te], Y[te]
print("train", Xtr.shape, "test", Xte.shape)
""")

code(r"""def cnn(T,F,n):
    i=Input((T,F)); x=Conv1D(64,7,padding="same",activation="relu")(i); x=BatchNormalization()(x); x=MaxPool1D(2)(x)
    x=Conv1D(128,5,padding="same",activation="relu")(x); x=BatchNormalization()(x); x=GlobalAveragePooling1D()(x)
    x=Dense(64,activation="relu")(x); return Model(i,Dense(n)(x))
def transformer(T,F,n):
    i=Input((T,F)); x=Conv1D(64,5,padding="same")(i); a=MultiHeadAttention(num_heads=4,key_dim=16)(x,x)
    x=LayerNormalization()(x+a); f=Dense(128,activation="relu")(x); f=Dense(64)(f); x=LayerNormalization()(f)
    x=GlobalAveragePooling1D()(x); return Model(i,Dense(n)(x))
def bigru(T,F,n):
    i=Input((T,F)); x=Bidirectional(GRU(64))(i); x=Dense(64,activation="relu")(x); return Model(i,Dense(n)(x))
BUILDERS = {"CNN": cnn, "Transformer": transformer, "BiGRU": bigru}

def run(builder, lr, bs):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(0)
    m = builder(int(Xtr.shape[1]), int(Xtr.shape[2]), NCLS)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=bs, verbose=0)
    return float(accuracy_score(yte, m.predict(Xte, verbose=0).argmax(-1)))
""")

code(r"""LRS = [1e-2, 1e-3, 1e-4]; BSS = [32, 64, 128]
rows = []; best = {}
for name, b in BUILDERS.items():
    grid = {}
    for lr in LRS:
        for bs in BSS:
            acc = run(b, lr, bs); grid[(lr, bs)] = acc
            rows.append({"Model": name, "lr": lr, "batch": bs, "Accuracy": round(acc*100, 2)})
            print(f"{name:12s} lr={lr:.0e} bs={bs:3d}  acc={acc*100:.1f}")
    bk = max(grid, key=grid.get); best[name] = {"lr": bk[0], "batch": bk[1], "acc": round(grid[bk]*100, 2)}
    print(f"  -> best {name}: lr={bk[0]:.0e} batch={bk[1]} acc={grid[bk]*100:.1f}\n")
df = pd.DataFrame(rows); df.to_csv(OUT/"hpsweep_table.csv", index=False)
json.dump(best, open(OUT/"hpsweep_best.json", "w"), indent=2)
print("BEST PER MODEL:"); print(json.dumps(best, indent=2))
df
""")

md(r"""## Result
The best configuration per model is saved to `hpsweep_best.json`. The paper's default
(learning rate 1e-3, batch size 64) can be reported as selected by this sweep.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent/"csi_hpsweep.ipynb"; out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
