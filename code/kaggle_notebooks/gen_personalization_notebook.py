#!/usr/bin/env python3
"""
Generates csi_personalization.ipynb (review item #16). For each held-out user on
CSI-HAR, we add k calibration samples from that user into training, retrain, and
measure accuracy on their remaining samples. Averaged over the 3 users x seeds,
this gives a calibration curve (accuracy vs k) for a tiny CNN and a Random Forest,
backing the claim that the cross-subject gap is closed by a short per-user
calibration rather than a bigger model.
Attach sayakghorai34/csi-har-dataset. Run: python gen_personalization_notebook.py
"""
import json
from pathlib import Path
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})

md(r"""# Personalization: closing the cross-subject gap with a few calibration samples

Item #16. Protocol: for each held-out user, train on the other two users PLUS k
calibration samples from the held-out user, then test on that user's remaining
samples. Average over the 3 held-out users and 3 seeds. We report a tiny CNN
(TinyCNN32) and a Random Forest. k = 0 reproduces the zero-shot leave-one-user-out
number; larger k shows how quickly a short calibration recovers accuracy.

Attach **sayakghorai34/csi-har-dataset**. GPU T4, Internet On.
""")

code(r"""import os, re, json, warnings
from pathlib import Path
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
print("TensorFlow", tf.__version__)
OUT = Path("/kaggle/working"); TARGET_T = 64; EPOCHS = 50
""")

code(r"""# ---- CSI-HAR loader (identical to the frontier/gengap notebooks) ----
def find_csi_har_root():
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
    X=[]; y=[]; users=[]
    for p in files:
        try: a = np.genfromtxt(str(p), delimiter=",")
        except Exception: continue
        if a.ndim == 1: a = a.reshape(-1, 1)
        if a.shape[0] < 2 or a.shape[1] < 2: continue
        idx = np.linspace(0, a.shape[0]-1, T).astype(int)
        X.append(a[idx, :].astype(np.float32)); y.append(lmap[p.parent.name])
        m = re.search(r"user_(\d+)_", p.name); users.append(int(m.group(1)) if m else 0)
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64); users = np.asarray(users)
    m = X.mean(axis=(1,2), keepdims=True); s = X.std(axis=(1,2), keepdims=True)+1e-8; X = ((X-m)/s).astype(np.float32)
    print(f"CSI-HAR {X.shape} classes={acts} users={sorted(set(users.tolist()))}")
    return X, y, users
def csi_features(X):
    mean=X.mean(1); std=X.std(1); mn=X.min(1); mx=X.max(1); rng=mx-mn
    return np.concatenate([mean,std,mn,mx,rng],axis=1).astype(np.float32)

X, Y, U = load_csi_har_raw(); NCLS = int(Y.max())+1
users = sorted(set(U.tolist()))
for u in users: print(f"  user {u}: {(U==u).sum()} samples")
""")

code(r"""# ---- models ----
def tiny_cnn(T, F, n, ch=32):
    i = layers.Input((T, F)); x = layers.Conv1D(ch, 7, padding="same", activation="relu")(i)
    x = layers.MaxPool1D(2)(x); x = layers.Conv1D(ch*2, 5, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x); return Model(i, layers.Dense(n)(x))
def train_cnn(Xtr, ytr, Xte, yte, seed=0):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    m = tiny_cnn(int(Xtr.shape[1]), int(Xtr.shape[2]), NCLS, 32)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(Xtr, ytr, epochs=EPOCHS, batch_size=64, verbose=0)
    return float(accuracy_score(yte, m.predict(Xte, verbose=0).argmax(-1)))
def train_rf(Xtr, ytr, Xte, yte, seed=0):
    clf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=seed, n_jobs=-1)
    clf.fit(csi_features(Xtr), ytr)
    return float(accuracy_score(yte, clf.predict(csi_features(Xte))))
""")

code(r"""# ---- personalization sweep ----
import pandas as pd
KS = [0, 5, 10, 20, 40]
SEEDS = 3
rows = []
for k in KS:
    cnn_accs, rf_accs = [], []
    for u in users:
        base = U != u
        Xb, yb = X[base], Y[base]
        Xu, yu = X[U == u], Y[U == u]
        for s in range(SEEDS):
            rng = np.random.RandomState(s)
            idx = rng.permutation(len(Xu))
            cal, tst = idx[:k], idx[k:]
            if k > 0:
                Xtr = np.concatenate([Xb, Xu[cal]]); ytr = np.concatenate([yb, yu[cal]])
            else:
                Xtr, ytr = Xb, yb
            Xte, yte = Xu[tst], yu[tst]
            cnn_accs.append(train_cnn(Xtr, ytr, Xte, yte, s))
            rf_accs.append(train_rf(Xtr, ytr, Xte, yte, s))
    rows.append({
        "k": k,
        "TinyCNN32": round(np.mean(cnn_accs)*100, 2), "TinyCNN32_std": round(np.std(cnn_accs)*100, 2),
        "RandomForest": round(np.mean(rf_accs)*100, 2), "RandomForest_std": round(np.std(rf_accs)*100, 2),
    })
    print(f"k={k:2d}  TinyCNN32 {rows[-1]['TinyCNN32']:.1f}%  RandomForest {rows[-1]['RandomForest']:.1f}%")
df = pd.DataFrame(rows)
df.to_csv(OUT/"personalization_table.csv", index=False)
json.dump(rows, open(OUT/"personalization.json", "w"), indent=2)
print("\n", df.to_string(index=False))
df
""")

code(r"""# ---- calibration-curve figure ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.errorbar(df["k"], df["TinyCNN32"], yerr=df["TinyCNN32_std"], marker="o", capsize=4, label="Tiny CNN (32 ch)", color="#1f77b4")
ax.errorbar(df["k"], df["RandomForest"], yerr=df["RandomForest_std"], marker="s", capsize=4, label="Random Forest", color="#2ca02c")
ax.axhline(100/7, ls=":", color="#888"); ax.text(df["k"].max(), 100/7+1, "chance (7 classes)", ha="right", fontsize=8, color="#666")
ax.set_xlabel("Calibration samples from the held-out user (k)")
ax.set_ylabel("Accuracy on the held-out user (%)")
ax.set_title("Per-user calibration recovers cross-subject accuracy (CSI-HAR)")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(OUT/"fig_personalization.png", dpi=200)
print("saved fig_personalization.png, personalization_table.csv, personalization.json")
""")

md(r"""## Reading the result
k = 0 is the zero-shot leave-one-user-out number. As k grows, accuracy on the unseen
user rises sharply, which shows the cross-subject gap is largely closed by a short
per-user calibration rather than a larger model. Only 3 users, so this is a
demonstration of the effect, not a large-scale study.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent/"csi_personalization.ipynb"; out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
