#!/usr/bin/env python3
"""
Generates csi_metrics.ipynb (review items #3, #4, and the data for #2).
On CSI-HAR (subject-independent leave-one-user-out), it:
  #3 builds confusion matrices for the best Tiny CNN, Random Forest, and Deep CNN,
  #4 reports precision / recall / F1 (per-class + macro) for those models,
  #2 dumps per-fold accuracy arrays for several models (incl. CNN and Transformer)
     so Friedman / Wilcoxon significance tests can be computed afterwards.
Attach sayakghorai34/csi-har-dataset. Run: python gen_metrics_notebook.py
"""
import json
from pathlib import Path
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})

md(r"""# Confusion matrices, P/R/F1, and per-fold accuracies (CSI-HAR, leave-one-user-out)

Items #3, #4, and the raw data for #2. Every model is trained on two users and
tested on the held-out third; predictions are pooled across the 3 folds so each
sample is predicted exactly once as an unseen user. Attach
**sayakghorai34/csi-har-dataset**; GPU T4, Internet On.
""")

code(r"""import os, re, json, warnings
from pathlib import Path
import numpy as np, tensorflow as tf, pandas as pd
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPool1D, GlobalAveragePooling1D, Dense, MultiHeadAttention, LayerNormalization
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
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
    return X, y, users, acts
def csi_features(X):
    mean=X.mean(1); std=X.std(1); mn=X.min(1); mx=X.max(1); rng=mx-mn
    return np.concatenate([mean,std,mn,mx,rng],axis=1).astype(np.float32)

X, Y, U, ACTS = load_csi_har_raw(); NCLS = len(ACTS)
folds = [(U != u, U == u) for u in sorted(set(U.tolist()))]
print("classes:", ACTS, "| folds:", len(folds))
""")

code(r"""# ---- model builders ----
def tiny_cnn(T,F,n):
    i=Input((T,F)); x=Conv1D(32,7,padding="same",activation="relu")(i)
    x=MaxPool1D(2)(x); x=Conv1D(64,5,padding="same",activation="relu")(x)
    x=GlobalAveragePooling1D()(x); return Model(i,Dense(n)(x),name="TinyCNN32")
def deep_cnn(T,F,n):
    i=Input((T,F)); x=Conv1D(64,7,padding="same",activation="relu")(i)
    x=BatchNormalization()(x); x=MaxPool1D(2)(x)
    x=Conv1D(128,5,padding="same",activation="relu")(x); x=BatchNormalization()(x)
    x=GlobalAveragePooling1D()(x); x=Dense(64,activation="relu")(x)
    return Model(i,Dense(n)(x),name="DeepCNN")
def transformer(T,F,n):
    i=Input((T,F)); x=Conv1D(64,5,padding="same")(i)
    a=MultiHeadAttention(num_heads=4,key_dim=16)(x,x); x=LayerNormalization()(x+a)
    f=Dense(128,activation="relu")(x); f=Dense(64)(f); x=LayerNormalization()(f)
    x=GlobalAveragePooling1D()(x); return Model(i,Dense(n)(x),name="Transformer")

def fit_predict_nn(builder, Xtr, ytr, Xte, seed=0):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    m=builder(int(Xtr.shape[1]),int(Xtr.shape[2]),NCLS)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(Xtr,ytr,epochs=EPOCHS,batch_size=64,verbose=0)
    return m.predict(Xte,verbose=0).argmax(-1)
def fit_predict_rf(Xtr, ytr, Xte, seed=0):
    clf=RandomForestClassifier(n_estimators=20,max_depth=10,random_state=seed,n_jobs=-1)
    clf.fit(csi_features(Xtr),ytr); return clf.predict(csi_features(Xte))

NN = {"TinyCNN32": tiny_cnn, "DeepCNN": deep_cnn, "Transformer": transformer}
""")

code(r"""# ---- run LOUO: pool predictions + record per-fold accuracy ----
preds = {name: (np.zeros_like(Y), np.zeros_like(Y)) for name in list(NN) + ["RandomForest"]}
# store as dict name -> (y_true_pooled, y_pred_pooled)
pooled = {name: {"true": [], "pred": []} for name in list(NN) + ["RandomForest"]}
perfold = {name: [] for name in list(NN) + ["RandomForest"]}

for fi, (tr, te) in enumerate(folds):
    Xtr, ytr, Xte, yte = X[tr], Y[tr], X[te], Y[te]
    for name, b in NN.items():
        yp = fit_predict_nn(b, Xtr, ytr, Xte, seed=0)
        pooled[name]["true"].append(yte); pooled[name]["pred"].append(yp)
        perfold[name].append(round(float(accuracy_score(yte, yp))*100, 2))
    yp = fit_predict_rf(Xtr, ytr, Xte, seed=0)
    pooled["RandomForest"]["true"].append(yte); pooled["RandomForest"]["pred"].append(yp)
    perfold["RandomForest"].append(round(float(accuracy_score(yte, yp))*100, 2))
    print(f"fold {fi}: " + "  ".join(f"{n}={perfold[n][-1]:.1f}" for n in perfold))

for name in pooled:
    pooled[name]["true"] = np.concatenate(pooled[name]["true"])
    pooled[name]["pred"] = np.concatenate(pooled[name]["pred"])
json.dump(perfold, open(OUT/"perfold_accuracy.json", "w"), indent=2)
print("\nper-fold accuracy saved for significance tests (#2):"); print(json.dumps(perfold, indent=1))
""")

code(r"""# ---- #3 confusion matrices + #4 precision/recall/F1 ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

prf_rows = []
for name in ["TinyCNN32", "RandomForest", "DeepCNN"]:
    yt, yp = pooled[name]["true"], pooled[name]["pred"]
    cm = confusion_matrix(yt, yp, labels=list(range(NCLS)))
    pd.DataFrame(cm, index=ACTS, columns=ACTS).to_csv(OUT/f"cm_{name}.csv")
    # figure
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(NCLS)); ax.set_xticklabels(ACTS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(NCLS)); ax.set_yticklabels(ACTS, fontsize=8)
    for i in range(NCLS):
        for j in range(NCLS):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] < cm.max()/2 else "black", fontsize=8)
    ax.set_title(f"{name} confusion matrix (CSI-HAR, LOUO)"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(OUT/f"fig_cm_{name}.png", dpi=200); plt.close(fig)
    # macro P/R/F1
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    acc = accuracy_score(yt, yp)
    prf_rows.append({"Model": name, "Accuracy": round(acc*100,2), "Precision": round(p*100,2), "Recall": round(r*100,2), "F1": round(f1*100,2)})
    print(f"\n== {name} ==\n", classification_report(yt, yp, target_names=ACTS, zero_division=0))

prf = pd.DataFrame(prf_rows)
prf.to_csv(OUT/"prf1_table.csv", index=False)
print("\nMacro precision/recall/F1 (CSI-HAR):"); print(prf.to_string(index=False))
print("\nsaved cm_*.csv, fig_cm_*.png, prf1_table.csv, perfold_accuracy.json")
prf
""")

md(r"""## Next
Download `cm_*.png` (item #3), `prf1_table.csv` (item #4), and `perfold_accuracy.json`
(item #2). The activity confusions to highlight are walk vs run, sit vs stand, and
fall vs lie down. Significance tests (Friedman / Wilcoxon) are computed locally from
the per-fold accuracies.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent/"csi_metrics.ipynb"; out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
