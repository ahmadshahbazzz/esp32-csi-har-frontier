#!/usr/bin/env python3
"""Generates csi_ntufi.ipynb (review item #5: third public CSI HAR dataset).

Runs the deployability-frontier pipeline on NTU-Fi_HAR (imhoangt/ntu-fi-dataset),
a clean whole-body HAR corpus in SenseFi format: {train,test}_amp/{activity}/{sample}.mat,
where the activity is the parent folder name. Classical (DT/RF/MLP-stats), a tiny-CNN
width sweep, and deep CNN + Transformer with int8 convertibility. Attach
imhoangt/ntu-fi-dataset. Run: python gen_ntufi_notebook.py
"""
import json
from pathlib import Path
cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})

md(r"""# Deployability frontier on a third dataset: NTU-Fi_HAR (item #5)

Whole-body HAR in SenseFi format ({train,test}_amp/{activity}/{sample}.mat). Same
pipeline as the main frontier. Attach **imhoangt/ntu-fi-dataset**; GPU T4.
""")

code(r"""import os, re, json, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.io import loadmat
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
warnings.filterwarnings("ignore")
print("TensorFlow", tf.__version__)
OUT = Path("/kaggle/working"); TARGET_T = 64; EPOCHS = 40; SEEDS = 3; PER_CLASS_CAP = 400
""")

code(r"""# ---- explore ----
ROOT = Path("/kaggle/input")
amp_dirs = [p for p in ROOT.rglob("*_amp") if p.is_dir()]
print("amp dirs:", [str(p.relative_to(ROOT)) for p in amp_dirs][:6])
# activities = immediate subfolders of the amp dirs
acts = sorted({c.name for d in amp_dirs for c in d.iterdir() if c.is_dir()})
print("activities:", acts)
""")

code(r"""def biggest_amp(matpath):
    d = loadmat(str(matpath)); keys=[k for k in d if not k.startswith("__")]
    arrs={k:np.asarray(d[k]) for k in keys}
    a = arrs[max(arrs, key=lambda k: arrs[k].size)]
    if np.iscomplexobj(a): a=np.abs(a)
    return a.astype(np.float32)

def load_ntufi(cap=PER_CLASS_CAP):
    amp_dirs=[p for p in ROOT.rglob("*_amp") if p.is_dir()]
    acts=sorted({c.name for d in amp_dirs for c in d.iterdir() if c.is_dir()})
    lmap={a:i for i,a in enumerate(acts)}
    X=[]; y=[]; per={a:0 for a in acts}
    for d in amp_dirs:
        for c in sorted([x for x in d.iterdir() if x.is_dir()]):
            files=sorted(c.glob("*.mat"))
            for f in files:
                if per[c.name] >= cap: break
                try: a=biggest_amp(f)
                except Exception: continue
                a=np.squeeze(a)
                if a.ndim==1: a=a.reshape(1,-1)
                if a.ndim!=2: continue
                # orient (time, features): time is the LONGER axis
                if a.shape[0] < a.shape[1]: a=a.T
                # resample time to TARGET_T
                idx=np.linspace(0,a.shape[0]-1,TARGET_T).astype(int)
                a=a[idx,:]
                X.append(a.astype(np.float32)); y.append(lmap[c.name]); per[c.name]+=1
    X=np.asarray(X,np.float32); y=np.asarray(y,np.int64)
    # z-score per window
    m=X.mean(axis=(1,2),keepdims=True); s=X.std(axis=(1,2),keepdims=True)+1e-8
    X=((X-m)/s).astype(np.float32)
    print("per-class counts:", per)
    return X, y, acts
X, Y, ACTS = load_ntufi(); NCLS=len(ACTS)
print("NTU-Fi X", X.shape, "classes", NCLS, "counts", np.bincount(Y))
""")

code(r"""def stat_features(X):
    mean=X.mean(1); std=X.std(1); mn=X.min(1); mx=X.max(1); rng=mx-mn
    return np.concatenate([mean,std,mn,mx,rng],axis=1).astype(np.float32)
def tiny_cnn(T,F,n,ch):
    i=layers.Input((T,F)); x=layers.Conv1D(ch,7,padding="same",activation="relu")(i)
    x=layers.MaxPool1D(2)(x); x=layers.Conv1D(ch*2,5,padding="same",activation="relu")(x)
    x=layers.GlobalAveragePooling1D()(x); return Model(i,layers.Dense(n)(x))
def tiny_mlp(T,F,n):
    i=layers.Input((T,F)); x=layers.Flatten()(i); x=layers.Dense(128,activation="relu")(x)
    return Model(i,layers.Dense(n)(x))
def deep_cnn(T,F,n):
    i=layers.Input((T,F)); x=layers.Conv1D(64,7,padding="same",activation="relu")(i)
    x=layers.BatchNormalization()(x); x=layers.MaxPool1D(2)(x)
    x=layers.Conv1D(128,5,padding="same",activation="relu")(x); x=layers.BatchNormalization()(x)
    x=layers.GlobalAveragePooling1D()(x); x=layers.Dense(64,activation="relu")(x)
    return Model(i,layers.Dense(n)(x))
def transformer(T,F,n):
    i=layers.Input((T,F)); x=layers.Conv1D(64,5,padding="same")(i)
    a=layers.MultiHeadAttention(num_heads=4,key_dim=16)(x,x); x=layers.LayerNormalization()(x+a)
    f=layers.Dense(128,activation="relu")(x); f=layers.Dense(64)(f); x=layers.LayerNormalization()(f)
    x=layers.GlobalAveragePooling1D()(x); return Model(i,layers.Dense(n)(x))
BUILD_CH=16
def fit_nn(builder,Xtr,ytr,Xte,yte,seed):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    m=builder(int(Xtr.shape[1]),int(Xtr.shape[2]),NCLS) if builder in (tiny_mlp,deep_cnn,transformer) \
      else builder(int(Xtr.shape[1]),int(Xtr.shape[2]),NCLS,BUILD_CH)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(Xtr,ytr,epochs=EPOCHS,batch_size=64,verbose=0)
    return accuracy_score(yte, m.predict(Xte,verbose=0).argmax(-1)), m
def int8_size(model,Xrep):
    def rep():
        for i in range(min(200,len(Xrep))): yield [Xrep[i:i+1].astype(np.float32)]
    c=tf.lite.TFLiteConverter.from_keras_model(model); c.optimizations=[tf.lite.Optimize.DEFAULT]
    c.representative_dataset=rep
    try:
        c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type=tf.int8; c.inference_output_type=tf.int8
        return len(c.convert())/1024.0, True
    except Exception: return None, False
""")

code(r"""tr,te=next(StratifiedShuffleSplit(1,test_size=0.2,random_state=0).split(X,Y))
Xtr,ytr,Xte,yte=X[tr],Y[tr],X[te],Y[te]; rows=[]
Ftr,Fte=stat_features(Xtr),stat_features(Xte)
for name,clf in [("Decision Tree",DecisionTreeClassifier(max_depth=12)),
                 ("Random Forest",RandomForestClassifier(n_estimators=20,max_depth=10,n_jobs=-1)),
                 ("MLP (statistics)",MLPClassifier(hidden_layer_sizes=(64,),max_iter=400))]:
    accs=[]
    for s in range(SEEDS):
        try: clf.set_params(random_state=s)
        except Exception: pass
        clf.fit(Ftr,ytr); accs.append(accuracy_score(yte,clf.predict(Fte)))
    rows.append({"Model":name,"Tier":"classical","Accuracy":round(np.mean(accs)*100,2),"std":round(np.std(accs)*100,2),"int8_kB":"emlearn C","Converts":"n/a"}); print(rows[-1])
for ch,tag in [(8,"Tiny CNN (8 ch)"),(16,"Tiny CNN (16 ch)"),(32,"Tiny CNN (32 ch)")]:
    BUILD_CH=ch; accs=[]; last=None
    for s in range(SEEDS): a,m=fit_nn(tiny_cnn,Xtr,ytr,Xte,yte,s); accs.append(a); last=m
    kb,ok=int8_size(last,Xtr)
    rows.append({"Model":tag,"Tier":"tiny-nn","Accuracy":round(np.mean(accs)*100,2),"std":round(np.std(accs)*100,2),"int8_kB":round(kb,1) if kb else None,"Converts":"yes" if ok else "no"}); print(rows[-1])
a,m=fit_nn(tiny_mlp,Xtr,ytr,Xte,yte,0); kb,ok=int8_size(m,Xtr)
rows.append({"Model":"Tiny MLP (net)","Tier":"tiny-nn","Accuracy":round(a*100,2),"std":0.0,"int8_kB":round(kb,1) if kb else None,"Converts":"yes" if ok else "no"}); print(rows[-1])
for name,b in [("Deep CNN",deep_cnn),("Transformer",transformer)]:
    a,m=fit_nn(b,Xtr,ytr,Xte,yte,0); kb,ok=int8_size(m,Xtr)
    rows.append({"Model":name,"Tier":"deep","Accuracy":round(a*100,2),"std":0.0,"int8_kB":round(kb,1) if kb else None,"Converts":"yes" if ok else "no"}); print(rows[-1])
df=pd.DataFrame(rows); df.to_csv(OUT/"ntufi_frontier_table.csv",index=False)
json.dump({"dataset":"NTU-Fi_HAR","n_classes":int(NCLS),"activities":ACTS,"n_samples":int(len(X)),"rows":rows}, open(OUT/"ntufi_frontier.json","w"), indent=2)
print("\n", df.to_string(index=False)); df
""")

md(r"""## Result
`ntufi_frontier_table.csv` gives the frontier on NTU-Fi_HAR, a clean whole-body HAR
corpus. Being a standard HAR benchmark, accuracy should be high (deep and tiny close),
adding a third saturated data point to the deployability story.
""")

nb={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).parent/"csi_ntufi.ipynb"; out.write_text(json.dumps(nb,indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
