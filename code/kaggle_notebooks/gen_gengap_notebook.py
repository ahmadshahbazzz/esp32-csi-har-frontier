#!/usr/bin/env python3
"""
Generates csi_gengap.ipynb: quantifies the cross-subject GENERALIZATION GAP on CSI-HAR by
evaluating the SAME models and pipeline under two protocols only differing in the split:
  (a) random stratified 80/20 split, 3 seeds  (the subject-LEAKING protocol many papers use)
  (b) subject-independent leave-one-user-out, 3 folds  (the honest protocol)
Reports per-model accuracy under each and the gap. This is the evidence for the paper's
reframed thesis: on the cheapest WiFi MCU deployment is easy; generalization is the wall.
Reuses the exact CSI-HAR loader / features / models from gen_frontier_notebook.py.
Attach sayakghorai34/csi-har-dataset. Run: python gen_gengap_notebook.py
"""
import json
from pathlib import Path
cells=[]
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(t): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t})

md(r"""# The cross-subject generalization gap on CSI-HAR

Same models, same features, same training. The only thing that changes is how the data is
split into train and test:

* **Random split**: stratified 80/20, 3 seeds. Samples from every user appear in both train
  and test, so the model can memorize per-user channel signatures. This is the protocol many
  CSI-HAR papers report.
* **Leave-one-user-out (LOUO)**: 3 folds, train on two users, test on the held-out third.
  No subject appears in both train and test. This is what a deployed device faces on a new
  person.

The difference between the two accuracies is the generalization gap. Attach
**sayakghorai34/csi-har-dataset**; GPU T4, Internet On.
""")

code(r"""import os, re, json, warnings
from pathlib import Path
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")
print("TensorFlow", tf.__version__)
OUT=Path("/kaggle/working"); TARGET_T=64; EPOCHS=60
""")

code(r"""# ---- CSI-HAR loader (identical to the frontier notebook) ----
def find_csi_har_root():
    for c in Path("/kaggle/input").rglob("CSI-HAR-Dataset"):
        if c.is_dir(): return c
    for c in Path("/kaggle/input").rglob("*"):
        if c.is_dir() and (c/"walk").is_dir() and (c/"run").is_dir(): return c
    return None
def load_csi_har_raw(T=TARGET_T):
    root=find_csi_har_root()
    if root is None: raise FileNotFoundError("CSI-HAR-Dataset not attached")
    files=[p for p in root.rglob("*_A.csv") if not p.name.startswith("Annotation")]
    acts=sorted({p.parent.name for p in files}); lmap={a:i for i,a in enumerate(acts)}
    X=[];y=[];users=[]
    for p in files:
        try: a=np.genfromtxt(str(p),delimiter=",")
        except Exception: continue
        if a.ndim==1: a=a.reshape(-1,1)
        if a.shape[0]<2 or a.shape[1]<2: continue
        idx=np.linspace(0,a.shape[0]-1,T).astype(int)
        X.append(a[idx,:].astype(np.float32)); y.append(lmap[p.parent.name])
        m=re.search(r"user_(\d+)_",p.name); users.append(int(m.group(1)) if m else 0)
    X=np.asarray(X,np.float32); y=np.asarray(y,np.int64); users=np.asarray(users)
    m=X.mean(axis=(1,2),keepdims=True); s=X.std(axis=(1,2),keepdims=True)+1e-8; X=((X-m)/s).astype(np.float32)
    print(f"CSI-HAR {X.shape} F={X.shape[2]} classes={acts} users={sorted(set(users.tolist()))}")
    return X,y,users
def csi_features(X):
    mean=X.mean(1); std=X.std(1); mn=X.min(1); mx=X.max(1); rng=mx-mn
    return np.concatenate([mean,std,mn,mx,rng],axis=1).astype(np.float32)
""")

code(r"""# ---- the two split protocols, each producing a list of (Xtr,ytr,Xte,yte) ----
X,Y,U=load_csi_har_raw(); NCLS=int(Y.max())+1
def folds_random(seeds=3):
    out=[]
    for s in range(seeds):
        sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=s)
        tr,te=next(sss.split(X,Y)); out.append((X[tr],Y[tr],X[te],Y[te]))
    return out
def folds_louo():
    out=[]
    for uu in sorted(set(U.tolist())):
        te=U==uu; out.append((X[~te],Y[~te],X[te],Y[te]))
    return out
PROTOCOLS={"random":folds_random(),"louo":folds_louo()}
for k,f in PROTOCOLS.items(): print(k, "folds:", len(f), "test sizes:", [len(t[3]) for t in f])
""")

code(r"""# ---- models (identical to the frontier notebook) ----
def _factory(name,s):
    if name=="DecisionTree": return DecisionTreeClassifier(max_depth=12,random_state=s)
    if name=="RandomForest": return RandomForestClassifier(n_estimators=20,max_depth=10,random_state=s,n_jobs=-1)
    return MLPClassifier(hidden_layer_sizes=(32,),max_iter=300,random_state=s)
def tiny_cnn(T,F,n,ch):
    i=layers.Input((T,F)); x=layers.Conv1D(ch,7,padding="same",activation="relu")(i)
    x=layers.MaxPool1D(2)(x); x=layers.Conv1D(ch*2,5,padding="same",activation="relu")(x)
    x=layers.GlobalAveragePooling1D()(x); return Model(i,layers.Dense(n)(x),name=f"TinyCNN{ch}")
def tiny_mlp_nn(T,F,n):
    i=layers.Input((T,F)); x=layers.Flatten()(i); x=layers.Dense(32,activation="relu")(x); return Model(i,layers.Dense(n)(x),name="TinyMLP_nn")
def train_nn(builder,Xtr,ytr,Xte,yte,seed=0):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    m=builder(int(Xtr.shape[1]),int(Xtr.shape[2]),NCLS)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(Xtr,ytr,epochs=EPOCHS,batch_size=64,verbose=0)
    return float(accuracy_score(yte,m.predict(Xte,verbose=0).argmax(-1)))
CLASSICAL=["DecisionTree","RandomForest","MLP_stats"]
NEURAL={"TinyCNN16":lambda T,F,n:tiny_cnn(T,F,n,16),"TinyCNN32":lambda T,F,n:tiny_cnn(T,F,n,32),
        "TinyMLP_nn":tiny_mlp_nn}
""")

code(r"""# ---- run every model under both protocols ----
import pandas as pd
def eval_classical(name,folds):
    accs=[]
    for (Xtr,ytr,Xte,yte) in folds:
        Ftr,Fte=csi_features(Xtr),csi_features(Xte)
        seeds=3 if len(folds)<=3 else 1
        for s in range(seeds):
            nm="MLP" if name=="MLP_stats" else name
            clf=_factory(nm,s) if name!="MLP_stats" else _factory("MLP",s)
            clf.fit(Ftr,ytr); accs.append(accuracy_score(yte,clf.predict(Fte)))
    return float(np.mean(accs)),float(np.std(accs))
def eval_neural(builder,folds):
    accs=[train_nn(builder,*f) for f in folds]
    return float(np.mean(accs)),float(np.std(accs))

rows=[]
for name in CLASSICAL:
    r=eval_classical(name,PROTOCOLS["random"]); l=eval_classical(name,PROTOCOLS["louo"])
    rows.append((name,r[0],r[1],l[0],l[1]))
    print(f"{name:12s} random {r[0]*100:5.1f}  louo {l[0]*100:5.1f}  gap {(r[0]-l[0])*100:5.1f}")
for name,b in NEURAL.items():
    r=eval_neural(b,PROTOCOLS["random"]); l=eval_neural(b,PROTOCOLS["louo"])
    rows.append((name,r[0],r[1],l[0],l[1]))
    print(f"{name:12s} random {r[0]*100:5.1f}  louo {l[0]*100:5.1f}  gap {(r[0]-l[0])*100:5.1f}")

df=pd.DataFrame(rows,columns=["Model","random_mean","random_std","louo_mean","louo_std"])
df["gap_pts"]=(df["random_mean"]-df["louo_mean"])*100
df.to_csv(OUT/"gengap_table.csv",index=False)
json.dump(df.to_dict(orient="records"),open(OUT/"gengap.json","w"),indent=2)
print("\nGENERALIZATION GAP (CSI-HAR)\n"+"-"*52)
print(df.assign(**{c:(df[c]*100).round(1) for c in ["random_mean","louo_mean"]})[["Model","random_mean","louo_mean","gap_pts"]].to_string(index=False))
print(f"\nMean gap across models: {df['gap_pts'].mean():.1f} points")
print("saved gengap_table.csv, gengap.json")
df
""")

md(r"""## Reading the result
A large positive gap means the random-split accuracy is inflated by subject leakage and does
not reflect performance on a new person. The honest (LOUO) accuracy is the one a deployed
device achieves. This is the binding constraint on the cheapest WiFi MCU: not memory or
compute (the deployability frontier shows even deep models fit), but cross-subject
generalization.
""")

nb={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},
    "language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).parent/"csi_gengap.ipynb"; out.write_text(json.dumps(nb,indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
