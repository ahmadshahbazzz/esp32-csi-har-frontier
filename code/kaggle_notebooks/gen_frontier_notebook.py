#!/usr/bin/env python3
"""
Generates the deployability-frontier notebook: classical ML (Decision Tree, Random
Forest), a tiny MLP, and a tiny-CNN width sweep for WiFi CSI HAR, producing the
accuracy-versus-cost frontier that shows what actually fits a bare PSRAM-less classic
ESP32 (about 108 KB usable SRAM).
Run: python gen_frontier_notebook.py  ->  csi_frontier_kaggle.ipynb

Pairs with the 5-architecture benchmark (csi_benchmark_kaggle.ipynb), which already
established the convertibility and memory walls for CNN, GRU, Transformer, KAN, and SSM.

Style: no em dashes anywhere in markdown or comments (author preference).
"""
import json
from pathlib import Path

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": t})

# ---------------------------------------------------------------- CELL 1: title
md(r"""# The Deployability Frontier of WiFi CSI HAR on a PSRAM-less ESP32

**A TinyML deployment-characterization study.** This notebook measures what actually fits
and runs for WiFi CSI Human Activity Recognition (HAR) on the cheapest, most common, and
weakest ESP32: the classic **ESP32-D0WD-V3**, which has no PSRAM and only about **108 KB**
of usable contiguous SRAM.

### What this notebook produces
1. A **deployability frontier table** per dataset: accuracy, model complexity, int8 model
   size, convertibility to TensorFlow Lite Micro, and an indication of on-device fit.
2. Three **publication figures**: accuracy by model and dataset, the accuracy-versus-size
   frontier, and model complexity on a logarithmic scale.
3. Deployable **artifacts**: int8 `.tflite` files for the tiny neural networks and
   `.joblib` models for the classical learners (later converted to C with emlearn).

### Models evaluated here
* **Classical ML:** Decision Tree, Random Forest, and a small MLP on handcrafted
  per-subcarrier statistics. These are the candidates expected to fit comfortably.
* **Tiny neural networks:** a tiny MLP and a tiny-CNN width sweep (8, 16, 32 channels),
  to locate the smallest networks that convert and could fit.

The five standard deep models (CNN, GRU, Transformer, KAN, SSM) are characterized in the
companion benchmark notebook, which establishes two deployment walls: a **convertibility
wall** (GRU and SSM do not export to TensorFlow Lite Micro) and a **memory wall**
(CNN, Transformer, and KAN convert but their int8 tensor arenas exceed about 108 KB).

### Datasets
* **UT-HAR** (`hylanj/wifi-csi-dataset-ut-har`)
* **CSI-HAR-Dataset** (`sayakghorai34/csi-har-dataset`)

Whichever datasets are attached will be processed; missing ones are skipped cleanly.

### How to run
Attach both datasets, set the accelerator to **GPU T4** and **Internet On**, then run all
cells. Outputs are written to `/kaggle/working`.
""")

# ---------------------------------------------------------------- CELL 2: setup
md(r"""## 1. Setup and configuration""")
code(r"""import os, json, time, warnings
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
warnings.filterwarnings("ignore")
print("TensorFlow", tf.__version__)

OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True); (OUT/"tflite").mkdir(exist_ok=True)
SEEDS = 3       # repeats for classical-ML accuracy mean and standard deviation
EPOCHS = 40     # training epochs for the tiny neural networks
TARGET_T = 64   # time window after downsampling, matching the on-device study
print("output dir:", OUT, "| seeds:", SEEDS, "| epochs:", EPOCHS, "| window T:", TARGET_T)
""")

# ---------------------------------------------------------------- CELL 3: UT-HAR loader
md(r"""## 2. Data loaders

### 2.1 UT-HAR
UT-HAR ships as NumPy arrays saved with a `.csv` extension, split into `data/` and
`label/` folders. Each sample is a sequence of length 250 over 90 subcarrier streams.
The loader reshapes to `(N, 250, 90)`, downsamples the time axis to `T` by uniform index
selection, and applies per-sample standardization.""")
code(r"""def _find(name):
    h = list(Path("/kaggle/input").rglob(name)); return sorted(h,key=lambda p:len(str(p)))[0] if h else None
def _load(p):
    try:
        with open(p,"rb") as f: return np.asarray(np.load(f, allow_pickle=True))
    except Exception: return np.genfromtxt(str(p), delimiter=",")

def load_ut_har(T=TARGET_T):
    pa={k:_find(f"{k}.csv") for k in ["X_train","y_train","X_test","y_test"]}
    miss=[k for k,v in pa.items() if v is None]
    if miss: raise FileNotFoundError(f"missing {miss}")
    ytr=_load(pa["y_train"]).astype(np.int64).flatten(); yte=_load(pa["y_test"]).astype(np.int64).flatten()
    Xtr=_load(pa["X_train"]).astype(np.float32); Xte=_load(pa["X_test"]).astype(np.float32)
    def seq(x,n):
        x=np.asarray(x)
        if x.ndim==3 and x.shape[1]==250 and x.shape[2]==90: return x
        if x.ndim==3 and x.shape[1]==90 and x.shape[2]==250: return x.transpose(0,2,1)
        if x.ndim==2 and x.shape[1]==250*90: return x.reshape(-1,250,90)
        return x.reshape(n,250,90)
    Xtr,Xte=seq(Xtr,len(ytr)),seq(Xte,len(yte))
    def dsT(x,T):
        if x.shape[1]==T: return x
        idx=np.linspace(0,x.shape[1]-1,T).astype(int); return x[:,idx,:]
    Xtr,Xte=dsT(Xtr,T),dsT(Xte,T)
    def norm(x):
        m=x.mean(axis=(1,2),keepdims=True); s=x.std(axis=(1,2),keepdims=True)+1e-8
        return ((x-m)/s).astype(np.float32)
    Xtr,Xte=norm(Xtr),norm(Xte)
    print(f"UT-HAR: train {Xtr.shape} test {Xte.shape} (T={T})")
    return Xtr,ytr,Xte,yte
""")

# ---------------------------------------------------------------- CELL 3b: CSI-HAR loader
md(r"""### 2.2 CSI-HAR-Dataset
This dataset stores one CSV per recording under `CSI-HAR-Dataset/<activity>/`, named
`user_<u>_sample_<s>_<activity>_A.csv`. Each data file is a variable-length
`(T_variable, 52)` amplitude matrix; the activity label is the parent folder name, which
correctly handles the folder named "lie down" that contains a space. The `Annotation_*.csv`
files hold per-row labels and are ignored. The set contains seven activities (bend, fall,
lie down, run, sitdown, standup, walk) recorded by three users with twenty samples each.

Because the set has exactly three users, we evaluate CSI-HAR with a **subject-independent,
leave-one-user-out** protocol: each fold trains on two users and tests on the held-out third,
and we report the mean and standard deviation across the three folds. This avoids the subject
leakage of a random split and is the standard rigour for HAR. The loader resamples every
recording to `T` time steps and standardizes per sample; folds are formed by user id.""")
code(r"""import re
def find_csi_har_root():
    for c in Path("/kaggle/input").rglob("CSI-HAR-Dataset"):
        if c.is_dir(): return c
    for c in Path("/kaggle/input").rglob("*"):   # fallback: a dir holding the activity subfolders
        if c.is_dir() and (c/"walk").is_dir() and (c/"run").is_dir(): return c
    return None

def load_csi_har_raw(T=TARGET_T):
    root=find_csi_har_root()
    if root is None: raise FileNotFoundError("CSI-HAR-Dataset not attached")
    files=[p for p in root.rglob("*_A.csv") if not p.name.startswith("Annotation")]
    if not files: raise FileNotFoundError(f"no *_A.csv under {root}")
    acts=sorted({p.parent.name for p in files})
    lmap={a:i for i,a in enumerate(acts)}
    X=[]; y=[]; users=[]
    for p in files:
        try: a=np.genfromtxt(str(p),delimiter=",")
        except Exception: continue
        if a.ndim==1: a=a.reshape(-1,1)
        if a.shape[0]<2 or a.shape[1]<2: continue
        idx=np.linspace(0,a.shape[0]-1,T).astype(int)   # uniform resample of the time axis to T
        X.append(a[idx,:].astype(np.float32)); y.append(lmap[p.parent.name])
        m=re.search(r"user_(\d+)_",p.name); users.append(int(m.group(1)) if m else 0)
    X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.int64); users=np.asarray(users)
    def norm(x):
        m=x.mean(axis=(1,2),keepdims=True); s=x.std(axis=(1,2),keepdims=True)+1e-8
        return ((x-m)/s).astype(np.float32)
    X=norm(X)
    print(f"CSI-HAR: {X.shape} (T={T}, F={X.shape[2]}, classes={acts}, users={sorted(set(users.tolist()))})")
    return X,y,users
""")

# ---------------------------------------------------------------- CELL 4: features
md(r"""## 3. Feature extraction for classical ML
For the classical learners, each recording is summarized by five statistics computed over
time for every subcarrier: mean, standard deviation, minimum, maximum, and range. This
turns a `(T, F)` recording into a compact `F times 5` feature vector that a Decision Tree or
Random Forest can use directly, and that is cheap to compute on a microcontroller.""")
code(r"""def csi_features(X):
    # X: (N, T, F) -> per-subcarrier mean, std, min, max, range over time -> (N, F*5)
    mean=X.mean(1); std=X.std(1); mn=X.min(1); mx=X.max(1); rng=mx-mn
    return np.concatenate([mean,std,mn,mx,rng],axis=1).astype(np.float32)
""")

# ---------------------------------------------------------------- CELL 5: classical ML
md(r"""## 4. Classical models
Three resource-frugal learners are trained on the statistical features. The Decision Tree is
depth-limited so it stays small and emlearn-friendly. Complexity is reported as tree node
count, total forest node count, or MLP parameter count, which is the quantity that maps to
on-device footprint. Each model is saved as a `.joblib` file for later C export.""")
code(r"""def _factory(name, s):
    if name=="DecisionTree": return DecisionTreeClassifier(max_depth=12, random_state=s)
    if name=="RandomForest": return RandomForestClassifier(n_estimators=20, max_depth=10, random_state=s, n_jobs=-1)
    return MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=s)

def run_classical(folds, seeds, tag=""):
    # folds: list of (Xtr,ytr,Xte,yte). Accuracy is aggregated over folds x seeds, so a
    # leave-one-user-out dataset reports across-user variance and a fixed-split dataset
    # reports across-seed variance.
    res={}
    for name in ["DecisionTree","RandomForest","TinyMLP"]:
        accs=[]; last=None; nfeat=0
        for (Xtr,ytr,Xte,yte) in folds:
            Ftr,Fte=csi_features(Xtr),csi_features(Xte); nfeat=Ftr.shape[1]
            for s in range(seeds):
                clf=_factory(name,s); clf.fit(Ftr,ytr)
                accs.append(accuracy_score(yte,clf.predict(Fte))); last=clf
        joblib.dump(last, OUT/f"{name}{tag}.joblib")   # one representative model for emlearn
        if name=="DecisionTree": size=last.tree_.node_count
        elif name=="RandomForest": size=sum(e.tree_.node_count for e in last.estimators_)
        else: size=sum(c.size for c in last.coefs_)+sum(c.size for c in last.intercepts_)
        res[name]=dict(acc_mean=float(np.mean(accs)),acc_std=float(np.std(accs)),
                       complexity=int(size),n_features=int(nfeat),n_runs=len(accs))
        print(f"  {name:13s}: acc {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}  complexity={size}  (n={len(accs)})")
    return res
""")

# ---------------------------------------------------------------- CELL 6: tiny NN
md(r"""## 5. Tiny neural networks and int8 conversion
A tiny MLP and a tiny-CNN width sweep (8, 16, 32 channels) are trained, then each is
converted to a fully integer (int8) TensorFlow Lite model using a representative dataset.
The int8 file size is recorded along with the parameter count and whether conversion
succeeded. The int8 file size is a lower bound on the on-device cost; the runtime tensor
arena is larger and is measured separately on the ESP32.""")
code(r"""def tiny_cnn(T,F,n,ch):
    i=layers.Input((T,F)); x=layers.Conv1D(ch,7,padding="same",activation="relu")(i)
    x=layers.MaxPool1D(2)(x); x=layers.Conv1D(ch*2,5,padding="same",activation="relu")(x)
    x=layers.GlobalAveragePooling1D()(x); o=layers.Dense(n)(x)
    return Model(i,o,name=f"TinyCNN{ch}")
def tiny_mlp_nn(T,F,n):
    i=layers.Input((T,F)); x=layers.Flatten()(i); x=layers.Dense(32,activation="relu")(x); o=layers.Dense(n)(x)
    return Model(i,o,name="TinyMLP_nn")

def train_nn(builder,Xtr,ytr,Xte,yte,n,seed=0):
    # No string "accuracy" metric: a Keras 3 regression in TF 2.19 raises a
    # dtype='string' type-promotion error during fit when a string metric is used.
    # Accuracy is computed below with scikit-learn, so the Keras metric is redundant.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    m=builder(int(Xtr.shape[1]),int(Xtr.shape[2]),int(n))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    Xtr=np.asarray(Xtr,dtype=np.float32); ytr=np.asarray(ytr,dtype=np.int64)
    m.fit(Xtr,ytr,epochs=EPOCHS,batch_size=64,verbose=0)
    acc=float(accuracy_score(yte,m.predict(Xte,verbose=0).argmax(-1)))
    return m,acc,int(m.count_params())

def to_int8(model,Xtr,name):
    def rep():
        for i in range(min(300,len(Xtr))): yield [Xtr[i:i+1].astype(np.float32)]
    c=tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations=[tf.lite.Optimize.DEFAULT]; c.representative_dataset=rep
    c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    try: c.inference_input_type=tf.int8; c.inference_output_type=tf.int8; b=c.convert()
    except Exception: b=c.convert()
    (OUT/"tflite"/f"{name}_int8.tflite").write_bytes(b); return len(b)

def run_tiny(folds, seeds, n, tag=""):
    # Accuracy is aggregated over folds x seeds (>=3 estimates), giving a mean and std for
    # every tiny network. The int8 model is exported once from a representative fit.
    res={}
    builders=[("TinyMLP_nn",tiny_mlp_nn)]+[(f"TinyCNN{ch}",lambda T,F,nn,ch=ch:tiny_cnn(T,F,nn,ch)) for ch in (8,16,32)]
    for name,b in builders:
        try:
            accs=[]; last=None; lastX=None; params=0
            for (Xtr,ytr,Xte,yte) in folds:
                for s in range(seeds):
                    m,acc,params=train_nn(b,Xtr,ytr,Xte,yte,n,seed=s)
                    accs.append(acc); last=m; lastX=Xtr
            try: sz=to_int8(last,lastX,name+tag); conv=True
            except Exception: sz=-1; conv=False
            res[name]=dict(acc_mean=float(np.mean(accs)),acc_std=float(np.std(accs)),params=int(params),
                           int8_kb=round(sz/1024,1) if sz>0 else -1,convert_ok=conv,n_runs=len(accs))
            print(f"  {name:12s}: acc {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}  params {params/1e3:.1f}K  int8 {res[name]['int8_kb']}kB  converts {conv}  (n={len(accs)})")
        except Exception as e:
            res[name]=dict(error=f"{type(e).__name__}: {e}"); print(f"  {name}: FAIL {e}")
    return res
""")

# ---------------------------------------------------------------- CELL 7: run + table
md(r"""## 6. Run all datasets and build the frontier table
Every attached dataset contributes one block of results. The combined table lists accuracy,
complexity, int8 size, convertibility, and an on-device fit indicator. The table and the
full results are written to `/kaggle/working` for the manuscript.""")
code(r"""import pandas as pd
# Build evaluation folds per dataset:
#  - UT-HAR keeps its canonical train/test split (one fold) and varies the seed SEEDS times.
#  - CSI-HAR uses subject-independent leave-one-user-out (one fold per user), seed fixed,
#    so its mean and std are across users (the rigorous HAR protocol).
def folds_ut_har():
    Xtr,ytr,Xte,yte=load_ut_har()
    n=int(max(ytr.max(),yte.max()))+1
    return [(Xtr,ytr,Xte,yte)], SEEDS, n
def folds_csi_har():
    X,y,users=load_csi_har_raw()
    n=int(y.max())+1; folds=[]
    for u in sorted(set(users.tolist())):
        te=users==u; tr=~te
        folds.append((X[tr],y[tr],X[te],y[te]))
    print(f"CSI-HAR leave-one-user-out: {len(folds)} folds")
    return folds, 1, n

DATASETS=[("UT-HAR","_uthar",folds_ut_har),("CSI-HAR","_csihar",folds_csi_har)]
all_results={}; all_rows=[]
for dname,tag,foldfn in DATASETS:
    try:
        folds,seeds,N=foldfn()
    except Exception as e:
        print(f"[skip {dname}] {type(e).__name__}: {e}"); continue
    proto = "leave-one-user-out" if len(folds)>1 else f"fixed split x {seeds} seeds"
    print(f"\n######## {dname} (classes={N}, protocol={proto}) ########")
    print("=== Classical ML ==="); classical=run_classical(folds,seeds,tag=tag)
    print("=== Tiny NN ===");      tiny=run_tiny(folds,seeds,N,tag=tag)
    all_results[dname]={"classical":classical,"tiny":tiny,"protocol":proto}
    for k,v in classical.items():
        all_rows.append({"Dataset":dname,"Model":k,"Type":"classical","Acc(%)":f"{v['acc_mean']*100:.2f} +/- {v['acc_std']*100:.2f}",
                         "Complexity":v["complexity"],"int8(kB)":"n/a (C via emlearn)","Converts":"n/a","Fits ~108KB":"yes (tiny)"})
    for k,v in tiny.items():
        if "error" in v: all_rows.append({"Dataset":dname,"Model":k,"Type":"tiny-nn","Acc(%)":"FAIL"}); continue
        all_rows.append({"Dataset":dname,"Model":k,"Type":"tiny-nn","Acc(%)":f"{v['acc_mean']*100:.2f} +/- {v['acc_std']*100:.2f}",
                         "Complexity":f"{v['params']/1e3:.1f}K params","int8(kB)":v["int8_kb"],
                         "Converts":"yes" if v["convert_ok"] else "no","Fits ~108KB":"measure on ESP32"})

if not all_rows: raise RuntimeError("No datasets loaded. Attach UT-HAR and/or CSI-HAR.")
tbl=pd.DataFrame(all_rows)
print("\nDeployability frontier (classical and tiny NN, all datasets)\n"+"-"*78)
print(tbl.to_string(index=False))
tbl.to_csv(OUT/"frontier_table.csv",index=False)
json.dump(all_results,open(OUT/"frontier_results.json","w"),indent=2,default=str)
print("\nsaved frontier_table.csv, frontier_results.json, per-dataset .joblib and *_int8.tflite")
print("Deep models (CNN, GRU, Transformer, KAN, SSM) come from the benchmark notebook.")
tbl
""")

# ---------------------------------------------------------------- CELL 7b: figures
md(r"""## 7. Figures for the paper
Three figures are produced and saved at 200 dpi:

1. **Accuracy by model and dataset.** How well each model classifies activities.
2. **Accuracy versus int8 size.** The tiny-NN frontier, on a logarithmic size axis.
3. **Model complexity.** Tree node count or network parameter count on a logarithmic axis,
   showing that the classical models are orders of magnitude smaller than the networks.

The int8 file size is a lower bound on on-device cost. The runtime tensor arena, which is
what meets the roughly 108 KB SRAM ceiling, is larger and is measured on the ESP32.""")
code(r"""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

recs=[]
for dname,blk in all_results.items():
    for k,v in blk["classical"].items():
        recs.append(dict(dataset=dname,model=k,kind="classical",acc=v["acc_mean"]*100,
                         complexity=v["complexity"],int8_kb=np.nan))
    for k,v in blk["tiny"].items():
        if "error" in v: continue
        recs.append(dict(dataset=dname,model=k,kind="tiny-nn",acc=v["acc_mean"]*100,
                         complexity=v["params"],int8_kb=(v["int8_kb"] if v["int8_kb"]>0 else np.nan)))
df=pd.DataFrame(recs)
df.to_csv(OUT/"frontier_tidy.csv",index=False)
dsets=list(all_results.keys()); colors=plt.cm.tab10.colors
models=list(dict.fromkeys(df.model))
x=np.arange(len(models)); w=0.8/max(1,len(dsets))

# Figure 1: accuracy by model and dataset
fig,ax=plt.subplots(figsize=(9,4.5))
for i,d in enumerate(dsets):
    sub=df[df.dataset==d].set_index("model").reindex(models)
    ax.bar(x+i*w,sub.acc.values,w,label=d,color=colors[i])
ax.set_xticks(x+w*(len(dsets)-1)/2); ax.set_xticklabels(models,rotation=30,ha="right")
ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0,100)
ax.set_title("Accuracy by model and dataset"); ax.legend(); ax.grid(axis="y",alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"fig_accuracy.png",dpi=200); plt.show()

# Figure 2: accuracy versus int8 size (tiny NNs that converted)
fig,ax=plt.subplots(figsize=(7.5,5))
tn=df[(df.kind=="tiny-nn") & df.int8_kb.notna()]
for i,d in enumerate(dsets):
    sub=tn[tn.dataset==d]
    if len(sub)==0: continue
    ax.scatter(sub.int8_kb,sub.acc,s=90,color=colors[i],label=d,zorder=3)
    for _,r in sub.iterrows():
        ax.annotate(r.model,(r.int8_kb,r.acc),fontsize=8,xytext=(4,4),textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("int8 model size (kB, log axis), a lower bound on on-device cost")
ax.set_ylabel("Test accuracy (%)"); ax.set_title("Accuracy versus int8 size: tiny-NN frontier")
ax.grid(True,which="both",alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"fig_frontier.png",dpi=200); plt.show()

# Figure 3: model complexity on a logarithmic scale
fig,ax=plt.subplots(figsize=(9,4.5))
for i,d in enumerate(dsets):
    sub=df[df.dataset==d].set_index("model").reindex(models)
    ax.bar(x+i*w,sub.complexity.values,w,label=d,color=colors[i])
ax.set_yscale("log"); ax.set_xticks(x+w*(len(dsets)-1)/2); ax.set_xticklabels(models,rotation=30,ha="right")
ax.set_ylabel("Complexity (tree nodes or NN parameters, log axis)")
ax.set_title("Model complexity: classical models are orders of magnitude smaller")
ax.legend(); ax.grid(axis="y",which="both",alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"fig_complexity.png",dpi=200); plt.show()
print("saved fig_accuracy.png, fig_frontier.png, fig_complexity.png, frontier_tidy.csv")
""")

# ---------------------------------------------------------------- CELL 8: conclusion
md(r"""## 8. Outputs and next steps

**Files written to `/kaggle/working`**
* `frontier_table.csv` and `frontier_results.json`: the deployability frontier.
* `frontier_tidy.csv`: long-format results used by the figures.
* `fig_accuracy.png`, `fig_frontier.png`, `fig_complexity.png`: the paper figures.
* `tflite/*_int8.tflite`: int8 models for the tiny neural networks.
* `*.joblib`: classical models for C export.

**On-device steps (classic ESP32)**
1. Convert the `.joblib` trees to C with emlearn.
2. Flash the tiny int8 models that fit, then measure latency, RAM, and energy.
3. Fill the latency and energy columns of the frontier table.

**Combine with the benchmark notebook** to present the full picture: classical models, then
tiny neural networks, then minimal deep models, then full deep models, with the
convertibility and memory walls marked.
""")

nb={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},
    "language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).parent/"csi_frontier_kaggle.ipynb"
out.write_text(json.dumps(nb,indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
