#!/usr/bin/env python3
"""
Generates csi_deeptier_csihar.ipynb: runs the FIVE deep architectures (CNN, BiGRU,
Transformer, Chebyshev-KAN, lightweight SSM) on the CSI-HAR dataset to settle whether
the convertibility and memory walls hold on the second dataset (reviewer request).

For each model it reports subject-independent (leave-one-user-out) accuracy, parameter
count, int8 TFLite convertibility, and int8 size, and saves the int8 model for the
convertible ones so they can be flashed to the classic ESP32 for the arena/fit check.
Pairs with the deployability-frontier notebook (classical + tiny tiers).
Run: python gen_deeptier_csihar_notebook.py
"""
import json
from pathlib import Path
cells=[]
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(t): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t})

md(r"""# Deep tier on CSI-HAR: do the two walls hold on the second dataset?

Runs the five standard deep architectures (CNN, BiGRU, Transformer, Chebyshev-KAN,
lightweight SSM) on **CSI-HAR** ($T=64$, $F=52$), reporting subject-independent
(leave-one-user-out) accuracy, parameter count, int8 convertibility, and int8 size.
Convertible models are saved as int8 TFLite so they can be flashed to the classic ESP32
to test the memory wall on this dataset (CSI-HAR has a smaller feature dimension than
UT-HAR, 52 vs 90, so its arenas are smaller).

Attach **sayakghorai34/csi-har-dataset**. Settings: GPU T4, Internet On.
""")

code(r"""import os, json, warnings
from pathlib import Path
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
print("TensorFlow", tf.__version__)
OUT=Path("/kaggle/working"); (OUT/"tflite").mkdir(parents=True,exist_ok=True)
TARGET_T=64; EPOCHS=40
""")

code(r"""# ----- CSI-HAR loader (subject-independent leave-one-user-out) -----
import re
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
def louo_folds():
    X,y,u=load_csi_har_raw(); n=int(y.max())+1; folds=[]
    for uu in sorted(set(u.tolist())):
        te=u==uu; folds.append((X[~te],y[~te],X[te],y[te]))
    return folds,n
""")

code(r"""# ----- Chebyshev-KAN + diagonal SSM (same as the 5-arch benchmark) -----
class ChebyKAN(layers.Layer):
    def __init__(self,out_dim,degree=3,**kw): super().__init__(**kw); self.out_dim=out_dim; self.degree=degree
    def build(self,shape):
        self.in_dim=int(shape[-1])
        self.coeff=self.add_weight(name="coeff",shape=(self.in_dim*(self.degree+1),self.out_dim),
            initializer=tf.keras.initializers.GlorotUniform())
    def call(self,x):
        x=tf.tanh(x); Ts=[tf.ones_like(x),x]
        for k in range(2,self.degree+1): Ts.append(2.0*x*Ts[-1]-Ts[-2])
        return tf.matmul(tf.concat(Ts,axis=-1),self.coeff)
class DiagSSMCell(layers.Layer):
    def __init__(self,units,**kw): super().__init__(**kw); self.units=units; self.state_size=units
    def build(self,shape):
        d=int(shape[-1])
        self.A=self.add_weight(name="A",shape=(self.units,),initializer=tf.keras.initializers.RandomUniform(-0.99,0.99))
        self.B=self.add_weight(name="B",shape=(d,self.units),initializer="glorot_uniform")
        self.C=self.add_weight(name="C",shape=(self.units,d),initializer="glorot_uniform")
    def call(self,inputs,states):
        h=states[0]*tf.tanh(self.A)+tf.matmul(inputs,self.B); return tf.matmul(h,self.C),[h]

def m_cnn(T,F,n):
    i=layers.Input((T,F)); x=layers.Conv1D(64,7,padding="same",activation="relu")(i)
    x=layers.BatchNormalization()(x); x=layers.MaxPool1D(2)(x)
    x=layers.Conv1D(128,5,padding="same",activation="relu")(x)
    x=layers.BatchNormalization()(x); x=layers.GlobalAveragePooling1D()(x)
    x=layers.Dense(64,activation="relu")(x); return Model(i,layers.Dense(n)(x),name="CNN")
def m_gru(T,F,n):
    i=layers.Input((T,F)); x=layers.Bidirectional(layers.GRU(64))(i)
    x=layers.Dense(64,activation="relu")(x); return Model(i,layers.Dense(n)(x),name="BiGRU")
def m_transformer(T,F,n):
    i=layers.Input((T,F)); x=layers.Conv1D(64,5,padding="same")(i)
    a=layers.MultiHeadAttention(num_heads=4,key_dim=16)(x,x); x=layers.LayerNormalization()(x+a)
    f=layers.Dense(128,activation="relu")(x); f=layers.Dense(64)(f); x=layers.LayerNormalization()(f)
    x=layers.GlobalAveragePooling1D()(x); return Model(i,layers.Dense(n)(x),name="Transformer")
def m_kan(T,F,n):
    i=layers.Input((T,F)); x=layers.Conv1D(64,7,padding="same",activation="relu")(i)
    x=layers.MaxPool1D(2)(x); x=layers.Conv1D(64,5,padding="same",activation="relu")(x)
    x=layers.GlobalAveragePooling1D()(x); x=ChebyKAN(64,degree=3)(x); return Model(i,ChebyKAN(n,degree=3)(x),name="ChebyKAN")
def m_ssm(T,F,n):
    i=layers.Input((T,F)); x=layers.Conv1D(64,5,padding="same",activation="relu")(i)
    x=layers.RNN(DiagSSMCell(64))(x); x=layers.Dense(64,activation="relu")(x); return Model(i,layers.Dense(n)(x),name="SSM")
BUILDERS={"CNN":m_cnn,"BiGRU":m_gru,"Transformer":m_transformer,"ChebyKAN":m_kan,"SSM":m_ssm}
""")

code(r"""# ----- train (no string metric: TF-2.19 Keras-3 raises dtype='string' otherwise) -----
def train_once(builder,Xtr,ytr,Xte,yte,n,seed=0):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    m=builder(int(Xtr.shape[1]),int(Xtr.shape[2]),int(n))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(np.asarray(Xtr,np.float32),np.asarray(ytr,np.int64),epochs=EPOCHS,batch_size=64,verbose=0)
    acc=float(accuracy_score(yte,m.predict(Xte,verbose=0).argmax(-1)))
    return m,acc,int(m.count_params())

def to_int8(model,Xtr,name):
    def rep():
        for i in range(min(300,len(Xtr))): yield [Xtr[i:i+1].astype(np.float32)]
    last=None
    for src in ["keras","savedmodel"]:
        for io in ["int8","float"]:
            try:
                if src=="keras": c=tf.lite.TFLiteConverter.from_keras_model(model)
                else:
                    d=str(OUT/"sm"/name)
                    try: model.export(d)
                    except Exception: tf.saved_model.save(model,d)
                    c=tf.lite.TFLiteConverter.from_saved_model(d)
                c.optimizations=[tf.lite.Optimize.DEFAULT]; c.representative_dataset=rep
                c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.TFLITE_BUILTINS]
                if io=="int8": c.inference_input_type=tf.int8; c.inference_output_type=tf.int8
                blob=c.convert(); (OUT/"tflite"/f"{name}_csihar_int8.tflite").write_bytes(blob)
                return len(blob)
            except Exception as e: last=e
    raise last
""")

code(r"""import pandas as pd
folds,N=louo_folds()
rows=[]; results={}
for name,b in BUILDERS.items():
    accs=[]; last=None; lastX=None; params=0
    for (Xtr,ytr,Xte,yte) in folds:
        m,acc,params=train_once(b,Xtr,ytr,Xte,yte,N); accs.append(acc); last=m; lastX=Xtr
    try: sz=to_int8(last,lastX,name); conv=True; kb=round(sz/1024,1)
    except Exception as e: conv=False; kb=-1
    results[name]=dict(acc_mean=float(np.mean(accs)),acc_std=float(np.std(accs)),params=int(params),converts=conv,int8_kb=kb)
    rows.append({"Model":name,"Acc(%)":f"{np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}","Params(K)":round(params/1e3,1),
                 "Converts":"yes" if conv else "no","int8(kB)":kb if kb>0 else "n/a"})
    print(f"  {name:12s} acc {np.mean(accs)*100:.2f}  params {params/1e3:.1f}K  converts {conv}  int8 {kb}kB")
tbl=pd.DataFrame(rows); print("\nDeep tier on CSI-HAR (subject-independent)\n"+"-"*60); print(tbl.to_string(index=False))
tbl.to_csv(OUT/"deeptier_csihar_table.csv",index=False)
json.dump(results,open(OUT/"deeptier_csihar_results.json","w"),indent=2)
print("\nsaved deeptier_csihar_table.csv, deeptier_csihar_results.json, tflite/*_csihar_int8.tflite")
print("Convertible models -> flash to the classic ESP32 to test the memory wall on CSI-HAR.")
tbl
""")

md(r"""## Next
Download `tflite/*_csihar_int8.tflite` for the models that converted (expected: CNN,
Transformer, Chebyshev-KAN; not BiGRU/SSM, which fail the convertibility wall on any
dataset). Flash each to the classic ESP32 and check whether `AllocateTensors` succeeds
within the ~108 kB budget; this settles the memory wall for CSI-HAR.
""")

nb={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},
    "language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).parent/"csi_deeptier_csihar.ipynb"; out.write_text(json.dumps(nb,indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
