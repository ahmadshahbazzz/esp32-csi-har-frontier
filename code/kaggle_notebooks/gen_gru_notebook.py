#!/usr/bin/env python3
"""Generates csi_gru_export.ipynb (reviewer Major #3 / spec, "attempt at least one portable
or unrolled recurrent implementation"). Tests whether a recurrent network can be exported to
the int8 TensorFlow Lite for Microcontrollers builtin-operator set when it is NOT traced
through the fused GPU kernel: (a) standard Keras GRU (fused), (b) Keras GRU(unroll=True),
(c) a manually-unrolled GRU cell over primitive ops. For each we report whether int8
conversion succeeds with builtins only (no Select-TF-ops) and, if so, the model size. This
tests the claim that non-deployability is a property of the export route, not the
architecture. Attach sayakghorai34/csi-har-dataset. CPU is fine."""
import json
from pathlib import Path
cells=[]
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(t): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t})

md(r"""# Recurrent export attempt: fused vs unrolled (Major #3)

Does a recurrent net deploy to the int8 TFLM builtin set if it is not traced through the
fused GPU kernel? We compare a standard GRU, `GRU(unroll=True)`, and a manually-unrolled
GRU cell. Attach **sayakghorai34/csi-har-dataset**.""")

code(r"""import os, re, json, warnings
from pathlib import Path
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
warnings.filterwarnings("ignore")
print("TF", tf.__version__); OUT=Path("/kaggle/working"); T=64""")

code(r"""def load_csihar(T=T):
    root=None
    for c in Path('/kaggle/input').rglob('CSI-HAR-Dataset'):
        if c.is_dir(): root=c; break
    files=[p for p in root.rglob('*_A.csv') if not p.name.startswith('Annotation')]
    acts=sorted({p.parent.name for p in files}); lm={a:i for i,a in enumerate(acts)}
    X=[];y=[]
    for p in files:
        try: a=np.genfromtxt(str(p),delimiter=',')
        except: continue
        if a.ndim==1 or a.shape[0]<2 or a.shape[1]<2: continue
        idx=np.linspace(0,a.shape[0]-1,T).astype(int); X.append(a[idx,:].astype(np.float32)); y.append(lm[p.parent.name])
    X=np.asarray(X,np.float32); y=np.asarray(y)
    mu=X.mean((1,2),keepdims=True); sd=X.std((1,2),keepdims=True)+1e-8
    return ((X-mu)/sd).astype(np.float32), y, len(acts)
X,Y,NC=load_csihar()
tr,te=next(StratifiedShuffleSplit(1,test_size=0.2,random_state=0).split(X,Y))
Xtr,ytr,Xte,yte=X[tr],Y[tr],X[te],Y[te]; F=X.shape[2]
print("X",X.shape,"classes",NC)""")

code(r"""def gru_fused(T,F,n):
    i=layers.Input((T,F)); x=layers.GRU(32)(i); return Model(i,layers.Dense(n)(x))
def gru_unroll(T,F,n):
    i=layers.Input((T,F)); x=layers.GRU(32,unroll=True)(i); return Model(i,layers.Dense(n)(x))
def gru_manual(T,F,n):
    # manual GRU cell unrolled over time using only dense/elementwise ops
    i=layers.Input((T,F)); H=32
    xz=layers.Dense(H); xr=layers.Dense(H); xh=layers.Dense(H)
    hz=layers.Dense(H,use_bias=False); hr=layers.Dense(H,use_bias=False); hh=layers.Dense(H,use_bias=False)
    h=tf.zeros_like(xz(i[:,0,:]))
    for t in range(T):
        xt=i[:,t,:]
        z=tf.sigmoid(xz(xt)+hz(h)); r=tf.sigmoid(xr(xt)+hr(h))
        hh_=tf.tanh(xh(xt)+hh(r*h)); h=(1-z)*h+z*hh_
    return Model(i,layers.Dense(n)(h))
BUILDERS={"GRU_fused":gru_fused,"GRU_unroll":gru_unroll,"GRU_manual_unrolled":gru_manual}

def train(b):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(0)
    m=b(T,F,NC); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(Xtr,ytr,epochs=30,batch_size=64,verbose=0); return m

def try_int8(m):
    def rep():
        for i in range(min(200,len(Xtr))): yield [Xtr[i:i+1].astype(np.float32)]
    c=tf.lite.TFLiteConverter.from_keras_model(m); c.optimizations=[tf.lite.Optimize.DEFAULT]
    c.representative_dataset=rep
    try:  # builtins only, full int8 -> deployable to TFLM
        c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type=tf.int8; c.inference_output_type=tf.int8
        tfl=c.convert(); return "builtins-int8 OK", round(len(tfl)/1024.0,1)
    except Exception as e1:
        return "needs Select-TF-ops: "+type(e1).__name__, None
""")

code(r"""rows=[]
for name,b in BUILDERS.items():
    try:
        m=train(b); acc=accuracy_score(yte,m.predict(Xte,verbose=0).argmax(-1))*100
        outcome,kb=try_int8(m)
    except Exception as e:
        acc=None; outcome="build/train failed: "+type(e).__name__; kb=None
    rows.append({"variant":name,"float_acc":round(acc,2) if acc else None,"int8_export":outcome,"int8_kB":kb})
    print(rows[-1])
import pandas as pd
pd.DataFrame(rows).to_csv(OUT/"gru_export.csv",index=False)
json.dump(rows,open(OUT/"gru_export.json","w"),indent=2)
print(pd.DataFrame(rows).to_string(index=False))""")

md(r"""## Result
`gru_export.csv`: if `GRU(unroll=True)` or the manually-unrolled cell converts with builtins
only while the fused GRU does not, this confirms the recurrent non-deployability is a
property of the export route (the fused `CudnnRNNV3` trace), not the architecture, exactly
as the manuscript now states.""")

nb={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).parent/"csi_gru_export.ipynb"; out.write_text(json.dumps(nb,indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
