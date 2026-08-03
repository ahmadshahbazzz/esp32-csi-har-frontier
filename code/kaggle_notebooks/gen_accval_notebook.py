#!/usr/bin/env python3
"""Generates csi_accval.ipynb (reviewer Major #2 / spec #12): for every deployable
neural model, report FLOAT accuracy, desktop-TFLite INT8 accuracy, and the accuracy loss
from quantization, on both datasets. CSI-HAR uses subject-independent leave-one-user-out
(predictions pooled across folds); UT-HAR uses its fixed split over 3 seeds. This makes
explicit whether each reported accuracy is pre- or post-quantization.
Attach hylanj/wifi-csi-dataset-ut-har + sayakghorai34/csi-har-dataset. GPU T4."""
import json
from pathlib import Path
cells=[]
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(t): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t})

md(r"""# Accuracy validation: float vs int8 (Major #2 / spec #12)

For each deployable neural model we report float accuracy, desktop-TFLite int8 accuracy,
and the quantization loss, on CSI-HAR (leave-one-user-out, pooled) and UT-HAR (fixed
split, 3 seeds). Attach **hylanj/wifi-csi-dataset-ut-har** and
**sayakghorai34/csi-har-dataset**.""")

code(r"""import os, re, glob, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
warnings.filterwarnings("ignore")
print("TF", tf.__version__)
OUT=Path("/kaggle/working"); T=64; EPOCHS=40""")

code(r"""# ---------- loaders ----------
def load_csihar(T=T):
    root=None
    for c in Path('/kaggle/input').rglob('CSI-HAR-Dataset'):
        if c.is_dir(): root=c; break
    files=[p for p in root.rglob('*_A.csv') if not p.name.startswith('Annotation')]
    acts=sorted({p.parent.name for p in files}); lm={a:i for i,a in enumerate(acts)}
    X=[];y=[];u=[]
    for p in files:
        try: a=np.genfromtxt(str(p),delimiter=',')
        except: continue
        if a.ndim==1 or a.shape[0]<2 or a.shape[1]<2: continue
        idx=np.linspace(0,a.shape[0]-1,T).astype(int)
        X.append(a[idx,:].astype(np.float32)); y.append(lm[p.parent.name])
        m=re.search(r'user_(\d+)_',p.name); u.append(int(m.group(1)) if m else 0)
    X=np.asarray(X,np.float32); y=np.asarray(y); u=np.asarray(u)
    mu=X.mean((1,2),keepdims=True); sd=X.std((1,2),keepdims=True)+1e-8
    return ((X-mu)/sd).astype(np.float32), y, u, acts

def _find(name):
    for p in Path('/kaggle/input').rglob(name):
        return p
    return None
def _load(p):
    try:
        with open(p,'rb') as f: return np.asarray(np.load(f,allow_pickle=True))
    except Exception: return np.genfromtxt(str(p),delimiter=',')
def load_uthar(T=T):
    pa={k:_find(f'{k}.csv') for k in ['X_train','y_train','X_test','y_test']}
    ytr=_load(pa['y_train']).astype(np.int64).flatten(); yte=_load(pa['y_test']).astype(np.int64).flatten()
    Xtr=_load(pa['X_train']).astype(np.float32); Xte=_load(pa['X_test']).astype(np.float32)
    def seq(x,n):
        x=np.asarray(x)
        if x.ndim==3 and x.shape[1]==250 and x.shape[2]==90: return x
        if x.ndim==3 and x.shape[1]==90 and x.shape[2]==250: return x.transpose(0,2,1)
        if x.ndim==2 and x.shape[1]==250*90: return x.reshape(-1,250,90)
        return x.reshape(n,250,90)
    Xtr,Xte=seq(Xtr,len(ytr)),seq(Xte,len(yte))
    idx=np.linspace(0,Xtr.shape[1]-1,T).astype(int); Xtr=Xtr[:,idx,:]; Xte=Xte[:,idx,:]
    def norm(x):
        m=x.mean((1,2),keepdims=True); s=x.std((1,2),keepdims=True)+1e-8; return ((x-m)/s).astype(np.float32)
    return norm(Xtr),ytr,norm(Xte),yte
""")

code(r"""# ---------- models ----------
def tiny_cnn(T,F,n,ch):
    i=layers.Input((T,F)); x=layers.Conv1D(ch,7,padding='same',activation='relu')(i)
    x=layers.MaxPool1D(2)(x); x=layers.Conv1D(ch*2,5,padding='same',activation='relu')(x)
    x=layers.GlobalAveragePooling1D()(x); return Model(i,layers.Dense(n)(x))
def tiny_mlp(T,F,n):
    i=layers.Input((T,F)); x=layers.Flatten()(i); x=layers.Dense(128,activation='relu')(x)
    return Model(i,layers.Dense(n)(x))
def deep_cnn(T,F,n):
    i=layers.Input((T,F)); x=layers.Conv1D(64,7,padding='same',activation='relu')(i)
    x=layers.BatchNormalization()(x); x=layers.MaxPool1D(2)(x)
    x=layers.Conv1D(128,5,padding='same',activation='relu')(x); x=layers.BatchNormalization()(x)
    x=layers.GlobalAveragePooling1D()(x); x=layers.Dense(64,activation='relu')(x)
    return Model(i,layers.Dense(n)(x))
def transformer(T,F,n):
    i=layers.Input((T,F)); x=layers.Conv1D(64,5,padding='same')(i)
    a=layers.MultiHeadAttention(num_heads=4,key_dim=16)(x,x); x=layers.LayerNormalization()(x+a)
    f=layers.Dense(128,activation='relu')(x); f=layers.Dense(64)(f); x=layers.LayerNormalization()(f)
    x=layers.GlobalAveragePooling1D()(x); return Model(i,layers.Dense(n)(x))
BUILDERS={'TinyCNN8':lambda T,F,n:tiny_cnn(T,F,n,8),'TinyCNN16':lambda T,F,n:tiny_cnn(T,F,n,16),
          'TinyCNN32':lambda T,F,n:tiny_cnn(T,F,n,32),'TinyMLP':tiny_mlp,'DeepCNN':deep_cnn,'Transformer':transformer}

def train(builder,Xtr,ytr,seed,ncls):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    m=builder(Xtr.shape[1],Xtr.shape[2],ncls)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(Xtr,ytr,epochs=EPOCHS,batch_size=64,verbose=0); return m

def int8_eval(model,Xtr,Xte,yte):
    def rep():
        for i in range(min(300,len(Xtr))): yield [Xtr[i:i+1].astype(np.float32)]
    c=tf.lite.TFLiteConverter.from_keras_model(model); c.optimizations=[tf.lite.Optimize.DEFAULT]
    c.representative_dataset=rep; c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type=tf.int8; c.inference_output_type=tf.int8
    tfl=c.convert()
    it=tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
    ind=it.get_input_details()[0]; outd=it.get_output_details()[0]; s,z=ind['quantization']
    preds=[]
    for x in Xte:
        xq=np.round(x/s+z).clip(-128,127).astype(np.int8)
        it.set_tensor(ind['index'],xq[None,...]); it.invoke()
        preds.append(int(it.get_tensor(outd['index'])[0].argmax()))
    return accuracy_score(yte,preds), len(tfl)/1024.0
""")

code(r"""rows=[]
# ---- CSI-HAR: leave-one-user-out, pooled predictions ----
Xc,yc,uc,acts=load_csihar(); ncls=len(acts)
folds=[(uc!=u,uc==u) for u in sorted(set(uc.tolist()))]
for name,b in BUILDERS.items():
    ft_true=[];ft_pred=[];i8_pred=[]
    for tr,te in folds:
        m=train(b,Xc[tr],yc[tr],0,ncls)
        fp=m.predict(Xc[te],verbose=0).argmax(-1)
        i8a,kb=int8_eval(m,Xc[tr],Xc[te],yc[te])
        # for pooled float/int8 we recompute int8 preds per fold
        ft_true.append(yc[te]); ft_pred.append(fp)
    ft_true=np.concatenate(ft_true); ft_pred=np.concatenate(ft_pred)
    facc=accuracy_score(ft_true,ft_pred)*100
    # int8 pooled: re-run int8 over folds
    i8t=[];i8p=[]
    for tr,te in folds:
        m=train(b,Xc[tr],yc[tr],0,ncls)
        def rep():
            for i in range(min(300,tr.sum())): yield [Xc[tr][i:i+1].astype(np.float32)]
        c=tf.lite.TFLiteConverter.from_keras_model(m); c.optimizations=[tf.lite.Optimize.DEFAULT]
        c.representative_dataset=rep; c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type=tf.int8; c.inference_output_type=tf.int8; tfl=c.convert()
        it=tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
        ind=it.get_input_details()[0]; outd=it.get_output_details()[0]; s,z=ind['quantization']
        for x in Xc[te]:
            xq=np.round(x/s+z).clip(-128,127).astype(np.int8); it.set_tensor(ind['index'],xq[None,...]); it.invoke()
            i8p.append(int(it.get_tensor(outd['index'])[0].argmax()))
        i8t.append(yc[te])
    i8acc=accuracy_score(np.concatenate(i8t),i8p)*100
    rows.append({'dataset':'CSI-HAR','model':name,'float_acc':round(facc,2),'int8_acc':round(i8acc,2),'loss_pts':round(facc-i8acc,2)})
    print(rows[-1])
""")

code(r"""# ---- UT-HAR: fixed split, 3 seeds ----
Xtr_u,ytr_u,Xte_u,yte_u=load_uthar(); ncls_u=int(max(ytr_u.max(),yte_u.max()))+1
for name,b in BUILDERS.items():
    fa=[];ia=[]
    for seed in range(3):
        m=train(b,Xtr_u,ytr_u,seed,ncls_u)
        fa.append(accuracy_score(yte_u,m.predict(Xte_u,verbose=0).argmax(-1))*100)
        i8a,_=int8_eval(m,Xtr_u,Xte_u,yte_u); ia.append(i8a*100)
    rows.append({'dataset':'UT-HAR','model':name,'float_acc':round(np.mean(fa),2),'int8_acc':round(np.mean(ia),2),'loss_pts':round(np.mean(fa)-np.mean(ia),2)})
    print(rows[-1])
df=pd.DataFrame(rows); df.to_csv(OUT/'accval_float_vs_int8.csv',index=False)
json.dump(rows,open(OUT/'accval.json','w'),indent=2)
print('\n',df.to_string(index=False)); df
""")

md(r"""## Result
`accval_float_vs_int8.csv` gives float vs int8 accuracy and the quantization loss per model
and dataset, so every table value in the paper can be labelled pre- or post-quantization.""")

nb={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).parent/"csi_accval.ipynb"; out.write_text(json.dumps(nb,indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
