#!/usr/bin/env python3
"""Generates csi_repro.ipynb (reviewer Major #6 + spec #6): reproducibility experiments on
CSI-HAR (subject-independent leave-one-user-out):
  D_repro4  multi-seed per-user LOUO: per held-out user, mean +/- std over 3 seeds.
  D_repro5  temporal-resampling ablation: accuracy vs window length T in {32,64,128,192}.
  A10       paired per-window agreement between the tiny CNN and the deep CNN.
Attach sayakghorai34/csi-har-dataset. GPU T4."""
import json
from pathlib import Path
cells=[]
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(t): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t})

md(r"""# Reproducibility: multi-seed LOUO, resampling ablation, paired agreement (Major #6)

All on CSI-HAR, subject-independent leave-one-user-out. Attach **sayakghorai34/csi-har-dataset**, GPU T4.""")

code(r"""import os, re, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
print("TF", tf.__version__); OUT=Path("/kaggle/working"); EPOCHS=40""")

code(r"""def load_csihar_raw():
    root=None
    for c in Path('/kaggle/input').rglob('CSI-HAR-Dataset'):
        if c.is_dir(): root=c; break
    files=[p for p in root.rglob('*_A.csv') if not p.name.startswith('Annotation')]
    acts=sorted({p.parent.name for p in files}); lm={a:i for i,a in enumerate(acts)}
    seqs=[];y=[];u=[]
    for p in files:
        try: a=np.genfromtxt(str(p),delimiter=',')
        except: continue
        if a.ndim==1 or a.shape[0]<2 or a.shape[1]<2: continue
        seqs.append(a.astype(np.float32)); y.append(lm[p.parent.name])
        m=re.search(r'user_(\d+)_',p.name); u.append(int(m.group(1)) if m else 0)
    return seqs, np.asarray(y), np.asarray(u), acts
def resample_norm(seqs, T):
    X=[]
    for a in seqs:
        idx=np.linspace(0,a.shape[0]-1,T).astype(int); X.append(a[idx,:])
    X=np.asarray(X,np.float32)
    mu=X.mean((1,2),keepdims=True); sd=X.std((1,2),keepdims=True)+1e-8
    return ((X-mu)/sd).astype(np.float32)
def feats(X):
    return np.concatenate([X.mean(1),X.std(1),X.min(1),X.max(1),X.max(1)-X.min(1)],1).astype(np.float32)
SEQS,Y,U,ACTS=load_csihar_raw(); NC=len(ACTS); USERS=sorted(set(U.tolist()))
print("samples",len(SEQS),"classes",NC,"users",USERS)
""")

code(r"""def tiny_cnn(T,F,n,ch):
    i=layers.Input((T,F)); x=layers.Conv1D(ch,7,padding='same',activation='relu')(i)
    x=layers.MaxPool1D(2)(x); x=layers.Conv1D(ch*2,5,padding='same',activation='relu')(x)
    x=layers.GlobalAveragePooling1D()(x); return Model(i,layers.Dense(n)(x))
def deep_cnn(T,F,n):
    i=layers.Input((T,F)); x=layers.Conv1D(64,7,padding='same',activation='relu')(i)
    x=layers.BatchNormalization()(x); x=layers.MaxPool1D(2)(x)
    x=layers.Conv1D(128,5,padding='same',activation='relu')(x); x=layers.BatchNormalization()(x)
    x=layers.GlobalAveragePooling1D()(x); x=layers.Dense(64,activation='relu')(x)
    return Model(i,layers.Dense(n)(x))
def fit_pred(builder,Xtr,ytr,Xte,seed,ch=None):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    m=builder(Xtr.shape[1],Xtr.shape[2],NC,ch) if ch else builder(Xtr.shape[1],Xtr.shape[2],NC)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    m.fit(Xtr,ytr,epochs=EPOCHS,batch_size=64,verbose=0)
    return m.predict(Xte,verbose=0).argmax(-1)
""")

code(r"""# ---- D_repro4: multi-seed per-user LOUO (T=64) ----
X64=resample_norm(SEQS,64)
per_user={}
for u in USERS:
    tr=U!=u; te=U==u; row={}
    for tag,fn in [("TinyCNN16",lambda s: fit_pred(tiny_cnn,X64[tr],Y[tr],X64[te],s,16)),
                   ("TinyCNN32",lambda s: fit_pred(tiny_cnn,X64[tr],Y[tr],X64[te],s,32))]:
        accs=[accuracy_score(Y[te],fn(s))*100 for s in range(3)]
        row[tag]=(round(np.mean(accs),2),round(np.std(accs),2))
    rf=RandomForestClassifier(n_estimators=20,max_depth=10,random_state=0).fit(feats(X64[tr]),Y[tr])
    row["RandomForest"]=(round(accuracy_score(Y[te],rf.predict(feats(X64[te])))*100,2),0.0)
    per_user[f"user_{u}"]=row; print("user",u,row)
json.dump(per_user,open(OUT/"louo_per_user_multiseed.json","w"),indent=2)
""")

code(r"""# ---- D_repro5: temporal-resampling ablation (TinyCNN16, LOUO pooled, per T) ----
abl=[]
for T in [32,64,128,192]:
    Xt=resample_norm(SEQS,T); yy=[];pp=[]
    for u in USERS:
        tr=U!=u; te=U==u
        pp.append(fit_pred(tiny_cnn,Xt[tr],Y[tr],Xt[te],0,16)); yy.append(Y[te])
    acc=accuracy_score(np.concatenate(yy),np.concatenate(pp))*100
    abl.append({"T":T,"TinyCNN16_LOUO_acc":round(acc,2)}); print(abl[-1])
pd.DataFrame(abl).to_csv(OUT/"resampling_ablation.csv",index=False)
""")

code(r"""# ---- A10: paired per-window agreement, TinyCNN16 vs DeepCNN (LOUO pooled, T=64) ----
tp=[];dp=[];yy=[]
for u in USERS:
    tr=U!=u; te=U==u
    tp.append(fit_pred(tiny_cnn,X64[tr],Y[tr],X64[te],0,16))
    dp.append(fit_pred(deep_cnn,X64[tr],Y[tr],X64[te],0))
    yy.append(Y[te])
tp=np.concatenate(tp); dp=np.concatenate(dp); yy=np.concatenate(yy)
agree=float((tp==dp).mean())*100
both_wrong_same=float(((tp!=yy)&(tp==dp)).sum()/max((tp!=yy).sum(),1))*100
paired={"window_level_agreement_pct":round(agree,2),
        "of_tiny_errors_pct_shared_with_deep":round(both_wrong_same,2),
        "n_windows":int(len(yy))}
json.dump(paired,open(OUT/"paired_agreement.json","w"),indent=2)
print(paired)
""")

md(r"""## Results
`louo_per_user_multiseed.json` (D_repro4), `resampling_ablation.csv` (D_repro5), and
`paired_agreement.json` (A10). The paired agreement is the sample-level statistic the
reviewer asked for before claiming the tiny and deep models make the same errors.""")

nb={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).parent/"csi_repro.ipynb"; out.write_text(json.dumps(nb,indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
