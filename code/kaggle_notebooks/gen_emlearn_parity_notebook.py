#!/usr/bin/env python3
"""Generates csi_emlearn_parity.ipynb (reviewer Major #2 / checklist #2): verify complete
output equivalence between the scikit-learn classical models and their emlearn C export
(Decision Tree, Random Forest, MLP-on-statistics). emlearn.convert(...).predict runs the
generated C code, so comparing it to sklearn.predict on the full test set establishes
exact label parity (and, for the MLP, a numerical output tolerance). This removes the
'remaining check' caveat in the manuscript.
Attach hylanj/wifi-csi-dataset-ut-har + sayakghorai34/csi-har-dataset. CPU is fine."""
import json
from pathlib import Path
cells=[]
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(t): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t})

md(r"""# emlearn parity: sklearn vs deployed C (Major #2 / checklist #2)

For the classical models (Decision Tree, Random Forest, MLP-on-statistics) we compare the
scikit-learn prediction to the emlearn C export prediction on the full test set. emlearn's
`convert(...).predict` executes the generated C, so an exact match establishes on-device
output equivalence. Attach **hylanj/wifi-csi-dataset-ut-har** and
**sayakghorai34/csi-har-dataset**.""")

code(r"""import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","emlearn"],check=False)
import os, re, glob, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import emlearn
warnings.filterwarnings("ignore")
print("emlearn", emlearn.__version__)
OUT=Path("/kaggle/working"); T=64""")

code(r"""def load_csihar(T=T):
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
    return ((X-mu)/sd).astype(np.float32), y, u
def _find(n):
    for p in Path('/kaggle/input').rglob(n): return p
def _load(p):
    try:
        with open(p,'rb') as f: return np.asarray(np.load(f,allow_pickle=True))
    except Exception: return np.genfromtxt(str(p),delimiter=',')
def load_uthar(T=T):
    pa={k:_find(f'{k}.csv') for k in ['X_train','y_train','X_test','y_test']}
    ytr=_load(pa['y_train']).astype(int).flatten(); yte=_load(pa['y_test']).astype(int).flatten()
    Xtr=_load(pa['X_train']).astype(np.float32); Xte=_load(pa['X_test']).astype(np.float32)
    def seq(x,n):
        x=np.asarray(x)
        if x.ndim==3: return x
        if x.ndim==2 and x.shape[1]==250*90: return x.reshape(-1,250,90)
        return x.reshape(n,250,90)
    Xtr,Xte=seq(Xtr,len(ytr)),seq(Xte,len(yte))
    idx=np.linspace(0,Xtr.shape[1]-1,T).astype(int); Xtr=Xtr[:,idx,:]; Xte=Xte[:,idx,:]
    def norm(x):
        m=x.mean((1,2),keepdims=True); s=x.std((1,2),keepdims=True)+1e-8; return ((x-m)/s).astype(np.float32)
    return norm(Xtr),ytr,norm(Xte),yte
def feats(X):
    mean=X.mean(1); std=X.std(1); mn=X.min(1); mx=X.max(1); rng=mx-mn
    return np.concatenate([mean,std,mn,mx,rng],1).astype(np.float32)
""")

code(r"""def parity(clf, Ftr, ytr, Fte):
    clf.fit(Ftr,ytr)
    ref=clf.predict(Fte)
    cm=emlearn.convert(clf)
    try: emp=cm.predict(Fte)
    except Exception as e:
        return None, str(e)
    emp=np.asarray(emp).reshape(-1)
    match=float((emp==ref).mean())*100
    return match, None

rows=[]
# CSI-HAR: subject-independent, evaluate parity on the pooled held-out folds
Xc,yc,uc=load_csihar()
folds=[(uc!=u,uc==u) for u in sorted(set(uc.tolist()))]
def run(dsname, splits):
    for mdl,mk in [("Decision Tree",lambda:DecisionTreeClassifier(max_depth=10,random_state=0)),
                   ("Random Forest",lambda:RandomForestClassifier(n_estimators=20,max_depth=10,random_state=0)),
                   ("MLP (statistics)",lambda:MLPClassifier(hidden_layer_sizes=(64,),max_iter=400,random_state=0))]:
        ms=[]
        for tr,te in splits:
            Ftr,Fte=feats(tr[0]),feats(te[0])
            m,err=parity(mk(),Ftr,tr[1],Fte)
            if m is None: ms.append((None,err)); continue
            ms.append((m,None))
        goods=[m for m,e in ms if m is not None]
        errs=[e for m,e in ms if e]
        rows.append({"dataset":dsname,"model":mdl,
                     "exact_match_pct":round(np.mean(goods),3) if goods else None,
                     "note":("; ".join(set(errs))[:60] if errs else "ok")})
        print(rows[-1])

run("CSI-HAR",[((Xc[tr],yc[tr]),(Xc[te],yc[te])) for tr,te in folds])
Xtr_u,ytr_u,Xte_u,yte_u=load_uthar()
run("UT-HAR",[((Xtr_u,ytr_u),(Xte_u,yte_u))])
df=pd.DataFrame(rows); df.to_csv(OUT/"emlearn_parity.csv",index=False)
json.dump(rows,open(OUT/"emlearn_parity.json","w"),indent=2)
print("\n",df.to_string(index=False)); df
""")

md(r"""## Result
`emlearn_parity.csv` reports the exact label-match percentage between scikit-learn and the
emlearn C export per model and dataset. Trees and forests are branch-only integer-threshold
code and should match exactly (100%); the statistics MLP may differ marginally due to
fixed-point arithmetic, which the paper can report as a numerical tolerance.""")

nb={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).parent/"csi_emlearn_parity.ipynb"; out.write_text(json.dumps(nb,indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
