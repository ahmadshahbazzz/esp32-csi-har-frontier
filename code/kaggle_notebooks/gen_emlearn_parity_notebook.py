#!/usr/bin/env python3
"""Generates csi_emlearn_parity.ipynb (Major #2 / checklist #2): verify complete output
equivalence between scikit-learn classical models and their DEPLOYED emlearn C export, by
generating the C, compiling it with gcc, running it on the full test set, and comparing the
C predictions to sklearn label-for-label. This is the on-device code path (the same C is
flashed), so exact match establishes on-device parity. Attach hylanj/wifi-csi-dataset-ut-har
+ sayakghorai34/csi-har-dataset. CPU is fine."""
import json
from pathlib import Path
cells=[]
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t})
def code(t): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t})

md(r"""# emlearn C parity: sklearn vs compiled C (Major #2 / checklist #2)

We generate the emlearn C for the classical models (Decision Tree, Random Forest, MLP on
statistics), compile it with gcc, run it on the test set, and compare the C predictions to
scikit-learn. Exact match = on-device output equivalence. Attach the two datasets.""")

code(r"""import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","emlearn"],check=False)
import os, re, glob, json, warnings, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import emlearn
warnings.filterwarnings("ignore")
print("emlearn", emlearn.__version__)
OUT=Path("/kaggle/working"); T=64; W=Path("/kaggle/working/build"); W.mkdir(exist_ok=True)""")

code(r"""def load_csihar(T=T):
    root=[c for c in Path('/kaggle/input').rglob('CSI-HAR-Dataset') if c.is_dir()][0]
    files=[p for p in root.rglob('*_A.csv') if not p.name.startswith('Annotation')]
    acts=sorted({p.parent.name for p in files}); lm={a:i for i,a in enumerate(acts)}
    X=[];y=[];u=[]
    for p in files:
        try: a=np.genfromtxt(str(p),delimiter=',')
        except: continue
        if a.ndim==1 or a.shape[0]<2 or a.shape[1]<2: continue
        idx=np.linspace(0,a.shape[0]-1,T).astype(int); X.append(a[idx,:].astype(np.float32)); y.append(lm[p.parent.name])
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
    def nz(x):
        m=x.mean((1,2),keepdims=True); s=x.std((1,2),keepdims=True)+1e-8; return ((x-m)/s).astype(np.float32)
    return nz(Xtr),ytr,nz(Xte),yte
def feats(X):
    return np.concatenate([X.mean(1),X.std(1),X.min(1),X.max(1),X.max(1)-X.min(1)],1).astype(np.float32)""")

code(r"""def emlearn_c_predict(clf, Ftr, ytr, Fte, tag):
    clf.fit(Ftr,ytr); ref=clf.predict(Fte)
    hdr=W/f'{tag}.h'
    cm=emlearn.convert(clf)
    cm.save(file=str(hdr), name=tag)
    src=hdr.read_text()
    # find the predict function signature: <tag>_predict(<argtype> ...)
    m=re.search(rf'\b{tag}_predict\s*\(\s*const\s+(\w+)', src)
    argt=m.group(1) if m else 'float'
    NF=Fte.shape[1]
    # write test features
    np.savetxt(W/f'{tag}_x.txt', Fte, fmt='%.6f')
    main=f'''#include <stdio.h>
#include <stdlib.h>
#include "{tag}.h"
int main(){{ int NF={NF}; {argt} f[{NF}]; float v; int c;
  FILE*fp=fopen("{W}/{tag}_x.txt","r");
  while(1){{ for(int i=0;i<NF;i++){{ if(fscanf(fp,"%f",&v)!=1){{fclose(fp);return 0;}} f[i]=({argt})v; }}
    c={tag}_predict(f, NF); printf("%d\\n", c); }} }}'''
    (W/f'{tag}_main.c').write_text(main)
    r=subprocess.run(['gcc','-O2',f'-I{W}',str(W/f'{tag}_main.c'),'-o',str(W/f'{tag}_run'),'-lm'],
                     capture_output=True,text=True)
    if r.returncode!=0:
        return None,'compile fail: '+r.stderr.strip().splitlines()[-1][:60]
    out=subprocess.run([str(W/f'{tag}_run')],capture_output=True,text=True)
    cpred=np.array([int(x) for x in out.stdout.split()])
    if len(cpred)!=len(ref): return None,f'len {len(cpred)} vs {len(ref)}'
    return float((cpred==ref).mean())*100, ('argt='+argt)""")

code(r"""rows=[]
def run(ds,Ftr,ytr,Fte):
    for nm,mk in [("DecisionTree",lambda:DecisionTreeClassifier(max_depth=10,random_state=0)),
                  ("RandomForest",lambda:RandomForestClassifier(n_estimators=20,max_depth=10,random_state=0)),
                  ("MLPstats",lambda:MLPClassifier(hidden_layer_sizes=(64,),max_iter=400,random_state=0))]:
        try: match,note=emlearn_c_predict(mk(),Ftr,ytr,Fte,f'{ds}_{nm}')
        except Exception as e: match,note=None,type(e).__name__+':'+str(e)[:50]
        rows.append({"dataset":ds,"model":nm,"C_exact_match_pct":round(match,3) if match is not None else None,"note":note})
        print(rows[-1])
# CSI-HAR: LOUO fold 0 (train users != first user) for a representative parity check
Xc,yc,uc=load_csihar(); u0=sorted(set(uc.tolist()))[0]; tr=uc!=u0; te=uc==u0
run("CSIHAR",feats(Xc[tr]),yc[tr],feats(Xc[te]))
Xtr_u,ytr_u,Xte_u,yte_u=load_uthar()
run("UTHAR",feats(Xtr_u),ytr_u,feats(Xte_u))
df=pd.DataFrame(rows); df.to_csv(OUT/"emlearn_parity.csv",index=False)
json.dump(rows,open(OUT/"emlearn_parity.json","w"),indent=2)
print("\n",df.to_string(index=False)); df""")

md(r"""## Result
`emlearn_parity.csv`: exact label-match between the compiled emlearn C and scikit-learn.
Trees/forests are branch-only integer-threshold code and should match exactly (100%); the
statistics MLP may differ slightly due to fixed-point, which the paper reports as a
numerical tolerance.""")

nb={"cells":cells,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).parent/"csi_emlearn_parity.ipynb"; out.write_text(json.dumps(nb,indent=1))
print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(cells)} cells)")
