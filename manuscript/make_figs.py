#!/usr/bin/env python3
"""Regenerates the result figures (Figs 4-9) with larger fonts and consistent
publication styling, from the released result CSVs (reviewer spec #10). Overwrites
fig_accuracy/frontier/complexity/memory/gengap/personalization/cm_*.png in place."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, csv, os
from pathlib import Path

R = Path("/home/ahmad/Downloads/Crypto_Project/frontier_results")
OUT = Path(".")
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "legend.fontsize": 13.5,
    "font.family": "sans-serif", "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})
UT, CSI = "#1f77b4", "#d62728"

def rows(f):
    with open(R/f) as fh: return list(csv.DictReader(fh))

# ---- frontier_tidy: dataset,model,kind,acc,complexity,int8_kb ----
ft = rows("frontier_tidy.csv")
def sub(ds, kinds): return [r for r in ft if r["dataset"]==ds and r["kind"] in kinds]

# nice model labels
LAB={"DecisionTree":"DT","RandomForest":"RF","MLP_stats":"MLP-stat","MLPstats":"MLP-stat",
     "TinyMLP_nn":"T-MLP","TinyMLP":"T-MLP","TinyCNN8":"CNN8","TinyCNN16":"CNN16","TinyCNN32":"CNN32"}
def lab(m): return LAB.get(m, m)

# --- fig_accuracy: grouped bars, classical+tiny, UT vs CSI ---
models=[r["model"] for r in sub("UT-HAR",{"classical","tiny-nn"})]
acc_ut={r["model"]:float(r["acc"]) for r in sub("UT-HAR",{"classical","tiny-nn"})}
acc_csi={r["model"]:float(r["acc"]) for r in sub("CSI-HAR",{"classical","tiny-nn"})}
x=np.arange(len(models)); w=0.38
fig,ax=plt.subplots(figsize=(8.4,4.6))
ax.bar(x-w/2,[acc_ut.get(m,0) for m in models],w,label="UT-HAR",color=UT)
ax.bar(x+w/2,[acc_csi.get(m,0) for m in models],w,label="CSI-HAR (LOUO)",color=CSI)
ax.set_xticks(x); ax.set_xticklabels([lab(m) for m in models],rotation=30,ha="right")
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0,105); ax.legend()
ax.set_title("Classical and tiny tiers: accuracy by dataset")
fig.savefig(OUT/"fig_accuracy.png"); plt.close(fig)

# --- fig_frontier: accuracy vs int8 size (tiny nets), log-x ---
fig,ax=plt.subplots(figsize=(7.6,4.8))
for ds,c in (("UT-HAR",UT),("CSI-HAR",CSI)):
    pts=[(float(r["int8_kb"]),float(r["acc"]),lab(r["model"])) for r in sub(ds,{"tiny-nn"}) if r["int8_kb"]]
    if not pts: continue
    xs,ys,ls=zip(*pts)
    ax.scatter(xs,ys,s=90,color=c,label=ds,zorder=3)
    for xx,yy,ll in pts: ax.annotate(ll,(xx,yy),textcoords="offset points",xytext=(6,4),fontsize=12)
ax.set_xscale("log"); ax.set_xlabel("int8 model size (kB, log scale)")
ax.set_ylabel("Accuracy (%)"); ax.legend(); ax.set_title("Accuracy vs int8 size (tiny networks)")
fig.savefig(OUT/"fig_frontier.png"); plt.close(fig)

# --- fig_complexity: complexity (log) per model, UT tier ---
comp=[(lab(r["model"]),float(r["complexity"])) for r in sub("UT-HAR",{"classical","tiny-nn"}) if r["complexity"]]
fig,ax=plt.subplots(figsize=(8.4,4.6))
ax.bar([m for m,_ in comp],[c for _,c in comp],color="#6a51a3")
ax.set_yscale("log"); ax.set_ylabel("Complexity (nodes or params, log)")
ax.set_xticklabels([m for m,_ in comp],rotation=30,ha="right")
ax.set_title("Model complexity")
fig.savefig(OUT/"fig_complexity.png"); plt.close(fig)

# --- fig_memory: measured arena vs ~150 kB budget ---
sm=rows("ondevice/summary_240.csv")  # model,latency_ms,latency_std_ms,arena_kB,energy_mJ
def clean(m): return m.replace("_"," ").replace("uthar","UT").replace("csihar","CSI")
mm=[(clean(r["model"]),float(r["arena_kB"])) for r in sm]
mm.sort(key=lambda t:t[1])
fig,ax=plt.subplots(figsize=(8.8,5.0))
ax.barh([m for m,_ in mm],[a for _,a in mm],color="#2f7d32")
ax.axvline(150,ls="--",color="#b03030",lw=2); ax.text(150,-0.6,"~150 kB budget",color="#b03030",fontsize=12,ha="center")
ax.set_xlabel("Peak tensor arena (kB)"); ax.set_title("Measured on-device arena vs SRAM budget")
fig.savefig(OUT/"fig_memory.png"); plt.close(fig)

# --- fig_gengap: random vs LOUO bars + gap ---
gg=rows("gengap_table.csv")
gm=[r["Model"] for r in gg]; rnd=[float(r["random_mean"])*100 for r in gg]; lo=[float(r["louo_mean"])*100 for r in gg]
gap=[float(r["gap_pts"]) for r in gg]
x=np.arange(len(gm)); w=0.38
fig,ax=plt.subplots(figsize=(9.0,4.8))
ax.bar(x-w/2,rnd,w,label="Random (subject-overlapping)",color="#9ecae1")
ax.bar(x+w/2,lo,w,label="LOUO (subject-independent)",color="#08519c")
for i,g in enumerate(gap): ax.text(x[i],max(rnd[i],lo[i])+1.5,f"-{g:.0f}",color="#d62728",ha="center",fontsize=12,weight="bold")
ax.set_xticks(x); ax.set_xticklabels([lab(m) for m in gm],rotation=30,ha="right")
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0,105); ax.legend(loc="lower left")
ax.set_title("Cross-subject gap on CSI-HAR (3 users, exploratory)")
fig.savefig(OUT/"fig_gengap.png"); plt.close(fig)

# --- fig_personalization: acc vs k, TinyCNN32 + RF, errorbars ---
pp=rows("personalization/personalization_table.csv")
k=[int(r["k"]) for r in pp]
fig,ax=plt.subplots(figsize=(7.6,4.8))
ax.errorbar(k,[float(r["TinyCNN32"]) for r in pp],yerr=[float(r["TinyCNN32_std"]) for r in pp],
            marker="o",capsize=4,lw=2,color=UT,label="Tiny CNN (32 ch)")
ax.errorbar(k,[float(r["RandomForest"]) for r in pp],yerr=[float(r["RandomForest_std"]) for r in pp],
            marker="s",capsize=4,lw=2,color="#2ca02c",label="Random Forest")
ax.axhline(100/7,ls=":",color="#888"); ax.text(max(k),100/7+2,"chance",ha="right",fontsize=12,color="#666")
ax.set_xlabel("Calibration samples from held-out user (k)"); ax.set_ylabel("Accuracy on held-out user (%)")
ax.legend(); ax.set_title("Per-user calibration recovers accuracy (CSI-HAR)")
fig.savefig(OUT/"fig_personalization.png"); plt.close(fig)

# --- fig_cm_*: confusion matrices, larger fonts ---
for name,fn in (("TinyCNN32","cm_TinyCNN32.csv"),("RandomForest","cm_RandomForest.csv"),("DeepCNN","cm_DeepCNN.csv")):
    rr=rows("metrics/"+fn); acts=list(rr[0].keys())[1:]
    M=np.array([[int(float(r[a])) for a in acts] for r in rr])
    fig,ax=plt.subplots(figsize=(5.6,5.0))
    im=ax.imshow(M,cmap="Blues")
    ax.set_xticks(range(len(acts))); ax.set_xticklabels(acts,rotation=45,ha="right",fontsize=12)
    ax.set_yticks(range(len(acts))); ax.set_yticklabels(acts,fontsize=12)
    thr=M.max()/2
    for i in range(len(acts)):
        for j in range(len(acts)):
            ax.text(j,i,M[i,j],ha="center",va="center",fontsize=12,
                    color="white" if M[i,j]>thr else "#111")
    ax.set_title(f"{name}",fontsize=15); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.grid(False)
    fig.savefig(OUT/f"fig_cm_{name}.png"); plt.close(fig)

print("regenerated:", *[p.name for p in OUT.glob("fig_*.png")])
