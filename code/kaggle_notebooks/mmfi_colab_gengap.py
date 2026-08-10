# ============================================================================
# MM-Fi 40-subject cross-subject generalization gap  --  run in GOOGLE COLAB
# ----------------------------------------------------------------------------
# WHY: MM-Fi's E01-E04.zip are public Google Drive files under a per-file
# download quota (gdown hits "Too many users have downloaded this file"). Reading
# your OWN Drive files in Colab does NOT hit that quota, so we copy them once.
#
# STEP 0 (in the Drive web UI, one time):
#   Open the MM-Fi shared folder, select E01.zip..E04.zip, "Make a copy" (or drag
#   into My Drive). Put them in a folder, e.g.  MyDrive/mmfi/E01.zip .. E04.zip
#   (You have 1 TB; copying within Drive is server-side and quota-free.)
#
# STEP 1: open https://colab.research.google.com , New notebook, paste the cells
#   below (each block = one cell), Run all. CPU runtime is fine.
# ============================================================================

# --- Cell 1: mount your Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Cell 2: extract ONLY wifi-csi + ground_truth from each env zip ---
import zipfile, os, glob
SRC = "/content/drive/MyDrive/mmfi"      # <-- CHANGE to where you put E01..E04.zip
OUT = "/content/mmfi"; os.makedirs(OUT, exist_ok=True)
zips = sorted(glob.glob(f"{SRC}/E0*.zip"))
assert zips, f"No E0*.zip found in {SRC} - fix SRC path"
for z in zips:
    print("extracting wifi-csi from", z)
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if ("wifi-csi" in n) or ("ground_truth" in n):
                zf.extract(n, OUT)
print("mat files:", len(glob.glob(f"{OUT}/**/wifi-csi/*.mat", recursive=True)))

# --- Cell 3: load per-subcarrier amplitude-stat features (CSIamp is [3,114,10]) ---
import numpy as np, re
from scipy.io import loadmat
from collections import defaultdict
mats = glob.glob(f"{OUT}/**/wifi-csi/*.mat", recursive=True)
def parse(p):
    s = re.search(r'/(S\d+)/', p); a = re.search(r'/(A\d+)/', p)
    return (s.group(1) if s else None, a.group(1) if a else None)
buckets = defaultdict(list)
for p in mats:
    s, a = parse(p)
    if s and a: buckets[(s, a)].append(p)
CAP = 20; rng = np.random.RandomState(0); X = []; y = []; g = []
for (s, a), paths in buckets.items():
    sel = paths if len(paths) <= CAP else list(rng.choice(paths, CAP, replace=False))
    for p in sel:
        try: amp = np.asarray(loadmat(p)['CSIamp'], dtype=np.float32)
        except Exception: continue
        if amp.ndim != 3: continue
        a3 = np.abs(amp).transpose(1, 0, 2).reshape(amp.shape[1], -1)   # (114, 3*10)
        fv = np.nan_to_num(np.concatenate([a3.mean(1), a3.std(1), a3.min(1), a3.max(1)])).astype(np.float32)
        X.append(fv); y.append(a); g.append(s)
X = np.array(X); y = np.array(y); g = np.array(g)
print("features:", X.shape, "| subjects:", len(set(g)), "| classes:", len(set(y)))

# --- Cell 4: random (subject-overlapping) vs subject-independent gen-gap ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
def rf(): return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
pr = rf().fit(Xa, ya).predict(Xb)
rand_acc = accuracy_score(yb, pr) * 100
print(f"RANDOM (subject-overlapping): acc={rand_acc:.1f}  f1={f1_score(yb,pr,average='macro')*100:.1f}")
tri, tei = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups=g))
pg = rf().fit(X[tri], y[tri]).predict(X[tei])
ind_acc = accuracy_score(y[tei], pg) * 100
print(f"SUBJECT-INDEPENDENT:          acc={ind_acc:.1f}  f1={f1_score(y[tei],pg,average='macro')*100:.1f}")
print(f">>> {len(set(g))}-subject gen-gap = {rand_acc - ind_acc:.1f} pts")
# (send me these numbers and I'll fold the 40-subject result into the paper table)
