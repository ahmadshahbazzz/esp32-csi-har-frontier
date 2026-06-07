#!/usr/bin/env python3
"""
Convert the classical-ML models trained on Kaggle (.joblib) into self-contained C headers
for the classic ESP32, using emlearn. Produces one <Model>.h per input plus a size report
that feeds the deployability-frontier table.

Usage:
    python export_emlearn.py <dir-with-joblibs> [out-dir]

Default <dir-with-joblibs> is ./frontier_artifacts, default out-dir is ./emlearn_out.
Run AFTER downloading the Kaggle outputs (DecisionTree*.joblib, RandomForest*.joblib, ...).
No em dashes in output text (author preference).
"""
import sys, glob, json
from pathlib import Path
import joblib
import emlearn

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("frontier_artifacts")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("emlearn_out")
out.mkdir(parents=True, exist_ok=True)

jobs = sorted(glob.glob(str(src / "*.joblib")))
if not jobs:
    print(f"No .joblib files in {src}. Download the Kaggle outputs there first.")
    sys.exit(1)

report = []
for jf in jobs:
    name = Path(jf).stem                      # e.g. DecisionTree_uthar
    cname = name.replace("-", "_")
    try:
        est = joblib.load(jf)
        cmodel = emlearn.convert(est, method="inline")
        hdr = out / f"{cname}.h"
        cmodel.save(file=str(hdr), name=cname)
        c_bytes = hdr.stat().st_size
        rec = {"model": name, "header": hdr.name, "c_source_bytes": c_bytes,
               "estimator": type(est).__name__}
        print(f"OK  {name:24s} -> {hdr.name}  ({c_bytes/1024:.1f} kB of C source)")
    except Exception as e:
        rec = {"model": name, "error": f"{type(e).__name__}: {e}",
               "estimator": type(joblib.load(jf)).__name__ if Path(jf).exists() else "?"}
        print(f"FAIL {name:24s} {rec['error']}")
    report.append(rec)

(out / "emlearn_report.json").write_text(json.dumps(report, indent=2))
print(f"\nWrote {len(report)} entries to {out/'emlearn_report.json'}")
print("Note: C source size is an upper bound proxy; true flash and RAM cost is measured on the ESP32.")
